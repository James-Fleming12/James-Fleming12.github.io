---
title: Building up to FlashAttention-2
description: Covering both FlashAttention and FlashAttention-2
pubDate: 2/24/2025
---
[FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691) are some of the most notable improvements in recents years that have allowed Large Language Models to be scaled to as large as they are today. They mark the most significant efficiency improvement in both training and inference and are used in almost every LLM that people use today. DeepSeek even went as far as to make specific implementation decisions to guide the model towards using it. This will cover both FlashAttention and the changes made in FlashAttention-2 (the one DeepSeek uses) but will skip an overview of the Scaled Dot Product Attention or Multi Head Attention that it is designed to replace, although there is a small overview of it within my overview of [[RWKV]].

### FlashAttention:
FlashAttention is an alternative to standard Dot Product Attention that produces the exact same outputs given the input with improvements on the algorithm's memory efficiency. The main performance bottleneck of standard attention comes in how it interacts with memory due to the scale of information it has to manage and perform calculations on. For this overview not much knowledge of low level I/O computer architecture is needed, but the basic idea that HBM (CPU memory) is slower but can hold more and SRAM (GPU memory) is much faster but can hold much less will be crucial. Since standard attention performs on the entire token sequence at once, it is required that it is performed on the much slower HBM.

The main idea behind FlashAttention comes in a form of tiling. The idea is to break up the sequence into smaller groups of "tiles" each holding less information. If the attention can find a way to break its algorithm up into these smaller pieces, it would be able to perform them on the much faster SRAM, allowing the algorithm to receive very notable speed improvements.

#### Forward Pass:
FlashAttention breaks the sequence up into blocks of size $B_r\times B_c$ where $B_c=\lfloor M/4d\rfloor$ is the column block size and $B_r=\min(\lfloor M/4d\rfloor,d)$ is the row block size (with $M$ on-chip memory and head dimension $d$). In order to get to this point the Queries $Q$, Keys $K$, and Values $V$ have to be calculated first just like usual. Once they are, $Q$ is broken up into $Q_1,\dots,Q_{T_r}$ (each of size $B_r\times d$) and $K$ and $V$ are broken up into $K_1,\dots,K_{T_c}$ and $V_1,\dots,V_{T_c}$ respectively (each of size $B_c\times d$). The algorithm iterates through all $i$ and $j$ to get information from each attention score $S_{ij}$ with which it adds some calculations to it's running summary of the sequence. For this post I will be shortening this to $S^{(j)}$ for the $j$-th tile we are performing on to simplify the notation. I will use the following definitions of $S^{(1)}$ to denote the first tile and $S^{(2)}$ to denote the next (which will be continued for each subsequent tile). As well we will need to define some $V^{(1)},V^{(2)}\in\mathbb{R}^{B_c\times d}$ which are derived in the same way.
$$
\begin{gather*}
S^{(1)}=Q(K^{(1)})^T\in\mathbb{R}^{B_r\times B_c}\\
S^{(2)}=Q(K^{(2)})^T\in\mathbb{R}^{B_r\times B_c}
\end{gather*}
$$
At the very start of the algorithm, some local computations need to be performed on the first tile. The algorithm has running values $m^{(j)}$ and $\ell^{(j)}$ for normalization. These are used to compute the exponentiated score for this tile $\tilde{P}^{(1)}$.
$$
\begin{gather*}
m^{(1)}=\text{rowmax}(S^{(1)})\in\mathbb{R}^{B_r}\\
\ell^{(1)}=\text{rowsum}(e^{S^{(1)}-m^{(1)}})\in\mathbb{R}^{B_r}\\
\tilde{P}^{(1)}=\text{diag}(\ell^{(1)})^{-1}e^{S^{(1)}-m^{(1)}}\in\mathbb{R}^{B_r\times B_c}
\end{gather*}
$$
These are then used to compute another running value which acts as the algorithm's current partial output. This calculation is very simple for the first tile but the method in which the information is accumulated in this output will be shown for the second tile.
$$
O^{(1)}=\tilde{P}^{(1)}V^{(1)}
$$
We then move on to the next tile in the sequence. The values of $m^{(1)}$, $\ell^{(1)}$, and $O^{(1)}$ will be regularly used in this calculation. First we perform similar calculations to get new normalization values and a new exponentiated score. I will be calling each of the new values for these simply $m$ and $\ell$ since they are now going to be a running value and the old values will be the only ones denoted.
$$
\begin{gather*}
m^{(2)}=m=\max(m^{(1)},\text{rowmax}(S^{(2)}))\\
\ell^{(2)}=\ell=e^{m^{(1)}-m}\ell^{(1)}+\text{rowsum}(e^{S^{(1)}-m})\\
\tilde{P}^{(2)}=\text{diag}(\ell)^{-1}e^{S^{(2)}-m}
\end{gather*}
$$
Once these are calculated we then calculate a new partial output $O^{(2)}$. This same procedure we denoted for tile 2 will then be repeated again, with this tile being treated as tile $S^{(1)}$ and its output being treated as $O^{(1)}$.
$$
O^{(2)}=\text{diag}(\ell^{(1)}/\ell)^{-1}O^{(1)}+\tilde{P}^{(2)}V^{(2)}
$$
Once each tile is ran through, the final output $O$ will be the exact same as if a scaled dot-product attention calculation was performed on the entire sequence. This form of tiling allows the information within each tile to be placed entirely on SRAM, which allows the computer to perform each step much quicker than it otherwise would on HBM, even though the algorithm is much more complex to human eyes. This algorithm is then parallelized over the batch size and each attention head to achieve to make it even more efficient.

This also has efficiency improvements for the backward pass during training. It is shown that both $S$ and $P$ can be very simply calculated if the rest of the information ($Q$, $K$, and $V$ for each layer and $m^{(j)}$ and $l^{(j)}$ for the $j$-th tile at that layer) is stored in memory, so we are able to save the memory that would otherwise be spent on keeping them for the backward pass at very little computational cost. This is a significant change even though it may not seem like it due to their size of each of $S$ and $P$ and their nature of mere intermediate matrices. The specificities of the backward pass will be shown later for FlashAttention-2 since it also makes a couple improvements on the calculations.
### FlashAttention-2:
FlashAttention-2 makes some seemingly small but very powerful improvements to the model. It gets a lot deeper into the architectural knowledge required for some of the changes made, especially those about workload balancing, but I consider those specific implementation details and I will be skipping over them. For FlashAttention-2 there are two main changes to the algorithm detailed above.

First, the method in which we compute the partial output $O^{(j)}$ has it's calculation split in two. It removes the need to perform the normalization calculation twice, which are much more computationally heavy on modern GPUs in comparison to the matrix multplications.
$$
\begin{gather*}
\tilde{O}^{(2)}=\text{diag}(\ell^{(1)})^{-1}O^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)}\\
O^{(2)}=\text{diag}(\ell^{(2)})^{-1}\tilde{O}^{(2)}
\end{gather*}
$$
Second, instead of saving both $m$ and $\ell$ in memory we save one collection value $L$ for each tile. This doubles down on the goal to limit the amount of interaction with HBM during training.
$$
L^{(j)}=m^{(j)}+\log(\ell^{(j)})
$$
As well one of the architectural changes that improves the algorithm the most comes in it's improved parallelism. As a reminder the first algorithm only parallelized over the batch and over each attention head. An additional sequence level parallelization is implemented in FlashAttention-2. Even though the algorithm is sequential, the calculations that locally stay within the tile can be performed ahead of time. This allows the algorithm to move faster through the sequence since a majority of the computation will already be down per tile. This is one of the most important changes to the low level design of the algorithm because it allowed the attention to extend to much longer sequences, which is one of the if not the biggest weakpoint of almost every attention algorithm.

#### Backward Pass:
The backward pass for the model also uses this same tiling system to compute the gradient, in which the gradients are computed in tiles to use the same GPU based speedups that the forward pass has. The backward pass uses the same tiles as the forward pass did. As stated above the only information stored for this pass is $Q$, $K$, and $V$ for each layer and $L_i$ and $O$ for each tile within that layer. As well the derivative in respect to the output $dO$ is simply derived from the specified loss function of the model. First both $S$ and $P$ are recalculated for the given tile.
$$
\begin{gather*}
S^{(j)}_i=Q_iK_j^T\in\mathbb{R}^{B_r\times B_c}\\
P_j^{(j)}=\exp(S_{ij}-L_i)\in\mathbb{R}^{B_c\times d}
\end{gather*}
$$
This allows each of $dS$ and $dP$ to be calculated very simply since they are per-tile values. They use another value $D=\text{rowsum}(dO\circ O)\in\mathbb{R^d}$ in their calculations.
$$
\begin{gather}
dS_i^{(j)}=P_i^{(j)}\circ(dP_i^{(j)}-D_i)\\
dP_i^{(j)}=dO_iV_j^T
\end{gather}
$$
Since each of $Q$, $K$, and $V$ are not per-tile values but for the entire sequence and are calculated as such, their derivatives need to take the entire sequence into account. This uses the same running summarization strategy as the forward pass and calculates a partial output for each derivative $dQ$, $dK$, and $dV$ as the backward pass is running through each tile. Once each tile is done the final state of each of these is then gradient used.
$$
\begin{gather*}
dV_j\leftarrow dV_j+(P_i^{(j)})^TdO_i\\
dQ_i\leftarrow dQ_i+dS_i^{(j)}K_j\\
dK_j\leftarrow dK_j+{dS_i^{(j)}}^TQ_i
\end{gather*}
$$