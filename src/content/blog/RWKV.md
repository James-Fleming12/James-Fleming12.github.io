---
title: An Overview of RWKV
description: An RNN made for modern language processing
pubDate: 02/14/2025
---
Transformers have shown the most potential when it comes to natural language processing and generation as shown by the dominance in the field by large language models and other such models. This has gone so far that most of the modern research for language processing lays solely in trying to further improve these models, which leaves alternatives very rare and very underdeveloped in comparison to the newest LLMs. This leaves many of the problems inherent to Transformers to be standard and left unquestioned, one of the most notable being the computational complexity of Attention. Advancements like [FlastAttention](https://arxiv.org/abs/2205.14135) make the problem less noticeable, but these models still have a large hurdle to climb once larger sequences are given. [RWKV]((https://arxiv.org/abs/2305.13048)) tries to tackle this problem by moving language processing back to it's recurrent neural network roots while still maintaining all the strengths of Transformers when it comes to training efficiency and accuracy.

### Standard Attention:
The standard attention used in Transformers is a combination of Scaled Dot Product Attention and Multi-Head Attention. Multi-Head Attention is not as important to know for RWKV, as it just produces multiple Scaled Dot Product Attention branches to enable more features to be understood. Scaled Dot Product Attention is the main culprit of the computational complexity of the model. Given some input $X$, the model produces three representations $Q$ (Queries), $K$ (Keys), and $V$ (Values) out of it. This is done through a set of linear projections (most often being a small sub-network).
$$
\begin{gather*}
q_t=W^Qh_t\\
k_t=W^Kh_t\\
v_t=W^Vh_t
\end{gather*}
$$
These are then used in the main attention algorithm shown below. You do not need to understand the use of it or beyond just knowing that it is used to share information between tokens of the input sequence.
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_{k}}}\right)V=\frac{\sum^T_{i=1}e^{q_t^T k_i}\odot v_i}{\sum^T_{i=1}e^{q_t^T k_i}}
$$
This algorithm has a quadratic time complexity $O(n^2)$ (since each token needs to perform calculations on all the other tokens) and since the input sequence is the text input into the model this creates problems once the input needs to be something larger than a paragraph.

## RWKV:
RWKV (standing for Receptance Weighted Key Value) is a modern Recurrent Neural Network tailored to emulate the abilities of Transformers witout the Attention mechanisms that limit the size of their input sequences. Just like any other RNN, RWKV is a single network that receives a single word input and outputs a prediction for the next word with the information of the past parts of the sequence being brought in by the calculations within. The architecture of the network has it's blocks surrounded by LayerNorm and Residual Connections. The final output is determined by a LM Head that takes the final model output and gives the probability of each word in the vocabulary. The main contribution of the model are the blocks which are as follows.

1. Time-Mixing (incorporating past information to process the current input)
2. Channel-Mixing (using current information to process further)

Each of these blocks work off of 4 fundamental elements that are present within both blocks. In the same way that the linear projections of traditional attention do not guarentee they have the intended meaning, these do the same, however their general significances are important to understand the derivation of each calculation.

1. The receptance vector $R$ (the receiver of past information)
2. The weight $W$ (the learnable positional weight decay vector)
3. The key vector $K$ (analogous to the key values of traditional attention)
4. The value vector $V$ (similar to the the values of traditional attention)

### Time-Mixing:
The Time-Mixing block marks the start of the network and acts to bring in information from past time steps to the current. Instead of $Q$, $K$, and $V$ this block splits the input into $r_t$, $k_t$ and $v_t$ as described below. It mixes the information from the current input to the block $x_t$ and the previous $x_{t-1}$ in a process called Token Shift with a learned mixing parameter $\mu$ (known as the Token Shifting Parameter). 
$$
\begin{gather*}
r_t=W_r\cdot(\mu_r\odot x_t+(1-\mu_r)\odot x_{t-1})\\
k_t=W_k\cdot(\mu_k\odot x_t+(1-\mu_k)\odot x_{t-1})\\
v_t=W_v\cdot(\mu_v\odot x_t+(1-\mu_v)\odot x_{t-1})
\end{gather*}
$$
The $k_t$ and $v_t$ values are then used to calculate the $WKV$ operator which acts as a running representation of the sequence. The calculation for it shown below (with learned parameters for time-decay $w$ and bias $u$) is done so that there is an inherent exponential decay in the while still having their information be incorporated. As well the algorithm is formatted so that the calculations done for the previous timestep can be reused in the current.
$$
wkv_t=\frac{\sum^{t-1}_{i=1}e^{-(t-1-i)w+k_i}\odot v_i+e^{u+k_t}\odot v_t}{\sum^{t-1}_{i=1}e^{-(t-1-i)w+k_i}+e^{u+k_t}}
$$
This is then recombined with the $r_t$ value after it is ran through a sigmoid $\sigma(\cdot)$ to obtain the final output $o_t$.
$$
o_t=W_o\cdot(\sigma(r_t)\odot wkv_t)
$$

### Channel-Mixing:
The Channel-Mixing block does the same general thing that the Time-Mixing block does but strays away from the information that the $WKV$ operator passes through the block and instead focuses on the information already processed. The block relies on two linear projections $r^\prime_t$ and $k^\prime_t$
$$
\begin{gather*}
r^\prime_t=W^\prime_r\cdot(\mu^\prime_r\odot x_t+(1-\mu_r^\prime)\odot x_{t-1})\\
k^\prime_t=W^\prime_k\cdot(\mu^\prime_k\odot x_t+(1-\mu_k^\prime)\odot x_{t-1})
\end{gather*}
$$
This then skips straight to the output calculation $o^\prime_t$. The main computational change comes in the squared ReLU of the key vector $k^\prime_t$ within the calculation.
$$
o^\prime_t=\sigma(r^\prime_t)\odot(W^\prime_v\cdot\max(k^\prime_t,0)^2)
$$

The paper then goes into further detail to prove the efficiency and stability of the model along with some optimizations specific to their implementation, but the general architecture stays simple and concise. This leaves room for further extension to try and tackle the problems inherent to any kind of gated recurrent structure.
