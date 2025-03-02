---
title: Building up to Vision Mamba
description: An alternative to ViTs using the Mamba Architecture in Computer Vision
pubDate: 03/01/2025
---
[Vision Mamba](https://arxiv.org/abs/2401.09417) and [Mamba](https://arxiv.org/pdf/2312.00752) present a very important alternative to Transformers when dealing with block-based sequence models. Although not as popular or present in top commercial models they offer something almost invaluable in the field, something other than attention that can perform the same task, which I personally have been dying to see get any form of coverage whatsoever. This post will cover State Space Models, Structured State Space Sequence Models, the general Mamba architecture, and the extension of it to computer vision in Vision Mamba.

# State Space Models:
The architecture of Mamba is built from the ground up based on the concept of State Space Models, often shortened to SSMs. They simply represent a common way in Probability and Control Theory to represent a dynamic system with a set of continuous state variables. The general form for these models is described below with two distinct sections in a recurrent system. The hidden state $z_t$ (the internal state of the system) and the observation $y_t$ (the output of the system).
$$
\begin{gather*}
z_t=g(u_t,z_{t-1},\epsilon_t)\\
y_t=h(z_t,u_t,\delta_t)
\end{gather*}
$$
$u_t$ represents an optional input or control signal and $\epsilon_t$ and $\delta_t$ are the system and observation noise at time $t$ respectively. Due to the generalities of the representation this can be decomposed into any number of different models. One of the simplest forms comes in with the Linear-Gaussian SSM that uses the representation to create a model with parameters $\theta_t=(A_t,B_t,C_t,D_t,Q_t,R_t)$.
$$
\begin{gather*}
z_t=A_tz_{t-1}+B_tu_t+\epsilon_t\\
y_t=C_tz_t+D_tu_t+\delta_t\\
\epsilon\sim\mathcal{N}(0,Q_t)\\
\delta_t\sim\mathcal{N}(0,R_t)
\end{gather*}
$$

# Structured State Space Sequence Models:
These models are then extended into their most popular form by Structured State Space Sequence Models, often shortened S4 Models. These are such an important version of the model in the AI space that when trying to search up anything about the underlying SSM you will still find information about S4 models, and even the Mamba paper itself calls S4 models SSMs for brevity. They are still represented by a form of continuous system described in variables, but the main improvements of the model come in the computations performed with them.
$$
\begin{gather*}
\frac{dh(t)}{dt}=Ah(t)+Bx(t)\\
y(t)=Ch(t)
\end{gather*}
$$
## Discretization:
One of the most important steps to an S4 model is the discretization step. The model transforms the continuous variables $\frac{dh(t)}{dt}$ of the model into discrete variables $h(t)$. This improves the stability and computational efficiency but more importantly moves the model towards real world scenarios in which the data used is often discrete. This is generalized to any fixed formula pair $(f_A,f_B)$ called the discretization rule that transform the continuous variables $(\Delta,A,B)$ into discrete variables $(\bar{A},\bar{B})$.
$$
\begin{gather*}
h_t=\bar{A}h_{t-1}+\bar{B}x_t\\
\bar{A}=f_A(\Delta,A)\\
\bar{B}=f_B(\Delta,A,B)
\end{gather*}
$$
One of the most important properties of these systems is that both the continuous and discrete variables are all time invariant, as in fixed for all time-step. This allows for parallel computation, precomputation, and even leads to better theoretical convergence. The most common discretization rule and the one that will be most important for this use case will be the zero-order hold, often shortened to ZOH. It assumes that the input $x(t)$ is constant between time steps and is very computationally efficient.
$$
\begin{gather*}
\bar{A}=\exp(\Delta A)\\
\bar{B}=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B
\end{gather*}
$$
## Computation:
These discrete variables are then used in the main computational step aptly called computation. There are two main computation options proposed in the model each with their own benefits and use cases. Linear Recurrence (shown in the first column) is better used for autoregressive tasks where the input is taken in parts, most often during inference. Global Convolution (shown in the second column) is used in situations where the entire sequence can be seen of time, most often during training.
$$
\begin{gather*}
h_t=\bar{A}h_{t-1}+\bar{B}x_t&\bar{K}=(C\bar{B},C\bar{AB},\dots,C\bar{A}^k\bar{B},\dots)\\
y_t=Ch(t)&y=x*\bar{K}
\end{gather*}
$$

# Mamba:
In order to make this architecture scalable and to make it more flexible to the input data, Mamba introduces a repeatable block structure that is built off of S4 models. The architecture for each block is very simple to understand but I do highly encourage looking at the diagrams within the [paper on page 8](https://arxiv.org/abs/2312.00752) for a visual backing to the explanation (that diagram is also the only one so far that will provide any meaningful help with understanding the topic across all the models explained here). Each block is composed of the following sequential parts where both the input and output receive linear projections to match dimensions. As well the nonlinearity section of the block is given a skip connection from the input after its processed with a separate linear projection.

1. Convolutional Layer
2. Activation (chosen as Swish or SiLU in the original paper)
3. Selective State Space Model
4. Nonlinearity (activation or multiplication)

## Selective State Space Models:
Mamba's main contribution comes in how it extends the S4 models covered previously. The model makes some of the state variables dependent on the input to the model. This creates a system that is more malleable and can provide zero-shot capabilities. Specifically each of $(\Delta,B,C)$ are defined with functions of the inputs instead of being the weights by themselves. In the original model each function takes the form of the following with generalized functions $(s_B,s_C,s_\Delta,\tau_\Delta)$.
$$
\begin{gather*}
B=s_B(x)\\
C=s_C(x)\\
\Delta=\tau_\Delta(\beta+s_\Delta(x))
\end{gather*}
$$
The rest of the model stays the same as the above defined linear recurrence S4 model. Each of the functions are then defined with the following. The $\text{Broadcast}_D$ function broadcasts the input across a $D$-dimensional feature space.
$$
\begin{gather*}
s_B(x)=W_B\cdot x+b_B\\
s_C(x)=W_C\cdot x+b_C\\
s_\Delta(x)=\text{Broadcast}_D(W_\Delta\cdot x+b_\Delta)\\
\tau_\Delta(z)=\log(1+\exp(z))
\end{gather*}
$$

# Vision Mamba:
The original Mamba architecture works well for autoregressive tasks where information can be accumulated over the sequence, but struggles when given other modalities. Vision Mamba extends the architecture to general Computer Vision tasks in a similar way that Vision Transformers did for Transformers. The architecture of the model can be broken down into 3 parts.

1. Image Preprocessing
2. Vision Mamba Encoder
3. MLP Prediction Head

## Image Pre-Processing:
Following the lead of the Vision Transformer architecture, the input image is turned into a set of equal size patches. Specifically each image $t\in\mathbb{R}^{H\times W\times C}$ is turned into a set of 2D patches $x_p\in\mathbb{R}^{J\times(P^2\cdot C)}$ where $(H,W)$ is the image size, $C$ is the number of channels, and $P$ is the patch size. Each patch is then linearly projected and given positional embeddings $E_\text{pos}\in\mathbb{R}^{(J+1)\times D}$. This is described below for the $j$-th patch of $t$ $t^j_p$ and a weight matrix $W\in\mathbb{R}^{(P^2\cdot C)\times D}$ to generate the input $T_0$ (which also includes a class token $t_\text{cls}$ which is used later for the final output).
$$
T_0=[t_\text{cls};t^1_pW;t^2_pW;\dots;t^J_pW]+E_\text{pos}
$$
## Vim Block:
The extension of the general Mamba Block to the specialized Vision Mamba (shortened to Vim) Block is used to reformat the dimensionality of the Mamba Blocks (the original 1D sequence based Mamba Block is not well suited for images) and introduce a bidirectional mechanism (transferring information both forwards and backwards). First, the previous sequence is normalized and linearly projected producing $x,z\in\mathbb{R}^{B\times M\times E}$ and $T^\prime_{l-1}\in\mathbb{R}^{B\times M\times D}$ shown below (where $E$ is the expanded state dimension and $D$ is the hidden state dimension).
$$
\begin{gather*}
T^\prime_{l-1}=\text{Norm}(T_{l-1})\\
x=W_x\cdot T^\prime_{l-1}\\
z=W_z\cdot T^\prime_{l-1}
\end{gather*}
$$
After this processing is finished, the bidirectional mechanism starts. The following all repeat twice, once for each of $o\in\{\text{forward},\text{backward}\}$ which change the direction in which the information is processed creating a bidirectional exchange of information. As a first step linear projections of $x$ are produced $B_o,C_o\in\mathbb{R}^{B\times M\times N}$ where $N$ denotes the SSM dimension. A time step calculation $\Delta_o\in\mathbb{R}^{B\times M\times E}$ is also calculated using the same information.
$$
\begin{gather*}
x^\prime_o=\text{SiLU}(\text{Conv1D}_o(x))\\
B_o=W_{B_o}\cdot x^\prime_o\\
C_o=W_{C_o}\cdot x^\prime_o\\
\Delta_o=\log(1+\exp(W_{\Delta_o}\cdot x^\prime_o+b_{\Delta_o}))
\end{gather*}
$$
This time step calculation is used in the discretization step to compute the state transition matrices. Specifically this computes both $\bar{A_o},\bar{B_o}\in\mathbb{R}^{B\times M\times E\times N}$.
$$
\begin{gather*}
\overline{A_o}=\Delta_o\otimes W_{A_o}\\
\overline{B_o}=\Delta_o\otimes B_o
\end{gather*}
$$
Once discretized, an extension of the linear recurrence model from the original S4 model is used and shown below. This iterates over $0\leq i\leq M-1$ and each of $h_o$ and $y_o$ are intialized to have $0$ values.
$$
\begin{gather*}
h_o=\overline{A_o}[:,i,:,:]\odot h_o+\overline{B_o}[:,i,:,:]\odot x^\prime_o[:,i,:,\text{None}]\\\
y_o[:,i,:]=h_o\odot C_o[:,i,:]
\end{gather*}
$$
Once both outputs for each direction are computed they are combined and produced as an output $T_l$.
$$
\begin{gather*}
y^\prime_\text{forward}=y_\text{forward}\odot\text{SiLU}(z)\\
y^\prime_\text{backward}=y_\text{backward}\text{SiLU}(z)\\
T_l=W_T\cdot(y^\prime_\text{forward}+y^\prime_\text{backward})+T_{l-1}
\end{gather*}
$$
## Prediction Head:
Once each Vim Block is used to process the information, the class token $t_\text{cls}$ is used to generalize the output of the model. The token is normalized before being passed to the final MLP layer to produce the prediction output $\hat{p}$.
$$
\begin{gather*}
T_l=\text{Vim}(T_{l-1})+T_{l-1}\\
f=\text{Norm}(T^0_L)\\
\hat{p}=\text{MLP}(f)
\end{gather*}
$$