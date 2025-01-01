---
title: A Tiny Overview of Diffusion-LM
description: A Model of Diffusion made for Language Generation
pubDate: 12/31/2024
---
Diffusion-LM (introduced in the [paper](https://arxiv.org/pdf/2205.14217) of the same name) is simply a method of extending Diffusion Models to Language Generation tasks. The paper does not so much as introduce an entire new architecture but rather introduces a method of building future architectures. Many of the model decisions are left up to interpretation and specific use scenarios. This post isn't going to cover everything super far in depth, especially the training process, and only acts as an introductory step to learn of the concepts that the model introduces.

The main importance of the paper and the model it introduces does not lie simply in the ability to perform diffusion-based text generation, but in a break away from the mold that most text generation is placed in. Most language generation models are known as Auto Regressive Models, meaning they generate their output sequentially and use the previous parts of the output as added context to the current generation. This is the model that Transformers and Large Language Models use and has been shown to provide great results but also has it's own problems, mostly in knowing the context of the future parts of the sequence before they are generated. Diffusion-LM provides an extensible and flexible method of generating these output sequences in parallel, which even without any added backbones or the such can already encode relationships between words that are generating together. Given an extra guidance model suited for the task it needs to perform it can correctly encode these relationships and build context in the sentence in any order it sees fit.

To understand Diffusion-LM one must first understand the base concept of what a Diffusion Model is. In the most simplistic of terms a Diffusion Model learns to take noise and turn it into something meaningful. This is done through training in a two step process. In the first step (Foward Process), noise is added to some input and in the second (Backward Process), the model attempts to denoise the input to get to the original input in a typically recursive process. These models then can be given a patch of random noise and still be able to generate something noticeable. Most models then have some form of outer model to help guide or condition the diffusion process to intake user input or allow further logic to be added.

Diffusion-LM works by denoising a set of random noise vectors into a set of what they call word vectors. During training it follows the same principles of adding noise to word vectors and then trying to reconstruct the original data. The main changes are introduced in the connection between the discrete representation of the text and the continuous need of Diffusion. Diffusion needs the input to be in a continuous vector space in order to perform the computations it needs on it. This is done with two main contributions.

1. Embedding (turning words into embedded vectors)
2. Rounding (turning embedded vectors into words)

### Embedding:
The Embedding process is the simpler of the two, as it resembles that of Word Embeddings in LLMs. The Embedding step encodes each word into learned word vectors that can then have computations performed on it. These are learned in tandem with the diffusion parameters of the model. For a sequence $\textbf{w}$ of length $n$ each word is mapped to a vector of $\mathbb{R}^d$.
$$
\text{EMB}(\textbf{w})=[\text{EMB}(w_1),\dots,\text{EMB}(w_n)]\in R^{nd}
$$

### Rounding:
The process of turning each word vector back into a word seems relatively simple on the outset. A simple argmax of the probability of each word given the vector should be enough, but in practice each vector does not commit to words hard enough for it to be a foolproof solution. Rather, the model introduces another Neural Network to interpret each vector. Defined by $f_\theta(x_t,t)$ that, given some noisy data $x_t$ and a time step $t$, will predict the word vector $x_0$. This also introduces a revamped objective function which measures the performance of the model during training. 
$$
\mathcal{L}^{\text{e2e}}_{x_0\text{-simple}}(x_0)=\sum^T_{t=1}\mathbb{E}_{x_t}||f_{\theta}(x_t,t)-x_0||^2
$$
An understanding of the equation is not necessary for this post, but rather just seeing that the output of the neural network is used to measure the accuracy of the model. This fixes a majority of the problem of vectors not commiting to words and encodes it directly into the diffusion process. 

This can then be improved with something they call the clamping trick. As the neural network is used both during the diffusion steps and decoding, it is clamped to the nearest known word vector. This creates a scenario in which the vector commits at the end of the diffusion process, but in between steps as well. Each diffusion step is defined below with some $\beta_s$ that defines the behavior and speed of the noise reduction in the network. Again it is not necessary to understand the equations they are simply there to showcase how the clamping works at each diffusion step.
$$
\begin{gather*}
x_{t-1}=\sqrt{\bar{\alpha}}\cdot\text{Clamp}(f_\theta(x_t,t))+\sqrt{1-\bar{\alpha}}\epsilon\\
\bar{\alpha}_t=\prod^t_{s=0}(1-\beta_s)
\end{gather*}
$$

Both of these combined create a method of diffusion that can work with and create language. This barely scratches the surface of the model, and again I am not going to be covering how the model is trained or the extensive explorations of controlling the text generation because that section of the model is meant to be changed. The specific guidance model is what changes the behavior of the model from something that works to generate random sentences to something that can respond to input of any modality. These typically come in the form of Transformers within the tests provided, and aren't even needed for some unique test cases. They can also be extended with length estimators and any such network, Diffusion-LM represents something that is asking to be extended and molded to whatever use case it needs to provide.