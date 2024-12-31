---
title: OrthCaps Research Paper
description: OrthCaps Paper Review
pubDate: 10/15/2024
---
Found in the [research paper](https://arxiv.org/abs/2403.13351) of the same name, an Orthogonal Capsule Network is a novel [CapsNet](https://james-fleming12.github.io/stuff/capsnet/) architecture that aims to "reduce redundancy, improve routing perforamnce, and decrease parameter counts". The paper tackles these challenges with three main new contributions. All three concepts rely heavily on the concept of ensuring that capsules represent different objects without any observable overlap and remain consistent in that.
1. Capsule Pruning (automatically removing similar capsules during runtime)
2. Orthogonal Sparse Attention Routing (a replacement of Dynamic Routing)
3. Orthogonalization (Householder Orthogonalization to ensure the previous qualities from pruning are kept)

### Architecture:
The paper offers two different models that use and implement the afformentioned concepts in different computation levels. OrthCaps-Shallow (called OrthCaps-S) and OrthCaps-Deep (called OrthCaps-D). The two share almost all of the same model features with their sole differences being in the intermediate Convolutional Capsule layers which will be described shortly. Both models can be broken up into 5 subsections not including the input layer and the ClassCaps layer (which simply acts as the output).
1. Convolutional Layer (preprocessing)
2. PrimaryCaps (turns the previous layer into capsules)
3. Pruned Capsule Layer (Covered in the [Capsule Pruning Section](#capsule-pruning))
4. Convolutional Capsules (With a special [Sparse Attention Routing](#sparse-attention-routing))
5. Flat Capsule Layer (simplifying the capsules to a single flat vector for interpretability)

The difference in the two models comes from not only the structure of the ConvCaps, but also in their routing mechanisms. They both use the same concepts, but the differences will be covered in a later chapter. For now the only difference that is relavent is in their structure. ConvCaps-S has a simple approach which only uses 0 to 2 capsules based on the dataset (0 for MNiST and 2 for CIFAR10). ConvCaps-D has a system of 7 capsule blocks each with 3 capsule layers. These are ...

### Capsule Pruning:
The Pruned Capsule Layer serves to remove any redundant or otherwise unneeded capsules based on similarity to other capsules. When two capsules share similar data, they theoretically represent the same object, which not only consumes more computational power but also serves to confuse and undermine other data from other capsules. The layer removes these capsules based on their [cosine similarity](https://www.sciencedirect.com/topics/computer-science/cosine-similarity), a measure of the similarity between two vectors based on the angle between them. A smaller angle between them, the more similar they are. This is then combined with another metric of importance, which is measured through the L2-Norm (defined as $||x||_2=\sqrt{\sum_{i=1}^n|x_i|^2}$)

This is optimized with the use of a mask matrix, a type of matrix that is used to eliminate the data out of certain sections of another matrix or vector. While iterating through each capsule and checking similarity, if two capsules are deemed to similar, the column correlating to the less important capsule is set to 0. This effictively prunes the capsule and removes it's information from the network

### Sparse Attention Routing:
Now comes the main differences between both model types mentioned previously. Their routing mechanism, influenced by the original Dynamic Routing procedures of typical Capsule Networks, removes the need for iteration completely in dynamic routing. They use concepts from [attention](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/), mainly with the concept of Queries $Q$, keys $K$, and values $V$ to separate the different values of the model. I didn't care much about this section of the paper since it was not my main priority so I am not going to be covering it here.

### Orthogonalization:
In order to ensure that the orthogonalized nature of the network that is captured in previous layers is kept constant throughout the network's progression, extra work is needed. The model uses the concept of and the properties of Orthogonal Matrices, any matrix that is the same as it's Transpose. When multiplied to spaces these represent rotations or reflections. Most notably, the angle between different vectors remains the exact same before and after the matrix is applied. The below equation represents the model progression at each layer, and since the activation function $g$ keeps the angles of each vector constant, only the $C$ and $W_V$ matrices need to be orthoganalized
$$
V_{l+1}=g(S_l)=g(CV)=g[(C\times W_V)U_l]
$$
The Routing Coefficient Matrix $C$ is hard so a work around is proposed in the below equation. It uses the $\alpha$-Entmax function defined within this [paper](https://arxiv.org/abs/1905.05702) for Sequence models. In order to get the point not much is needed to be known about the function except for the fact that it is nonlinear. This means that the angles it generates are technically not going to be the exact same, but the paper proves that the function itself renders $C$ sparse. This encourages the same behavior of retaining low similarity between capsules without needing go through extra steps to orthogonalize the matrix directly.
$$
\begin{gather*}
C=\alpha\text{-Entmax}(QK^T/\sqrt{d})=\alpha\text{-Entmax}(W_QU_lU_l^TW_K^T/\sqrt{d})\\
\alpha\text{-EntMax}(x_i)=\max\left(\frac{x_i-\tau}{\alpha},0\right)^{\frac{1}{\alpha-1}}
\end{gather*}
$$
The Weight Matrices $W_V$ are properly diagonalized using the properties of Householder Transformations, Transformations with the form $I-2vv^T$ that describe reflections about a plane. Any orthogonal $n\times n$ matrix can be defined using the product of at most $n$ Householder Transformations. This means that if you train the network to rather learn these transformations instead of the matrix itself, it will keep itself orthogonalized. This is done with a set of learnable $b_i$ column vectors that are randomly generated at the beginning of the training process.
$$
W=\prod_{i=0}^{d-1}\left(I-\frac{2b_ib_i^T}{||b_i||^2}\right)
$$