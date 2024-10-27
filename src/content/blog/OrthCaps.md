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
The Pruned Capsule Layer serves to remove any redundant or otherwise unneeded capsules based on similarity to other capsules. When two capsules share similar data, they theoretically represent the same object, which not only consumes more computational power but also serves to confuse and undermine other data from other capsules. The layer removes these capsules based on their [cosine similarity](https://www.sciencedirect.com/topics/computer-science/cosine-similarity), a measure of the similarity between two vectors based on the angle between them. A smaller angle between them, the more similar they are. This is then combined with another metric of importance, ...

### Sparse Attention Routing:
