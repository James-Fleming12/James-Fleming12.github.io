---
title: Swim Transformer Overview
description: A Patch-based Vision Transformer Architecture
pubDate: 1/01/25
---
A Swim Transformer (abbreviated as Swim-T) is an extension of a typical [Vision Transformer](https://arxiv.org/abs/2010.11929) made to replicate the strengths of a [Convolutional Neural Network](https://www.geeksforgeeks.org/introduction-convolution-neural-network/). This post serves to cover the model architecture as well as to provide just the bare bones knowledge of both ViTs and CNNs to understand the concepts. It also won't act as an in-depth explantation and will more so be a surface level overview of the model, which is completely not my fault that's how much detail the paper went into itself and I'm not invested enough to read the source code.

### Convolutional Neural Network (CNN):
A Convolutional Neural Network represents the most basic and most popular forms of Neural Network based computer vision. The architecture is built off of the concept of the Convolutional Layer and Pooling Layer. The Convolutional Layer processes a collection of neurons at once and puts their collective information into one neuron in the next, and the Pooling Layer helps to create smaller representations of the previous layers. This creates spatially local patches of dense information that are used to represent the input image in a dimensionally reduced manner. This manner of gradual dimensionality reduction also allows each layer to be used as extra information in detection or segmentation tasks to provide finer results.

### Vision Transformer (ViT):
A Vision Transformer is an extension of the [Transformer](https://arxiv.org/abs/1706.03762) architecture for computer vision. The input image is embedded into separate patches with additional position embeddings before being processed. This then is processed using a system of global multi-head self attention based transformer blocks that process the entire image in parallel. The extension to Transformers not only allows computer vision models to integrate very simply with language systems but also to generalize themselves and become more broad in scope.

### Swim Transformer:
The Swim Transformer is a typical Vision Transformer based on the gradually increasing the amount of information each patch encapsulates. This is done through the same general image embedding and repetition of transformer blocks with significant changes being made to the blocks. The paper introduces the concept of a Swim Transformer Block, which contains all the new contributions of the paper. Each Block is lead by a Patch Merging layer except for the first which receives added Linear Embeddings. The Patches are merged 4 at a time and are all maintained with the same general structure. The biggest change to each block comes in how attention is handled. Each pair of two blocks go back and forth on the method of attention they use, with the first using W-MSA and the second using SW-MSA.

In order to stay true to the patch-based system promised by the Swim Transformer, the attention mechanism cannot be global anymore. It needs to stay contained within the patch itself and only within the patch itself. Each block contains both their respective attention mechanism and also a MultiLayer Perceptron (which uses [GELU](https://arxiv.org/abs/1606.08415) activations surprisingly) to store the main logical structures of the model, with each having a LayerNorm and skip connections in between like a standard transformer.
$$
\begin{gather*}
\hat{z}^l=\text{W-MSA}(\text{LN}(z^{l-1}))+z^{l-1}\\
z^l=\text{MLP}(\text{LN}(\hat{z}^l))+\hat{z}^l
\end{gather*}
$$
The above concepts work great in concept but lacks in the inter-patch connections that make the Vision Transformer so strong. The main interest point comes in the Shifted window partitioning introduced in the second layer. In order to maintain these connections between patches, each patch is shifted by half it's size to form brand new patches. This creates an intermingling system that can serves a purpose similar to that of a CNN with the performance of a ViT.
$$
\begin{gather*}
\hat{z}^{l+1}=\text{SW-MSA}(\text{LN}(z^l))+z^l\\
z^{l+1}=\text{MLP}(\text{LN}(\hat{z}^{l+1}))+\hat{z}^{l+1}
\end{gather*}
$$

Overall the architecture and paper itself are relatively underwhelming in comparison to the promise of the other papers I have covered, but the concept stays strong and interesting enough for me to have read it this much, so I guess that's my fault.