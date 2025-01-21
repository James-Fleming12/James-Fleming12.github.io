---
title: Space-Time Correspondence Networks
description: An Architecture for Live Multi-Class Video Segmentation
pubDate: 1/17/25
---
[Space-Time Correspondence Networks](https://arxiv.org/abs/2106.05210) represent a model of neural network like live multi-object video segmentation networks, meaning they take the current frame of the footage and use the past frames and masks to process the current with more accuracy. Although not the newest or most promising kind of model, I do think they serve a very important purpose of straying away from the Vision Transformer laden environment that Computer Vision has gotten into. These models still work off of an Encoder Decoder structure but, although having it's roots int he concepts, strays away from any form of Attention mechanism. It instead opts for a CNN style view at generating keys and values, which will be covered later. This post will also cover [Space-Time Memory Networks](https://arxiv.org/abs/1904.00607) being the predecessor to the above mentioned models only generally being suited for single object segmentation.

## Space-Time Memory Networks:
Space-Time Memory Networks, as mentioned above, are a type of Neural Network made by Video Object Segmentation (often shortened simply to VOS). They take the current frame as an input and hold in memory all the past frames and their respective outputted masks which are both used in conjuction to produce the current frame's mask. The overall architecture can be split up into 3 parts.
1. Embeddings (generated for both Query and Memory)
2. Space-Time Memory Read (processes the embeddings)
3. Decoder (gives the final output mask)

### Embedding:
First, the information present in both the Query (current) and Memory (past) frames needs to be transformed into a workable format. This is done through a process of encoder embedding which generates a set of matrices to represent each frame. This is split up into a Key (represents general information) and Value (represents in-depth information) maps. The only main difference between the processing for both Query and Memory comes in the number of inputs for each, with Memory having both the Frame and the Mask output from the model previously, as well as the fact that Memory Embeddings can be processed earlier and used from memory. These are both processed using some form of parallel Convolutional Layer, one outputting the keys and one outputting the values, with [ResNet50](https://arxiv.org/abs/1512.03385) being chosen in the paper. Each memory frame and it's respective mask are concatenated along the channel dimension before processing, and after processing each map type is concatenated into two Key and Value matrices representing every memory frame. The final embeddings are a pair of 2D maps for the Query Embedding $(k^Q\in\mathbb{R}^{H\times W\times C/8},v^Q\in\mathbb{R}^{H\times W\times C/2})$ and a pair of 3D maps for the Memory Embedding $(k^M\in\mathbb{R}^{T\times H\times W\times C/8},v^M\in\mathbb{R}^{T\times H\times W\times C/2})$

### Space-Time Memory Read:
Once the Embeddings are generated, they then need to be compared. This is where a typical Attention mechanism would be put into place, but Space-Time Memory Networks rather use a system of matrix multiplications, softmaxes, and concatenations, which more so resemble a typical Neural Network. The general algorithm is described below with a normalizing factor $Z$ and a similarity function $f$. First, the both key maps are multipled and passed through a Softmax to generate probabilities, which are then multiplied to the Memory value map. At the very end, $[\cdot,\cdot]$ represents a concatenation to the Query value map.
$$
\begin{gather*}
y_i=[v_i^Q,\frac{1}{Z}\sum_{\forall_j}f(\textbf{k}_i^Q,\textbf{k}^M_j)\textbf{v}_j^M]\\
Z=\sum_{\forall_j}f(\textbf{k}_i^Q,\textbf{k}^M_j)\\
f(\textbf{k}_i^Q,\textbf{k}^M_j)=\exp(\textbf{k}_i^Q\cdot\textbf{k}^M_j)
\end{gather*}
$$

### Decoder:
The final stage of the model simply performs some decompressions (a form of reverse Convolution) on the above generated matrix. This is chosen to be the refinement module from [SharpMask](https://arxiv.org/abs/1603.08695), but can really be any form of deconvolutional network. The only step of preprocessing is by compressing it back down to $256$ channels from a previous denoted $C$ channels decided by the Embedding network with a simple Convolutional Layer and skip connection. The refinement module gradually upscales the feature map with the input to each module being the previous steps output feature map as well as the feature map of the same scale from the Query Encoder, which ensures that the final output segments the objects within the image given. Once upscaled through the entire network, the final output is then passed through an additional Convolutional Layer and a Softmax to generate the mask, which is represented by a 2 channel output (each representing the probabilities for foreground and background features).

## Space-Time Correspondence Networks:
Although formatted specifically for single-object segmentation, Space-Time Memory Networks also showed promise for multi-object segmentation by simply processing the same information multiple times for each object within the image. This, however, is wildly inefficient and where Space-Time Correspondence Networks come in. They introduce changes to the Embedding and Memory Read modules to fit a multi-object modality better, as well as introducing some general efficiency improvements.

### Embedding:
The main change in the new model comes in how Key and Value maps are generated. The Query Embeddings stay as is, but the Key Embeddings are generated without the mask, having it represent the general information of the image as a whole. This allows each frame's key map to be processed only once, instead of once for each object within the image, as well as allowing the Query key maps to be reused as Memory key maps. This also makes some changes to how values are generated with a system of feature reuse. The last layer features of both the Key map and each Value map are concatenated and passed through two ResBlocks and a CBAM block to generate the final Value maps.

The model also introduces a system of efficient memory management where every fifth query frame is considered a memory frame, with the key map and to be processed query maps being saved for that frame. This follows the general structure of Space-Time Memory Networks, however strays away when it comes to the use of the most recent frame as temporary memory. It was found to be almost detrimental to use this frame in the current processing due to how similar both the key maps would be for the given frames, a direct consequence from using shared key encoders.

### Memory Reading and Decoding:
Since the Query frame no longer has a Key map, the method of Memory Reading has to be changed. It also needs to be altered to fully take advantage of the fact that each object class shares a key map, which can cut down on the computation significantly. The process is shown below for some pairwise affinity matrix $S$ and a general affinity matrix $W$ (which is normalized by $\sqrt{C^k}$ but not shown in the equations), which both can be reused for every value map generated. The final output being a simple multiplication of $W$ to the value map of the given class.
$$
\begin{gather*}
S_{ij}=c(k_i^M,k_j^Q)\\
W_{ij}=\frac{\exp(S_{ij})}{\sum_n(\exp(S_{nj}))}\\
v^Q=v^M W
\end{gather*}
$$
The pairwise similarity function $c$ can be defined as any function $c:\mathbb{R}^{C^k}\times\mathbb{R}^{C^k}\rightarrow\mathbb{R}$ that gives a scalar value from two vectors, which really should denote the similarity between them. The dot product used in the original Space-Time Memory Networks can be used but two novel functions are tested and optimized within the model. The [Cosine Similarity](https://www.geeksforgeeks.org/cosine-similarity/) and the [Euclidean Distance](https://www.geeksforgeeks.org/cosine-similarity/) are introduced as options, being defined below. Each optimization shown in the paper have their own strengths and weaknesses and I am not going to be covering them here, but the general idea that $c$ can be any simlarity function is what the main idea is.
$$
\begin{gather*}
S^{\cos}_{ij}=\frac{k^M_i\cdot k^Q_j}{||k_i^M||_2\times||k^Q_j||_2}\\
S^{\text{L2}}_{ij}=-||k_i^M=k_j^Q||^2_2
\end{gather*}
$$