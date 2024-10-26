---
title: Capsule Network Overview
description: A basic survey of a basic capsule network
pubDate: 10/13/2024
---
Capsule Networks represent a new paradigm for computer vision models to develop into something that resembles a logical view of the world for computers. The paths are endless with increased accuracy both theoretically and in practice at the expense of added computational complexity. This post will cover both the original Capsule Networks paper ([Dynamic Routing](https://arxiv.org/abs/1710.09829)) and the most prominent followup paper ([EM Routing](https://openreview.net/pdf?id=HJWLfGWRb))
### Previous Problems:
Two main problems existed in the implementation and very basis of neural networks in computer vision
1. Pooling works, but it loses so much information that it shouldn't
2. Objects at different orientations are treated as different objects

Capsule Networks attempt to solve these problems with a new look at how neural networks could work. Similar to methods of [Symbolic AI](https://www.datacamp.com/blog/what-is-symbolic-ai) in the past, instead of just having each neuron in theory represent the information at that location, each neuron should represent a part in the image. That could be a line, circle, arm, etc. basically anything that helps to break down the original image into smaller more distinguishable parts.
### Capsules:
The very backbone of a Capsule Network comes in how it's neurons are treated. In typical neural networks every neuron is represented by a scalar value. Each scalar value is theoretically supposed to represent the prescence of data at that point and together they are joined with probabilistic values to get a prediction. Capsule Networks rewire neurons to be represented by vectors. This allows them to represent more than just a prescence, as the vector can have it's dimensions represent orientation, scale, or really any value the algorithm sees fit.

Because of their nature, typical activation functions can't be applied to vectors in the manner they can to scalars. Thus each vector is oftened squased to a range of 0 to 1 representing it's activation. This allows each capsule to separate their activation away from the information it represents, further improving the information transfer within the network. The original paper uses a nonlinear squashing function that ensures the short vectors are shrunk to almost zero and long vectors are shrunk to almost 1.

### Original CapsNet:
The original Capsule Network [paper](https://arxiv.org/abs/1710.09829) was a simple exploration of the idea. Due to it being tailored for the MNiST dataset, it only needed the barebone architecture necessary to get the network running. It consisted of the following 3 layers not including the Input and Output layer, which were both typical input and output layers seen in previous MNiST models.
1. Convolutional Layer (basic representation)
2. PrimaryCaps (intermediate representation)
3. DigitCaps (complete representation)

The Convolutional Layer speaks for itself if you know anything about neural networks. It is just a convolutional layer that acts as preprocessing for the PrimaryCaps layer so it has more dense information to handle. The main logic changes come once the PrimaryCaps layer is reached. The PrimaryCaps layer learns how to take the information from the convolutional layer and turn it into a set of neurons. The convolutional layer therefore needs to capture very broad features from within the image. Within the paper they landed on 256 9 x 9 convolution kernels (i.e. capturing the information from a 9x9 block 256 times at each position in the image). This creates a 256 x A x B tensor, with A and B being a size based on the image, which is 20 x 20 in the original paper based on MNiST.

The transition from scalar values in the Convolutional layer to the vectors of a Capsule Network is handled by the PrimaryCaps layer. The transition is mostly for subsequent layers to have more information to work with, so the capsules themselves do not perform their intended use case of representing concepts in the objects yet. In order to transform the scalar quantities from it is as simple as applying the kernel more times to each location, in the paper a total of 8 to create an 8D vector for each stride. This results in 32 6 x 6 8D capsule tensors with the 6 x 6 dimensions coming from the previous tensor size.

The real magic behind the original capsule network implementation comes in the connections between the PrimaryCaps and the DigitCaps. The only connection where the custom routing mechanism is this one, but the mechanism will be covered in a later section. In simplistic terms the DigitCaps layer represents the output in a vector form. It is 10 16D vectors representing each digit found in the dataset and each vector receives data from each previous capsule. The activations from this layer then go on to decide the final output of the system.
### Dynamic Routing:
The connections between the capsule layers need to be unique. In order to incorporate the concept that the capsules represent specific features of the object, the entire basis of locality in the layers data needs to be removed. This is where the paper's routing mechanism comes in. It is based on a system of **routing by agreement**, meaning each capsule has dynamic connections to the next layer. The weight (how much data is sent between each neuron) is determined live by the state of the network. 

The lower-level features (the capsules from the previous layer) represent parts of the higher-level features (the capsules in the next layer). This means that the only way that a higher level feature exists is if the lower level features can agree on it. This is calculated through a iterative approach where a repetition of computing agreement scores (the numerical value for how much a capsule agrees with the consensus, $a_{ij}$) and altering routing coefficients (the dynamic weight for each connection, $c_{ij}$) is performed. The process can be thought of as running through the layer multiple times, changing the routing metrics each time to ensure a perfect transition of data to the next layer. First the total output of the lower-level capsule $s_j$ is calculated using prediction vectors $\hat{u}_{j|i}$ and a weight matrix $W_{ij}$, which is the learnable parameter of the predictions.
$$
\begin{gather*}
s_j=\sum_i c_{ij}\hat{u}_{i|j}\\
\hat{u}_{j|i}=W_{ij}u_i
\end{gather*}
$$
These are then used to produce the agreement scores (with $v_{ij}$ being the vector output of the higher-level capsule) and change the coupling coefficient. The coupling coefficient adds the agreement scores to its own original value $b_{ij}$ (often instantiated to 0). The collection of routing coefficients are also put under a [softmax](https://medium.com/@hunter-j-phillips/a-simple-introduction-to-softmax-287712d69bac) function to ensure they total up to 1.
$$
\begin{gather*}
v_j=\frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||}\\
a_{ij}=v_j\cdot \hat{u}_{j|i}\\
c_{ij}=\frac{\exp(b_{ij})}{\sum_k\exp(b_{ik})}
\end{gather*}
$$ 
These steps repeat a fixed number of times to ensure that a consensus can be reached in the layer. If the consensus is reached before many other models also introduce a convergence check that stops the iterations early if the layer has already reached a consensus.

### EM Routing:
...
