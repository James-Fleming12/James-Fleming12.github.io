---
title: Capsule Network Overview
description: A basic survey of a basic capsule network
pubDate: 10/13/2024
---
A powerful and 
### Previous Problems:
Two main problems existed in the implementation and very basis of neural networks for computer vision
1. Pooling works, but it loses so much information that it shouldn't
2. Objects at different orientations are treated as different objects

Capsule Networks attempt to solve these problems with a 
### Capsules:
...
### Dynamic Routing:
The original [Capsule Network paper](https://arxiv.org/abs/1710.09829) was a simple exploration of the idea. Due to it being tailored for the MNiST dataset, it only needed the barebone architecture necessary to get the network running. It consisted of the following 3 layers not including the Input layer: 
1. Convolutional Layer
2. PrimaryCaps
3. DigitCaps

The Convolutional Layer speaks for itself if you know anything about neural networks. It is just a convolutional layer that acts as preprocessing for the PrimaryCaps layer so it has more dense information to handle. The main logic changes come once the PrimaryCaps layer is reached. The PrimaryCaps layer learns how to take the information from the convolutional layer and turn it into a set of neurons. 
### EM Routing:
...
