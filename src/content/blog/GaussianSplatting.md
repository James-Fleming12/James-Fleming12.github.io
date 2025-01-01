---
title: Gaussian Splatting Overview
description: A Method of Novel View Synthesis based on Gaussian Distributions
pubDate: 1/01/25
---
Gaussian Splatting presents a model for generating novel views of a scene given a number of pictures or videos of said scene. It does so by constructing a new 3D representation of the scene made up of a set of 3D gaussians. This post will cover a comprehensive overview of the model and also includes a shallow look at the most popular previous method of novel view synthesis to understand Gaussian Splatting's contributions. This [video](https://youtu.be/VkIJbpdTujE?si=DzVAgB9yuRfWDDDZ) by Computerphile is a very good starting point for understanding both the basic nature of Gaussian Splatting as well as the strengths of the model, but this post will serve to go more in depth on the model without going so in detail that I go insane.

### Neural Radiance Fields (NeRFs):
NeRFs represented the defacto method of novel view synthesis before Gaussian Splatting was introduced. They work off of a method similar to [Ray Tracing](https://developer.nvidia.com/discover/ray-tracing), where each ray is "shot" using a Neural Network trained on the input images. This means that the neural network was ran for every single pixel on the screen whenever the viewpoint was changed, creating a very inefficient system. As well, the nature of the model made it so that any small changes to the scene it was trying to reconstruct would require an entire retraining of the network. This is where Gaussians come in. Gaussian Splatting renders novel views in a method of [3D Rasterization](https://www.cs.princeton.edu/courses/archive/spring20/cos426/lectures/Lecture-14.pdf), which allows it to be approximately 50 times faster. As well, since it creates a new 3D representation of the scene, you can move or even remove pieces of the scene without requiring a full retraining of the model. This creates not only a more efficient method, but also one that is more extensible and flexible to developer needs.

### 3D Gaussians:
Gaussian Splatting generates scenes based off of a set of 3D Gaussian Distributions. A typical Gaussian Distribution (also called a Normal Distribution) is a probability distribution that represents a bell curve and is the typical distribution that one can visualize. Once extended to a 2D Multivariate Gaussian Distribution these represent 2D hills of probability, and one can imagine that once extended to 3D these represent oval-like structures. Instead of representing probabilities these are simply to represent 3D objects and are used due to the differentiability of their functions and explicit nature which makes them great candidates for what the model needs. They are defined by a 3D [Covariance Matrix](https://www.geeksforgeeks.org/covariance-matrix/) $\Sigma$ and are defined below. 
$$
G(x)=e^{-\frac{1}{2}(x)^T\Sigma^{-1}(x)}
$$
Along with $\Sigma$, the Gaussian also has the properties of position $p$, opacity $\alpha$, and Spherical Harmonic Coefficients for it's color $c$. The use of SH Coefficients allows the Gaussian to be different varying colors at different viewing angles which also allows it to capture lighting effects. The Covariance Matrix controls the basic geometric properties of the Gaussian (skew, scale, etc.) and is then decomposed into the product of 4 matrices to further simplify updating and optimizing them. The decomposition is defined below with some scaling matrix $S$ and some rotation matrix $R$
$$
\Sigma=RSS^TR^T
$$
Then once the Gaussians are projected onto the camera, they simply need to generate a new Covariance Matrix to represent the Gaussian in the viewspace. This is done through generating a new $\Sigma^\prime$ matrix below. A further exploration of the projection and rasterization process will be covered in the portion about the Tile-Based Rasterizer, but for basic knowledge this is also used with a system of $\alpha$-blending to mix the colors of different slightly transparent gaussians.
$$
\Sigma^\prime=JW\Sigma W^T J^T
$$

### Optimization:
The model works off of optimizing the given properties of each Gaussian to match the images given. The gaussians are first generated from a [Structure from Motion (SfM)](https://www.mathworks.com/help/vision/ug/what-is-structure-from-motion.html) point cloud, which acts as a good starting point but doesn't exactly perform optimally for reconstruction. This follows an iterative process of rendering and comparing the resulting image to the training views. This uses a process of Stochastic Gradient Descent (a simple subset of Gradient Descent based on performing the updates after each sample) to alter all the above mentioned qualities of each gaussian, with an activation function being placed on both $\alpha$ and $\Sigma$ to help maintain them in logical ranges. The Loss Function is defined below as a mix of $L_1$ (a simple difference between the representation and image) and $L_{\text{D-SSIM}}$ (standing for Differentiable Structural Similarity Index, which captures information that the $L_1$ loss misses) defined with the reconstructed image $I_x$ and the target image $I_y$ (as well as $C_1$ and $C_2$ being stabilization constants and $\text{D-SSIM}$ being done on $N$ patches on the image).
$$
\begin{gather*}
L=(1-\lambda)L_1+\lambda L_{\text{D-SSIM}}\\
L_1=\sum|I_x(x,y)-I_y(x,y)|\\
\text{SSIM}=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma^2_y+C_2)}\\
L_\text{D-SSIM}=1-\frac{1}{N}\sum_{i=1}^N\text{SSIM}_i(I_x,I_y)
\end{gather*}
$$

### Adaptive Density Control:
One of the biggest problems with the current systems is in the density of each Gaussian. Adaptive Density Control is a set of methods to help control the population of Gaussians within the representation both by adding and removing them where needed. Removing unneeded gaussians is a relatively simple process, and each 100 iterations any gaussians with a $\alpha$ below a threshold $\epsilon_\alpha$ are removed. This removes all Gaussians that are practically transparent. As well, at every 3000 iterations all the $\alpha$ values are set close to zero, which creates a scenarion in which only those that are necessary are returned back to their original state. 

Adding new Gaussians is a tad more complex and is broken up into two problem statements. Situations where a region is missing gaussians it needs (Under-Reconstruction) and situations where a gaussians is too large (Over-Reconstruction). Under-Reconstruction is solved by creating a clone of the nearest Gaussian and is detected by any form of high residuals generated from the loss. Over-Reconstruction is solved by splitting the larger gaussian into two, dividing their scale by $\phi=1.6$, and is detected by any form of structured negative residuals generated from the loss.

### Tile-Based Rasterizer:
The last major contribution made by the paper for the model comes in the form of it's method of Rasterization. It uses a tile-based rasterizer to help optimize both rasterization and $\alpha$-blending. The first step within it can be seen as a style of preprocessing the information present within the representation before rasterization. The process starts off by splitting the camera view into $16\times 16$ tiles. This will allow for further optimizations later based on each tile. As well, it follows simple 3D rendering techniques like backface culling (ignoring gaussians with normals that are pointing away from the camera) and ignoring any gaussians in extreme positions (gaussians outside of the viewport and those too close to the camera).

The Rasterization algorithm first starts by performing a [Radix Sort](https://www.geeksforgeeks.org/radix-sort/) of each Gaussian based on the View Space depth and Tile ID it belongs to. This sorted list can then be broken down into lists for each tile, which are inherently sorted by view depth (closest to furthest). The Rasterization then begins, with the $c$ and $\alpha$ values of each gaussian accumulating on the pixels it occupies. This process in of itself has many many optimizations to cover which I will not be doing, but the most important is in the parallelization of the process. Each tile is given it's own Thread, meaning that every single tile can be working at the same time, making the rasterization process much quicker.