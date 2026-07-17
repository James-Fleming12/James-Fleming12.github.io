---
title: Isotropic Spaces and HDC
description: Why Isotropic Spaces are important for HDC to learn, why CNNs learn isotropic spaces, and why more contextual models like transformers learn anisotropic spaces.
pubDate: 6/16/2026
---

This is a quick overview that I wanted to write to understand the types of feature spaces that Hyperdimensional Computing models learn. This all boils down to the idea that HDC models (mostly their projections) need uniform feature spaces, and more complex loss functions and architectures that are popular in modern AI/ML work learn feature representations that are too contextual, leading to collapse when applied to a standard HDC model.

This will cover the background necessary for the analysis, a demonstration of why HDC needs the structure that anisotropic spaces have, and also a quick overview of some theoretical results that show that CNNs learn isotropic spaces (which is why they work so well with HDC) and that transformers learn anisotropic spaces (which is why the don't work with HDC).

## What are Isotropic and Anisotropic Spaces
Before we get into the analysis itself, we need to define the different types of feature spaces that we will be talking about.

At a high level, the isotropy of a feature space describes how much variance is distributed across its dimension. Isotropic Spaces are directionally uniform. If you map the feature vectors into a high-dimensional space, they form a roughly symmetrical hypersphere, meaning no single direction holds significantly more information (variance) than any other. Anisotropic spaces are directionally skewed. The data collapses into a hyperellipsoid or a narrow cone, meaning a few dominant directions capture almost all the variance, rendering the remaining dimensions effectively useless.

Formally, we define these spaces through the lens of their covariance structure. Let $x \in \mathbb{R}^d$ be a random vector representing a feature embedding, assuming it is mean-centered such that $\mathbb{E}[x] = 0$. The geometry of the space is governed by its covariance matrix $\Sigma = \mathbb{E}[xx^T]$. Since $\Sigma$ is symmetric and positive semi-definite, it admits an eigendecomposition $\Sigma = U\Lambda U^T$, where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$ contains the eigenvalues sorted such that $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0$.

A space is perfectly isotropic if its covariance matrix is proportional to the identity matrix, that is if $\Sigma = \sigma^2 I_d$, meaning $\lambda_1 = \lambda_2 = \dots = \lambda_d = \sigma^2$ (every orthogonal basis vector contributes equally to the total variance). A space is anisotropic if the spectrum of $\Lambda$ is skewed. The degree of anisotropy is rigorously quantified by the condition number of the covariance matrix, $\kappa = \lambda_1 / \lambda_d$, or by observing rapid decay in the cumulative spectral energy, meaning $\sum^k_{i=1}\lambda_i \approx \text{Tr}(\Sigma)$ for a subspace dimension $k \ll d$.

For the upcoming theorems (specifically regarding the JL lemma and deep learning architectures), we need two specific, operational definitions of anisotropy that appear in the literature.

In the context of HDC and the Johnson-Lindenstrauss Lemma, the robustness of random projections depends on the data not occupying a degenerate lower-dimensional subspace. This is measured via effective dimensionality $D_\text{eff}$ defined below.
$$
D_\text{eff}=\frac{(\text{Tr}(\Sigma))^2}{\text{Tr}(\Sigma^2)}=\frac{(\sum^d_{i=1}\lambda_i)^2}{\sum^d_{i=1}\lambda^2_i}
$$
For a perfectly isotropic space, $D_\text{eff}=d$. As the space becomes increasingly anisotropic (e.g. as the domainant eigenvalue $\lambda_1\gg \lambda_i$), $D_\text{eff}\rightarrow 1$. The Johnson-Lindenstrauss lemma's distance-preserving bounds degrade heavily as $D_\text{eff}$ drops relative to $d$ (as will be formalized later).

In the context of representation degradation in Transformers, anisotropy is most rigorously evaluated by the expectation of cosine similarity between independent feature vectors. If we draw two independent representations $x$ and $y$ from the dataset, we measure this as $A(x,y)$ shown below.
$$
A(x,y)=\mathbb{E}\left[\frac{x^Ty}{\|x\|_2\|y\|_2}\right]
$$
In a fully isotropic space, orthogonal directions dominate in high dimensions, and this expectation strictly approaches $0$. In an anisotropic space, the mean vector shifts away from the origin and representations cluster into a narrow cone, causing $A(x,y)\gg 0$. This specific formulation is what proves fatal when piping Transformer outputs into HDC models (as will be formalized later).

## Why Hyperdimensional Computing needs Isotropic Feature Spaces
Hyperdimensional Computing relies on random projections to map data into a high-dimensional space, typically $D\approx 10,000$. The theoretical justification for why this random mapping preserves semantic information lies in the Johnson-Lindenstrauss Lemma. The JL lemma states that for any set of $N$ points $X\subset\mathbb{R}^d$, and for any $0<\epsilon<1$, there exists a linear mapping $f:\mathbb{R}^d\rightarrow\mathbb{R}^D$ (where $D\geq\mathcal{O}(\epsilon^{-2}\log N)$) such that all pairwise Euclidean distances are preserved within a factor of $1\pm\epsilon$, formalized as follows.
$$
(1-\epsilon)\|u-v\|^2_2\leq\|f(u)-f(v)\|^2_2\leq(1+\epsilon)\|u-v\|^2_2\quad\forall u,v\in X
$$
In practice, $f$ is implemented as a random projection matrix $W\in\mathbb{R}^{D\times d}$ where entries are sampled from an isotropic distribution (e.g. $W_{ij}\sim\mathcal{N}(0,1/D)$).

The JL lemma guarantees that relative distances are preserved, but it makes no guarantees about the absolute scale of those distances relative to the noise floor of the system, which is where anisotropy breaks HDC. In an anisotropic space, the effective dimensionality $D_\text{eff}\ll d$. The dataset is dominated by a principle component vector $c$, meaning any two distinct feature vectors $u$ and $v$ can be decomposed into the large shared component and a tiny orthogonal signal component, formalized below where $\|c\|\gg \|\delta_u\|,\|\delta_v\|$.
$$
u=c+\delta_u\qquad v=c+\delta_v
$$
Because the random projection $W$ is linear, projecting these vectors yields the following results.
$$
Wu=Wc+W\delta_u\qquad Wv=Wc+W\delta_v
$$
While the JL lemma technically preserves the tiny distance $\|W\delta_u-W\delta_v\|\approx\|\delta_u-\delta_v\|$, the representations $Wu$ and $Wv$ are overwhelmingly dominated by the shared $Wc$ vector, so their cosine similarity in the projected space approaches $1$.
$$
\frac{(Wu)^T(Wv)}{\|Wu\|\|Wv\|}\approx\frac{\|Wc\|^2}{\|Wc\|^2}=1
$$
This geometric scale is fatal for HDC because HDC models do not just operate on continuous random projections, but instead also heavily rely on low-precision quantizations (which will be bipolarization here) to achieve computational efficiency. The standard HDC encoding step applies a sign function to the projection.
$$
h_u=\text{sgn}(Wu)=\text{sgn}(Wc+W\delta_u)
$$
Because $\|c\|\geq\|\delta_u\|$, the sign of the summation is almost entirely determined by $Wc$. The distinguishing signal $\delta_u$ acts as negligible noise that fails to flip the sign boundary. Because HDC lacks backpropagation to adjust $W$ and "learn" to scale up the minor distinguishing features, the model is fundamentally blind to them. Isotropic spaces are strictly required by definition to have no single dimension dominate the pre-quantization projection, leading to robust feature spaces that work under HDC pipelines.

## Convolutional Neural Networks learn Isotropic Features
The paper [The Singular Values of Convolutional Layers](https://arxiv.org/abs/1805.10408) (Sedghi et al., 2018) provides a rigorous framework showing that the preservation of isotropy in Convolutional Neural Networks is a direct consequence of their architectural inductive biases. Specifically, the weight-sharing mechanism and the constraint of local receptive fields prevent the representation collapse observed in global attention mechanisms.

### The Convolutional Operator as a Toeplitz Matrix
A convolutional layer can be represented as a linear operator $C(w)$ acting on the input feature map $X \in \mathbb{R}^d$. If we consider a 1D convolution with a filter $w \in \mathbb{R}^k$, the operation $X * w$ is equivalent to matrix-vector multiplication $T_w X$, where $T_w$ is a Toeplitz matrix defined by $T_{i,j} = w_{i-j}$. If we assume circular boundaries (padding), $T_w$ becomes a circulant matrix. A fundamental property of circulant matrices is that they are diagonalized by the Discrete Fourier Transform (DFT) matrix $F$, formalized as follows where $\hat{w}$ is the Fourier transform of the filter $w$, and $F^*$ is the conjugate transpose of $F$.
$$
T_w = F \text{diag}(\hat{w}) F^*
$$
If the input $X$ has a stationary covariance structure diagonalizable by $F$ (i.e., $\Sigma_X = F \text{diag}(s_X) F^*$), the covariance after linear convolution is exactly
$$
\Sigma_{\text{out}} = F \text{diag}(|\hat{w}|^2 \odot s_X) F^*
$$
This rigorously proves that the variance is modulated only by the filter's frequency magnitude $|\hat{w}|^2$, preventing the arbitrary dense mixing that leads to rank collapse in global attention mechanisms. The inductive bias here is translational invariance, meaning the same local, band-limited operation applies regardless of position.

### Spectral Stability and Rank Preservation
Theoretical analysis shows that the spectral properties of CNNs avoid anisotropy through two primary mechanisms.

1. **Filter Localization and Bounded Singular Values:** Because the filter $w$ has local support ($k \ll d$), its Fourier transform $\hat{w}$ cannot be infinitely sharp due to the uncertainty principle of the Fourier transform. Therefore, the singular values of $C(w)$, which are bounded by the maximum frequency magnitude $\|\hat{w}\|_\infty$, remain relatively uniform. There is no mathematical mechanism in a local, band-limited filter to push a single singular value to dominate the entire matrix, effectively upper-bounding the condition number $\kappa(\Sigma)$ and keeping the spectrum of the resulting feature representation flat.
2. **Avoidance of Global Normalization:** In Transformers, the Softmax bottleneck forces the attention distribution to sharpen, eventually concentrating weight on a few dominant tokens, which is the geometric origin of the "coning" effect. In CNNs, there is no global Softmax over the spatial or feature dimensions in the layer operation itself. The interaction is limited strictly to the local neighborhood.

Formally, consider the covariance $\Sigma^{(l)}$ at layer $l$. In modern deep networks, we must account for the non-linear activation function $\sigma(\cdot)$, meaning the covariance update between neighboring layers can be defined as follows.
$$
\Sigma^{(l+1)} = \mathbb{E}[\sigma(C(w_l)X_l)\sigma(C(w_l)X_l)^T]
$$
For homogeneous activation functions like ReLU (where $\sigma(cx) = c\sigma(x)$ for $c > 0$), the non-linearity attenuates the variance by a constant factor but preserves the diagonal dominance of the underlying covariance matrix. Because $C(w_l)$ is a localized, translation-invariant operator, it does not possess the massive singular values that characterize global attention matrices. Consequently, the condition number of $\Sigma^{(l)}$ does not grow exponentially with depth $l$ as it does in Transformers. The variance remains distributed across the spectral components of the image, preserving the effective dimensionality $D_{\text{eff}}$ described in the previous section.

### Consequences
In summary, the paper confirms that the CNN architecture functions as a stationary kernel. A kernel $K(x, x')$ is stationary if it depends only on the shift between inputs, i.e. $K(x, x') = K(x - x')$. Because convolutions are translationally invariant, the expected dot product between two shifted patches relies purely on their relative distance. This spatial stationarity mathematically guarantees that the feature space does not collapse into a single preferred spatial location or dominant direction. By avoiding the global coupling of features (which invariably introduces bias toward the "most frequent" patterns in the training data) CNNs maintain an isotropic distribution where the signal-to-noise ratio is preserved across the entire dimensionality of the manifold.

## Transformers learn Anisotropic Features

While Convolutional Neural Networks preserve isotropy through localized, stationary operations, Transformers are driven by a mechanism that does the exact opposite.

The paper [Anisotropy Is Inhernet to Self-Attention in Trasformers](https://arxiv.org/abs/2401.12143) provides a rigorous theoretical foundation showing that the representation degeneration problem, often referred to as the "coning effect", is not an artifact of the specific loss functions or datasets, but a fundamental geometric consequence of the self-attention operator itself.

### Self-Attention as a Row-Stochastic Contraction
To understand the geometric collapse, we model the standard self-attention mechanism. Given a sequence of $N$ token embeddings $X \in \mathbb{R}^{N \times d}$, and projection matrices for queries and keys $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, the attention matrix $A \in \mathbb{R}^{N \times N}$ is computed as follows.
$$
A = \text{Softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right) = \text{Softmax}\left(\frac{X W_Q W_K^T X^T}{\sqrt{d_k}}\right)
$$
Crucially, the softmax function is applied row-wise. This guarantees that $A$ is a strictly positive matrix where every row sums exactly to $1$ (meaning $A\textbf{1} = \textbf{1}$). The output of the attention head is then $Z = AXW_V$. Because $A$ is a row-stochastic matrix, the operation $AXW_V$ means that every output token representation $z_i$ is a convex combination of the value vectors. Geometrically, taking convex combinations of a set of points pulls all resulting points strictly inside the convex hull of the original set. Over successive layers, this acts as a low-pass filter or a contraction mapping, causing the pairwise distances between token representations to strictly decrease.

### Perron-Frobenius and Directional Collapse
Because the attention matrix $A$ has strictly positive entries, the Perron-Frobenius theorem dictates its spectral properties. The matrix has a unique, strictly positive dominant eigenvalue of $\lambda_1=1$, associated with the constant eigenvector $\textbf{1}$. All other eigenvalues are strictly less than $1$ in absolute value.

When this operator is applied iteratively across the depth $L$ of a Transformer, the spectral components associated with the smaller eigenvalues decay exponentially. The representations are inevitably pulled toward the dominant eigenvector. As $L\rightarrow\infty$, the token representations converge to a rank-1 state, formalized below where $c\in\mathbb{R}^d$ is a single, dominant contextual vector.
$$
\lim_{L\rightarrow\infty} X^{(l)}=\textbf{1}c^T
$$
In practice with finite layers, the space doesn't collapse entirely to rank-1, but the covariance matrix $\Sigma$ becomes massively skewed, so a single dominant eigenvalue $\lambda_1\gg\lambda_i$ emerges.

### The Shifted Mean
This contraction does not pull the representations toward the origin, but rather toward an arbitrary nonzero vector. Modern Transformer MLPs rely on non-negative or one-sided activation functions (e.g., ReLU, GeLU). Because the attention matrix applies a strictly positive convex combination over these activated states (and because standard architectures lack a mechanism to strictly re-center the mean to zero post-attention) the expected value of the embeddings drifts into the positive orthant. Consequently, the mean vector of the embeddings shifts heavily away from the origin ($\|\mathbb{E}[x]\| \gg 0$). 

When the mean is far from the origin and the variance is iteratively squeezed into a dominant direction by the row-stochastic attention matrix, the feature space forms a narrow cone. Recall the cosine anisotropy formulation from the first section:
$$
A(x,y) = \mathbb{E}\left[\frac{x^T y}{\|x\|_2 \|y\|_2}\right]
$$
As the vectors structurally cluster around the shifted mean vector $c$, the angle between any two independent representations $x$ and $y$ approaches zero. This mathematical skew causes the expected cosine similarity to approach $1$, effectively destroying the spatial orthogonality.

### Consequences
In summary, the transformer's inductive bias is designed to globally route information and merge contexts, which mathematically necessitates a highly anisotropic space ($D_\text{eff}\rightarrow 1$). If these skewed representations are passed into an HDC random projection matrix, the guarantees from the JL lemma fail (as demonstrated in the earlier mathematical example). The distinguishing features of the tokens are completely overwhelmed by the dominant spatial cone, causing catastrophic aliasing when the hypervectors are quantized.

## Re-Isotropicizing Transformer Representations

We have established a fundamental incompatibility. Transformers act as a contraction mapping that funnels representations into an anisotropic cone, while HDC requires a perfectly isotropic space for the Johnson-Lindenstrauss random projections to preserve distinct concepts upon quantization.

To bridge this gap and utilize powerful contextual embeddings within HDC frameworks, we must actively intervene to break the coning effect. This is typically done through two primary mechanisms: post-hoc geometric transformations or representation-level loss constraints.

### Post-Hoc Whitening Transformations
If we freeze a pre-trained Transformer, we can apply a linear transformation to its output space to explicitly flatten the variance before piping the vectors into the HDC projection matrix. This is known as a whitening transformation.

Given a dataset of Transformer embeddings $X \in \mathbb{R}^{N \times d}$, we first compute the empirical mean $\mu = \frac{1}{N}\sum_{i=1}^N x_i$ and the empirical covariance matrix:
$$
\hat{\Sigma} = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
$$
A whitening matrix $W_{\text{white}}$ is derived such that $W_{\text{white}}^T W_{\text{white}} = \hat{\Sigma}^{-1}$. Using the ZCA (Zero-phase Component Analysis) formulation, this is computed via the eigendecomposition $\hat{\Sigma} = U \Lambda U^T$:
$$
W_{\text{ZCA}} = U \Lambda^{-1/2} U^T
$$
By applying this transformation to our centered embeddings ($x_{\text{iso}} = W_{\text{ZCA}}(x - \mu)$) we force the new covariance matrix to equal the identity matrix ($\Sigma_{\text{iso}} = I_d$). The narrow cone is mathematically stretched back into a perfect hypersphere. When $x_{\text{iso}}$ is passed to the HDC projection matrix, the random projections will successfully capture distinguishing features because the effective dimensionality has been restored to $D_{\text{eff}} = d$.

### Contrastive Learning and Uniformity
Rather than fixing the space after training, we can alter the Transformer's optimization objective to resist the self-attention contraction. While standard cross-entropy loss does nothing to prevent anisotropy, contrastive learning frameworks (like the InfoNCE loss used in CLIP or SimCSE) explicitly optimize for spatial uniformity.

Contrastive loss operates on the unit hypersphere (enforced via $L_2$ normalization). It maximizes the similarity of positive pairs $(x, x^+)$ while explicitly minimizing the cosine similarity of negative pairs $(x, x^-)$:
$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(x, x^+)/\tau)}{\sum_{j=1}^K \exp(\text{sim}(x, x^-_j)/\tau)}
$$
Theoretical analysis of contrastive learning shows that as the number of negative samples $K \to \infty$, the loss strongly penalizes clustering. It acts as an opposing force to the Perron-Frobenius contraction of the attention matrix, pushing independent representations to be highly orthogonal. This restores the condition $A(x,y) \approx 0$ for independent samples, naturally generating the isotropic space that HDC requires to function without catastrophic aliasing.

### Conclusion
Hyperdimensional Computing is not fundamentally incompatible with modern deep learning, but it is geometrically rigid. By understanding the spectral properties of the architectures we use, leveraging the natural stationarity of CNNs, or explicitly correcting the row-stochastic collapse of Transformers, we can design hybrid models that capitalize on both deep contextual understanding and highly efficient, robust symbolic reasoning.