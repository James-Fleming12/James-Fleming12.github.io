---
title: Discrete Flow Matching
description: Building up to Discrete Flow Matching including overviews of Normalizing Flows, Flow Generative Models, and Flow Matching.
pubDate: 05/31/2025
---
Flow Generative Models and Flow Matching have been shown to be potent options for image and video generation, providing a more stable and smooth alternative to diffusion models. Rather than relying on random noise to form patterns, these models go back to the distribution-based roots of generative models and work to transform simpler distributions to those that encompass any given dataset. This form of generation was picked up by META to make Discrete Flow Matching, a new paradigm of language modeling that has similar strengths to similar diffusion-based language models. For this blog I am going to be explaining Discrete Flow Matching from the ground up in the simplest way possible, although each concept is very math heavy. Due to the reliance of each paper on a mathematical backing and their theoretical nature I won't be covering any concrete use cases or giving any architectural explanation since the concepts themselves are the main points I want to be taken away from this.

## Normalizing Flows:
A key problem in Probability Theory and Machine Learning is how to model complex distributions optimally. This problem arises most often when dealing with real data, which often does not follow valid patterns when complex enough. Variational Inference tackles this task by using simpler distributions, or tractable ones, to model the more complex one as closely as possible. The approximation is done through an optimization, with the exact nature of the optimization and the form in which the simpler distribution is represented being the key differences between different forms of Variational Inference.

The method of [Normalizing Flows](https://arxiv.org/abs/1505.05770) handles this approximation through a sequence of invertible transformations that turn the tractable distribution into the complex one. The simpler distribution is said to be the initial density which "flows" through the sequence of mappings to reach the desired distribution. When an invertible smooth transformation $f$ is used to transform a random variable $z$ with $z^\prime=f(z)$, its distribution $q(z)$ can also be mapped. This relationship is shown below with a Jacobian change of variables (where the inverse jacobian $\frac{\partial f^{-1}}{\partial z^\prime}$ is used since the adjustment in density depends on the inverse effect of the transformation).
$$
q(z^\prime)=q(z)\left|\det\frac{\partial f^{-1}}{\partial z^\prime}\right|=q(z)\left|\det\frac{\partial f}{\partial z}\right|^{-1}
$$
This relationship can then be extended to map a relationship between $z_0$ and its distribution $q_0$ to some distribution $q_K(z)$ through a series of $K$ transformations $f_k$. The path traversed by the random variables $z_k=f_k(z_{k-1})$ with initial distribution $q_0(z_0)$ is called the flow and the path formed by the successive transformations is called the normalizing flow. This is shown below in log space for numerical stability.
$$
\begin{gather*}
\ln q_K(z_K)=\ln q_0(z_0)-\sum^K_{k=1}\ln\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|\\
z_K=f_K\circ\dots\circ f_2\circ f_1(z_0)
\end{gather*}
$$
This method, specifically called a Finite Normalizing Flow, gives the flow the property of the law of unconscious statistician, where the expectations with respect to $q_K$ can be computed without knowing $q_K$ explicitly given an accurate choice of transformations. This allows information about the complex distributions to be approximated with the simpler one and the defined transformations.
$$
\mathbb{E}_{q_K}[h(z)]=\mathbb{E}_{q_0}[h(f_K\circ f_{K-1}\circ\dots\circ f_1(z_0))]
$$

## Flow Generative Models:
The goal of most generative models is to model a distribution rather than a function. These objectives may sound very similar, and they are, but the difference comes in the structure inherent to distributions. Having that continuous structure allows the model to generate data that doesn't match exactly with the function it has learned but is close to it, allowing it to generate new data.

Flow Generative Models use the concept of distribution modeling and normalizing flows to model complex distributions and generate new images and data. Given some unknown distribution $p_x(x)$, a simpler distribution $q_z(z)$, and a set of transformations $f_\theta=f_L\circ f_{L-1}\circ\dots\circ f_1$, a Flow Generative Model trains to maximize the log likelihood of the observed data (a system that penalizes the model for deviating from the unknown distribution).
$$
\begin{gather*}
\mathcal{L}(\theta)=\mathbb{E}_{x\sim p_\text{data}}[\log p_x(x;\theta)]\\
\mathcal{L}(\theta)=\mathbb{E}_{x\sim p_\text{data}}\left[\log p_z(f^{-1}_\theta(x))+\log\left|\det\frac{\partial f^{-1}_\theta(x)}{\partial x}\right|\right]
\end{gather*}
$$

## Flow Matching:
Another type of Normalizing Flows come in the form of Continuous Normalizing Flows. Instead of modeling the flow between the simpler distribution to the target one with a finite set of functions, a continuous normalizing flow represents the flow with a differential equation. The flow $\phi_t$ is defined in a differential equation with some vector field $v_t$.
$$
\begin{gather*}
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x))\\
\phi_o(x)=x
\end{gather*}
$$
The vector field is modeled as a neural network $v_t(x;\theta)$ with parameters $\theta$. Given the form of the model the form of the flow $\phi_t$ can be derived by solving the differential equation and can be used to transform some initial probability density $p_0$ to a target density $p_t$. This is done with the same change of variables operation being represented as $*$.
$$
\begin{gather*}
p_t=[\phi_t]_*p_0\\
[\phi_t]_*p_0(x)=p_0(\phi_t^{-1}(x))\det\left[\frac{\partial \phi_t^{-1}}{\partial x}\right]
\end{gather*}
$$
Flow Matching uses this definition to model a flow generative model. The model trains to approximate some target probability path $p_t(x)$, where $p_1=q$ with $q$ being the distribution of the data, that has a corresponding vector field $u_t(x)$. Since both probability paths can be derived directly from their respective vector fields, in order to simplify the calculations the vector fields themselves are compared instead of the paths.
$$
\mathcal{L}_\text{FM}(\theta)=\mathbb{E}_{t,p_t(x)}\|v_t(x)-u_t(x)\|^2
$$
Since the exact nature of the dataset's distribution is unknown, $p_t(x)$ and $u_t(x)$ have to be approximated from the known pieces of data. This is done by splitting both of them into paths, each of which correspond to a conditional probability. Each is the conditional probability path $p_t(x|x_1)$ scaled by the original probability $q(x_1)$, where $x$ is the latent state at time $t$ and $x_1$ is some ground truth datapoint of the dataset. These are defined with boundary rules where $p_0(x|x_1)=p(x)$ starts from some simple defined distribution and $p_1(x|x_1)$ is concentrated around $x=x_1$. For a typical image dataset, this could theoretically mean that at $t=0$ $x$ could be pure noise and at $t=1$ $x\approx x_1$ meaning it should be almost the original image.
$$
\begin{gather*}
p_t(x)=\int p_t(x|x_1)q(x_1)dx_1\\
p_1(x)=\int p_1(x|x_1)q(x_1)dx_1\approx q(x)
\end{gather*}
$$
This definition allows the path to be estimated through sampling the dataset, instead of needing to know the distribution outright. It also allows a better form for $u_t(x)$ to be derived through the above definition and a simplification with Bayes Theorem.
$$
u_t(x)=\int u_t(x|x_1)p_t(x_1|x)dx_1=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx
$$
This still leaves a problem within the model that forms intractable integrals. Since the original loss requires the integrated form $p_t(x)$ to solve for $u_t(x)$, every single part of the dataset needs to be integrated to form $p_t(x)$ with this setup, making it intractable. This is solved within the loss by simply using the conditional form of $u_t(x|x_1)$ instead with $x_1\sim q(x_1)$ (from the dataset) and $x\sim p_t(x|x_1)$ (from the conditional probability path), which is proven to have the same optima as the original, therefore being able to train $v_t$ to the same model.
$$
\mathcal{L}_\text{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\|v_t(x)-u_t(x|x_1)\|^2
$$

### Base Example:
The definition for Flow Matching allows for any choice of conditional probability path and conditional vector fields, but the example provided by the paper is the simplest option available, modeling them as a Gaussian Distribution. The parameters of the distribution $\mu:[0,1]\times\mathbb{R}^d\rightarrow\mathbb{R}^d$ and $\sigma:[0,1]\times\mathbb{R}\rightarrow\mathbb{R}_{>0}$ are made to be time-dependent and are used to model the desird probability path. At $t=0$, $\mu_0(x_1)=0$ and $\sigma_0(x_1)=1$ so that $p(x)=\mathcal{N}(x|0,I)$ and at $t=1$, $\mu_1(x_1)=x_1$ and $\sigma_1(x_1)=\sigma_\text{min}$, where $\sigma_\text{min}$ is set low enough so that $p_1(x|x_1)$ is concentrated and centered at $x_1$. This creates a path that transforms some simpler datapoint into a more complex datapoint within the desired distribution.
$$
p_t(x|x_1)=\mathcal{N}(x|\mu_t(x_1),\sigma_t(x_1)^2I)
$$
An infinite number of vector fields for any given probability path can be derived, but the simplest is chosen within the paper as an example. The below definition uses the same $x_1\sim q(x_1)$ and $x\sim p_t(x|x_1)$ from the CFM loss function and is conditioned on $x_1$.
$$
\begin{gather*}
\psi_t(x)=\sigma_t(x_1)x+\mu_t(x_1)\\
[\psi_t]_*p(x)=p_t(x|x_1)
\end{gather*}
$$
This flow can then derive the vector field that generates the conditional probability path $p_t(x|x_1)$ in terms of $\sigma_t(x)$ and $\mu_t(x)$, which then acts as the comparison for training $v_t$.
$$
\begin{gather*}
\frac{d}{dt}\psi_t(x)=u_t(\psi_t(x)|x_1)\\
u_t(x|x_1)=\frac{\sigma_t^\prime(x_1)}{\sigma_t(x_1)}(x-\mu_t(x_1))+\mu_t^\prime(x_1)
\end{gather*}
$$
The CFM loss can then be derived and simplified by reparameterizing $p_t(x|x_1)$  of $x_0$, the datapoint drawn from the simpler distribution that acts as the initial state of the flow.
$$
\mathcal{L}_\text{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\|v_t(\psi_t(x_0))-u_t(\psi_t(x_0)|x_1)\|^2
$$

## Discrete Flow Matching:
[Discrete Flow Matching](https://arxiv.org/abs/2407.15595) extends the format for generation show in Flow Matching to a discrete space, which is not as simple as it is for diffusion due to the nature of Flow Matching as something that follows a function. A lot of syntax and new definitions that differ from the original Flow Matching description are required to understand the explanation of the model. The model aims to model some sequence of tokens $x$ of $N$ elements $(x^1,x^2,\dots,x^N)$ each of size $d$. As well $\mathcal{D}$ is used to denote the entire set of possible sequences $\mathcal{D}=[d]^N$, where $[d]=\{1,\dots,d\}$. A random variable in the space of $\mathcal{D}$ is denoted by $X$ which has a PMF $P(X=x)$, sometimes denoted as $p(x)$.

For marginalization (representing the distribution in terms of only one variable), $p(x^i)$ represents the marginal probability for $x^i$, with the complement of $x^i$ being shown as $x^{\bar{i}}$, which is all the arguments excluding $i$. These definitions are used to define a PMF to compare sequences $x$ and $y$ as well in the form of a delta function $\delta_y$, where $\delta_y(x)=1$ only if $x=y$. This is given shorthand $\delta_y(x^i)=\delta_{y^i}(x^i)$ and $\delta_y(x^{\bar{i}})=\delta_{y^{\bar{i}}}(x^{\bar{i}})$.
$$
\delta_y(x)=\prod^N_{i=1}\delta_{y^i}(x^i),\text{ where }\delta_{y^i}(x^i)=\begin{cases}1&x^i=y^i\\0&x^i\neq y^i\end{cases}
$$
The source samples $X_0\sim q$ and target samples $X_1\sim q$ are drawn from a joint distribution $\pi(x,y)$ in which $\pi(x,y)$ describes the probability of observing a pair $(X_0=x,X_1,y)$. This is defined where $p(x)=\sum_{y\in\mathcal{D}}\pi(x,y)$ (the marginal over $X_0$ recovers the source distribution), $q(y)=\sum_{x\in\mathcal{D}}\pi(x,y)$ (the marginal over $X_1$ recovers the target distribution), and in the simplest case $(X_0,X_1)\sim p(X_0)q(X_1)$ (both parts are drawn independently from their distributions). The probability path of the dataset's distribution $p_t(x)$ can then be defined in the most general case with the following.
$$
\begin{gather*}
p_t(x)=\sum_{x_0,x_1\in\mathcal{D}}p_t(x|x_0,x_1)\pi(x_0,x_1)\\
p_t(x|x_0,x_1)=\prod^N_{i=1}p_t(x^i|x_0,x_1)
\end{gather*}
$$
This is then split up into conditional probability paths just like in Flow Matching with a sum of $m$ conditional probabilities $w^j$. This is combined below with a scheduler $\kappa^{i,j}_t$, which defines how much the path should affect a certain token, where $\sum_j\kappa_t^{i,j}=1$ and $\kappa^{i,j}_t\leq 0$. The rest is left up to choice and the scheduler can be defined independently for each token or uniformly across them all.
$$
p_t(x^i|x_0,x_1)=\sum^m_{j=1}\kappa^{i,j}_tw^j(x^i|x_0,x_1)
$$
In order to fit Flow Matching into a discrete space, a [Continuous-Time Discrete Markov Chain](https://galton.uchicago.edu/~lalley/Courses/313/ContinuousTime.pdf), a structure that describes how some sample $X_t$ jumps between states of $\mathcal{D}$ depending on a time $t$, is used. This then changes the evolution from a differential equation to the following form factor, where $u_t$ follows the same importance that the vector fields did in the original format.
$$
X_{t+h}\approx\delta_{X^i_t}(\cdot)+h u^i_t(\cdot,X_t)
$$
The value of $u_t$ for the dataset's distribution can be derived simply with a set of conditional probabilities. The definition is left open with two restrictions being that $\sum_{x^i\in[d]}u^i_t(x^i,z)=0$ and $u^i_t(x^i,z)\geq 0$ have to apply. The value of $p_t(x_0,x_1|z)$ can also be derived through Bayes Theorem, which will hold importance later.
$$
\begin{gather*}
u^i_t(x^i,z)=\sum_{x_0,x_1\in\mathcal{D}}u^i_t(x^i,z|x_0,x_1)p_t(x_0,x_1|z)\\
p_t(x_0,x_1|z)=\frac{p_t(z|x_0,x_1)\pi(x_0,x_1)}{p_t(z)}
\end{gather*}
$$
The value of $u_t$ for the model's generation distribution can also be derived, although in a much different manner. The model has weights $\hat{w}^j_t(x^i,z)$ which are controlled by $a^{i,j}_t=\dot{\kappa}^{i,j}_t-\kappa^{i,j}_t\dot{\kappa}^{i,\ell}_t/\kappa^{i,j}_t$ (controls transitions between states) and $b^i_t=\dot{\kappa}^{i,\ell}_t/\kappa^{i,\ell}_t$ (ensures normalization), where $\ell=\text{arg min}_{j\in[m]}\left[\dot{\kappa}^{i,j}_t/\kappa^{i,j}_t\right]$ (selects the dominant transition).
$$
\begin{gather*}
u^i_t(x^i,z)=\sum^m_{j=1}a^{i,j}_t\hat{w}^j_t(x^i|x_0,x_1)+b^i_t\delta_z(x^i)\\
\hat{w}^j_t(x^i,z)=\sum_{x_0,x_1\in\mathcal{D}}w^j(x^i|x_0,x_1)p_t(x_0,x_1|z)
\end{gather*}
$$
The path of $u^i_t$ can be described with the posteriors $\hat{w}^j_t$, meaning that it accurately models the transitions being predicted by itself. This allows for training to occur solely within $\hat{w^j_t}$, which is also required since the nature of the derivation of $u^i_t$ renders it intractable for training. During inference $u^i_t$ is used to improve the results of the model.

The loss is formulated similarly to [Cross-Entropy](https://www.geeksforgeeks.org/what-is-cross-entropy-loss-function/) and measures how closely the prediction of $\hat{w}^j_t$ matches up to the ground truth $w_j$ (defined below where $X_t\sim p_t(X_t|X_0,X_1)$, which represents some intermediate noisy step, and $Y^i_j\sim w_j(Y^i_j|X_0,X_1)$, which represents the ground truth) which was derived above for the probability paths. The ground truth defines how the data has shifted during some given timestep $t$, which is up to the model itself to decide (with random noise, deterministically, etc.).
$$
\mathcal{L}(\theta)=-\sum_{j\in[m],i\in[N]}\mathbb{E}_{t,(X_0,X_1),X_t,Y^j_t}\log\hat{w}^j_t(Y^i_j|X_t;\theta)
$$
The paper goes into many of the almost infinite possibilities that the structure defined here provides which will not be covered here, although I do highly recommend seeing them for further clarification.

## Conclusion:
As covered in my previous blogpost, Diffusion Models are finally making a splash in the language modeling market with the release of Gemini Diffusion from Google. This opens up the gateway for future innovation in such a new field which I consider to have limitless potential. The strengths inherent to diffusion language models however are not unique to diffusion itself, but rather the structure inherent to any form of generative model, which all can parallelize and output information freely in any order they want. This opens up the possibility of other formats of generation to be even more impactful to language generation than diffusion, which is the reason behind Discrete Flow Matching even existing. The open-ended nature and even the mere existence of the modality offers limitless opportunity to make something new, something that may be the next big innovation.