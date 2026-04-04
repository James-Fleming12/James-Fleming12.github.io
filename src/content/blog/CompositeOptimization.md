---
title: Basic Composite Optimization
description: Proximal Point Methods, Proximal Gradient, Accelerated Proximal Gradient, Proximal Newton, and Proximal Quasi-Newton
pubDate: 4/02/2026
---

This is another quick little review like the last ones. It's going to cover the general problem setup and the background of Proximal Point Methods, and then cover Proximal Gradient, Proximal Newton and Proximal Quasi-Newton and some of their convergence properties. It's also going to cover [Accelerated Proximal Gradient](https://www.cs.cmu.edu/~airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf) and [Regularized Proximal Quasi-Newton](https://arxiv.org/abs/2210.07644), and the convergence information from their papers including some explanations that I personally needed.

## Problem Setup:
A composite optimization problem is an optimization problem over a function with two parts, a differentiable $f(x)$ and a potentially non-differentiable $g(x)$.
$$
\min f(x)+g(x)
$$
It is often assumed that $f(x)$ will have some highly complex structure while $g(x)$ is a simple function that has a structure that can be taken advantage of. Consider the function $f(x)+\|x\|_1$, where the $L_1$ regularization promotes sparsity in the weights by pushing those close to zero to zero exactly, which is very helpful, although the non-differentiability of $\|x\|_1$ can cause headaches.

Although there are many methods of non-differentiable optimization out there, like Subgradient Descent (which for future reference would have a convergence rate of $O(1/\epsilon^2)$ to reach an accuracy of $\epsilon$), they lose a lot of information by not taking advantage of the fact that $f(x)$ is differentiable, which is where specific composite optimization problems come in to play.

## Proximal Point Methods:
The main piece of important background before getting into the algorithms themselves is the concept of a Proximal Point Method. Standard gradient descent uses the gradient gauges how far a step should be based on how much the function is supposed to decrease by. Proximal point methods do the same, but through a minimization subproblem since the gradient information is not available. They make use of the proximal operator $\text{prox}_f(\cdot)$, which balances minimizing the function value and how far the step itself is.
$$
\text{prox}_f(v)=\text{arg}\min_{x}\left(f(x)+\frac{1}{2}\|x-v\|^2\right)
$$
A proximal point method simply takes this operator with a stepsize, defined below as $\eta$, and uses it for each iteration.
$$
x_{k+1}=\text{arg}\min_z\{f(z)+\frac{1}{2\eta}\|z-x_k\|^2\}=\text{prox}_{\eta f}(x_k)
$$
Although this general form is almost impossible to solve at each iteration, there is often some sort of structure in $f(\cdot)$ that we can take advantage of to find a better method of finding it. All of the composite optimization problems covered here will also make the act of solving the subproblem much simpler.

### Interpretations:
There are two interpretations of the algorithm that I want to cover to hopefully help with the intuition behind it if it didn't make sense originally. The more intuitive of the two relates PPMs with the Moreau Envelope $M_f(\cdot)$, which provides a way to create a smoothed version of any lower semi-continuous convex function. It can be defined below with some parameter $\lambda\in\mathbb{R}$.
$$
M_{\eta f}(v)=\inf_{x}\left(f(x)+\frac{1}{2\eta}\|x-v\|^2_2\right)
$$
The Moreau Envelope keeps the original function's minimal value and has the same minimizers, so it's extremely helpful for non-smooth optimization. The main benefit of the envelope comes in the fact that $M_{\lambda f}$ is always differentiable even if $f$ isn't.
$$
\nabla M_{\eta f}(x_k)=\frac{1}{\eta}(x_k-\text{prox}_{\eta}f(x_k))
$$
By isolating the proximal operator, one can see that a PPM is simply performing gradient descent on the Moreau Envelope of the function.
$$
x_{k+1}=x_k-\eta\nabla M_{\eta f}(x_k)
$$
Another equally important interpretation is that PPMs are performing implicit subgradient updates, so updating based on the subgradient $g_{k+1}\in\partial f(x_{k+1})$ of the next iterate.
$$
x_{k+1}=x_k-\rho g_{k+1}
$$
This can be intuitively understood since the PPM iteration looks for the exact properties of the next point rather than solely relying on the current. It's derived from the optimality condition of the proximal operator $0\in\partial f(x_{k+1})+\frac{1}{\rho}(x_{k+1}-x_k)$ after some basic manipulation.
$$
\begin{gather*}
\frac{x_k-x_{k+1}}{\rho}\in\partial f(x_{k+1})\\
g_{k+1}=\frac{x_k-x_{k+1}}{\rho}\\
x_{k+1}=x_k-\rho g_{k+1}
\end{gather*}
$$
These both go to show that PPM has a much more solid foundation for convergence than a simple subgradient descent would, which is why it is so commonly used.

## Proximal Gradient Method:
work in progress...

### Convergence:
If $f$ is an L-Smooth convex function, $g$ is a lower-semicontinuous convex function, and a stepsize $\frac{1}{L}$ is used, then we know that the method has a convergence rate of $O(1/\epsilon)$ to reach an error $\epsilon$, which is detailed below.
$$
F(x_k)-F(x^*)\leq\frac{L\|x_0-x^*\|^2}{2k}
$$
In order to prove this rate, we first need to prove that the iterates satisfy $F(x_{k+1})\leq F(x_k)$, which is a property that is guaranteed when $f$ is $L$-Smooth and a stepsize $\frac{1}{L}$ is used. The proof follows by definition of the surrogate, which ...

## Accelerated Proximal Gradient (FISTA):
work in progress...

### Convergence:
If $f$ is an L-Smooth convex function and $g$ is a proper, lower-semicontinuous convex function, then we know that the method has a convergence rate of $O(1/\sqrt{\epsilon})$ to reach an error $\epsilon$. This can be detailed below where $\alpha=1$ if a constant stepsize is used, and $\alpha=\eta$ if a backtracking line search step size is used, where $\eta$ is the acceptance threshold.
$$
F(x_k)-F(x^*)\leq\frac{2\alpha L\|x_0-x^*\|^2}{(k+1)^2}
$$
The proof ...

## Proximal Newton Method:
work in progress...

### Global Convergence:
If $f$ is $m$-strongly convex and $L$-smooth, $g$ is convex, and $t_k$ is chosen using a backtracking line search, then we know that the method has a convergence rate of $O(\log(1/\epsilon))$. This can be detailed below with some consta $\gamma\in(0,1)$ which is proprtional to $\frac{m}{L}$ and penalized by the search parameters for the algorithm behind $t_k$.
$$
F(x_{k+1})-F(x^*)\leq(1-\gamma)(F(x_k)-F(x^*))
$$
To make the relationship between proximal newton and the previous methods clearer, we can see the convergence rates of proximal gradient and accelerated proximal gradient under the same conditions as this theorem in terms of the condition number $\kappa=\frac{L}{m}$ which conveys how stretched the function is, so functions with a higher $\kappa$ are harder to optimize. Since strong convexity is added to $f$, Proximal gradient's convergence rate becomes $O(\kappa\log(1/\epsilon))$ and the acceleration gets it to $O(\sqrt{\kappa}\log(1/\epsilon))$. Proximal Newton's convergence rate doesn't rely on the condition number since it uses the hessian of the function, removing any guess work that the other two make about the curvature of the function.

The proof ...

### Local Convergence:
The real strength of the method can be seen once the iterate gets close enough to the optimal $x^*$. If $f$ is $m$-strongly convex, $L$-smooth, and has an $M$-Lipschitz hessian, $g$ is convex, and $\|x_k-x^*\|<\epsilon$ where $\epsilon\approx\frac{m}{M}$, then we know the method has a convergence rate $O(\log\log(1/\epsilon))$ which is detailed below.
$$
\|x_{k+1}-x^*\|\leq\frac{M}{2m}\|x_k-x^*\|^2
$$
The local and global convergence rates are also called undamped and damped convergence, since once the iterate gets close enough to $x^*$, then we know that the backtracking line search will always return $t_k=1$, leading to the steps not being scaled and thus not being damped.

The proof ...

## Proximal Quasi-Newton (Proximal L-BFGS):
work in progress...

## Regularized Proximal Quasi-Newton:
work in progress...