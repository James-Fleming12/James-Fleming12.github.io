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
Since we assume that $f(x)$ is a hard function to optimize over, to simplify the proximal calculations we can consider using a quadratic approximation of $f(x)$ constructed at each iterate $x_k$, where $L$ defines the curvature of the approximation.
$$
Q_L(x,x_k)=f(x_k)+\left<x-x_k,\nabla f(x_k)\right>+\frac{L}{2}\|x-x_k\|^2+g(x)
$$
Through minimizing this surrogate we can arrive at the following iteration scheme.
$$
x_{k+1}=\text{arg}\min_x\left\{\frac{L}{2}\left\|x-\left(x_k-\frac{1}{L}\nabla f(x_k)\right)\right\|^2+g(x)\right\}
$$
Since this same quadratic approximation is what gradient descent does under the hood, this simplifies down to using the proximal operator on a standard gradient descent step of $f$. This leads to an algorithm that can optimize over $f$ much quicker than the previous subgradient descent while still being feasible at each iteration.
$$
\begin{gather*}
z_k=x_k-\frac{1}{L}\nabla f(x_k)\\
x_{k+1}=\text{prox}_{\frac{1}{L}g}(z_k)
\end{gather*}
$$
Since $g$ is assumed to have a simple structure, there is more often than not a simple closed form solution to the proximal operator step. Consider the case of a simple case of $g(x)$ being an indicator function for some convex set $C$.
$$
g(x)=\mathbb{1}_{C}(x)=\begin{cases}0&\text{if }x\in C\\\infty&\text{if }x\notin C\end{cases}
$$
This definition of $g$ makes the proximal operator become a projection onto $C$, causing the algorithm to turn into projected gradient descent.
$$
\text{prox}_{\frac{1}{L}\mathbb{1}_C(x)}(z)=\Pi_C(z)=\text{arg}\min_{x\in C}\|x-z\|^2
$$
We can also return back to our original example of $g(x)=\|x\|_1$, which once simplified has a proximal operator with a very simple closed form.
$$
\begin{gather*}
g(x)=\|x\|_1=\sum|x_i|\\
\text{prox}_{\frac{1}{L}\|\cdot\|_1}(z)_i=\text{sgn}(z_i)\max\left(|z_i|-\frac{1}{L},0\right)
\end{gather*}
$$
This specific form of the algorithm is called the Iterative Soft-Thresholding Algorithm, shortened to ISTA, and is going to be of slight importance later for naming conventions.

### Convergence:
If $f$ is an L-Smooth convex function, $g$ is a lower-semicontinuous convex function, and a stepsize $\frac{1}{L}$ is used, then we know that the method has a convergence rate of $O(1/\epsilon)$ to reach an error $\epsilon$, which is detailed below.
$$
F(x_k)-F(x^*)\leq\frac{L\|x_0-x^*\|^2}{2k}
$$
In order to prove this rate, we first need to prove that the iterates satisfy $F(x_{k+1})\leq F(x_k)$, which is a property that is guaranteed when $f$ is $L$-Smooth and a stepsize $\frac{1}{L}$ is used. The proof is a simple one that relies on the properties of the surrogate $Q_L(x,x_k)$. By definition we already know that $Q_L(x_k,x)=F(x_k)$ and since $x_{k+1}$ is derived through a minimization we have $Q_L(x_{k+1},x_k)\leq Q_L(x_k,x_k)$. Since $f$ is $L$-smooth, we know that the descent lemma holds, which if $g(x_{k+1})$ is added to both sides proves $F(x_{k+1})\leq Q_L(x_{k+1},x_k)$. Putting these together we can arrive at the desired result.
$$
F(x_{k+1})\leq Q_L(x_{k+1},x_k)\leq Q(x_k,x_k)=F(x_k)
$$
Now we can return to the original convergence theorem. Since $f$ is $L$-smooth, we can apply the descent lemma to the function to provide a bound. Adding $g(x_{k+1})$ to both sides gives us an inequality in terms of $F$ and our surrogate $Q_L$.
$$
\begin{gather*}
f(x_{k+1})\leq f(x_k)+\left<\nabla f(x_k),x_{k+1}-x_k\right>+\frac{L}{2}\|x_{k+1}-x_k\|^2\\
F(x_{k+1})\leq Q_L(x_{k+1},x_k)
\end{gather*}
$$
Since $x_{k+1}$ is the result of a minimization subproblem, which we know is a convex problem since $g$ and the quadratic approximation are both convex, we can use the standard derivative optimality conditions to derive some properties of $x_{k+1}$. We can specifically derive the following, which allows us to know that there is some subderivative of $g$ at $x_{k+1}$ that is equal to $-\nabla f(x_k)-L(x_{k+1},x_k)$.
$$
0\in\nabla f(x_k)+L(x_{k+1},x_k)+\partial g(x_{k+1})
$$
We can then use the convexity of $f$ and $g$ to derive the following bounds for some arbitrary $x$ using the tangent line property, where we replace the subgradient in the inequality with our above defined subgradient.
$$
\begin{gather*}
g(x)\geq g(x_{k+1})+\left<-\nabla f(x_k)-L(x_{k+1},x_k),x-x_{k+1}\right>\\
f(x)\geq f(x_k)+\left<\nabla f(x_k),x-x_k\right>
\end{gather*}
$$
Using these two we can redefine the bound $F(x_{k+1})\leq Q_L(x_{k+1},x_k)$ into one over an arbitrary $x$ and simplify by expanding the inner products.
$$
\begin{gather*}
F(x_{k+1})\leq F(x)+\left<L(x_k-x_{k-1}),x_{k-1}-x\right>+\frac{L}{2}\|x_{k+1}-x_k\|^2\\
F(x_{k+1})\leq F(x)+\frac{L}{2}(\|x_k-x\|^2-\|x_{k+1}-x\|^2)
\end{gather*}
$$
We can then define $x=x^*$ which defines a recursive relationship between $x_k$ and $x_{k+1}$, which means we can also bound the total sum of both over the $k$ iterations.
$$
\begin{gather*}
F(x_{k+1})-F(x^*)\leq\frac{L}{2}(\|x_k-x^*\|^2-\|x_{k+1}-x^*\|^2)\\
\sum^{k-1}_{i=0}(F(x_{i+1})-F(x^*))\leq\frac{L}{2}\sum^{k-1}_{i=0}(\|x_i-x^*\|^2-\|x_{i+1}-x^*\|^2)
\end{gather*}
$$
Since the right-hand side forms a telescoping sum, we can collapse it down to the initial and final terms and simplify.
$$
\sum^{k-1}_{i=0}(F(x_{i+1})-F(x^*))\leq\frac{L}{2}(\|x_0-x^*\|^2-\|x_k-x^*\|^2)=\frac{L}{2}(\|x_0-x^*\|^2)
$$
Now we can finally use the fact that $F(x_k)$ is non-increasing under this setting so that we can know that $F(x_k)-F(x^*)$ will be the smallest term in the sum. This means that $k(F(x_k)-F(x^*))$ will be less than the entire sum, so we can derive the bound below, which completes the proof.
$$
k(F(x_k)-F(x^*))\leq\sum^{k-1}_{i=0}(F(x_{i+1})-F(x^*))\leq\frac{L}{2}(\|x_0-x^*\|^2)\\
$$

## Accelerated Proximal Gradient (FISTA):
work in progress...

### Convergence:
If $f$ is an L-Smooth convex function and $g$ is a proper, lower-semicontinuous convex function, then we know that the method has a convergence rate of $O(1/\sqrt{\epsilon})$ to reach an error $\epsilon$. This can be detailed below where $\alpha=1$ if a constant stepsize is used, and $\alpha=\eta$ if a backtracking line search step size is used, where $\eta$ is the acceptance threshold.
$$
F(x_k)-F(x^*)\leq\frac{2\alpha L\|x_0-x^*\|^2}{(k+1)^2}
$$
The proof ...

## Proximal Newton Method:
In the same way that we can apply the ideas behind gradient descent and nesterov acceleration to PPMs, we can also apply Newton's Method. The same general principle of using a quadratic approximation of $f$ is done in Proximal Newton, but the hessian of $f$ is used to improve the approximation.
$$
\begin{gather*}
\hat{f}_k(x)=f(x_k)+\nabla f(x_k)^T(x-x_k)+\frac{1}{2}(x-x_k)^T\nabla^2f(x_k)(x-x_k)\\
\min_x\left(\hat{f}_k(x)+g(x)\right)
\end{gather*}
$$
We can then simplify this subproblem into the iteration rule for the method, where we can define $v=x_k=H^{-1}\nabla f(x_k)$ and $\nabla^2f(x_k)=H_k$ for notational simplicity.
$$
\begin{gather*}
x_{k+1}=\text{prox}^{H_k}_g(x_k-H^{-1}_k\nabla f(x_k))\\
x_{k+1}=\text{arg}\min_x\left(g(x)+\frac{1}{2}\|x-v\|^2_{H_k}\right)
\end{gather*}
$$
To derive this update rule, we can first use the variable $\Delta x=x-x_k$ to make the subproblem simpler to work with.
$$
\min_{\Delta x}\left\{\nabla f(x_k)^T\Delta x+\frac{1}{2}\Delta x^TH_k\Delta x+g(x_k+\Delta x)\right\}
$$
The first two terms of the minimization are the first two terms of an expanded square missing the constant third term.
$$
\frac{1}{2}\|\Delta x+H^{-1}_k\nabla f(x_k)\|^2_{H_k}=\frac{1}{2}\Delta x^TH_k\Delta x+\nabla f(x_k)^T\Delta x+\text{constant}
$$
This means we can simplify the iteration to reveal the sacled proximal operator used in the algorithm.
$$
x_{k+1}=\text{arg}\min_x\left\{\frac{1}{2}\|x-(x_k-H^{-1}_k\nabla f(x_k))\|^2_{H_k}+g(x)\right\}
$$

### Backtracking Line Search:
As will be covered later, proximal newton has the best convergence we have seen yet once within a certain range of the optimum, which is the range where we know that the the quadratic approximation made is highly accurate to the actual landscape of the function. The biggest issue with the algorithm is that it still suffers from the same instability that Newton's Method itself suffers from. When far from the optimum in strongly convex settings, the function itself does not need to have the same format as a quadratic, so although the quadratic approximation is highly accurate locally, it often makes incorrect assumptions about the surrounding landscape.

The aggressiveness of the approximation makes the general direction of each iteration quite accurate, but the scale of each step is often inaccurate and is the main problem when it comes to the algorithm diverging. This is why the use of a backtracking line search stepsize $t_k$ is so common, and will be required by the theorems following.
$$
x_{k+1}=t_k\text{prox}^{H_k}_g(x_k-H^{-1}_k\nabla f(x_k))
$$
As a quick recap, a backtracking line searh is built on the idea of satisfying the Armijo Condition, an inequality that ensures that the actual decrease of the step taken is at least $\alpha\in(0,0.5)$ times the predicted decrease.
$$
F(x_k+td_k)\leq F(x_k)+\alpha t\Delta_k
$$
To ensure that each step satisfies this, we start with a full step $t_k=1$. If the Armijo condition holds, then we simply take the step. If not, we shrink the stepsize $t_k\leftarrow \beta t_k$ by some $\beta\in(0,1)$ and repeat until it does satisfy the inequality.

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