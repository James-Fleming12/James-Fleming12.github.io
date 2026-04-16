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
k(F(x_k)-F(x^*))\leq\sum^{k-1}_{i=0}(F(x_{i+1})-F(x^*))\leq\frac{L}{2}(\|x_0-x^*\|^2)
$$

## Accelerated Proximal Gradient:
One of the methods of speeding up the convergence of gradient descent with momentum is Nesterov's Acceleration, where we evaluate the gradient at the point that the momentum term is going towards, rather than at the current iterate. The same principle can be used to derive Accelerated Proximal Gradient, which is what the paper on the [Fast Iterative Shrinkage-Thresholind (FISTA)](https://www.cs.cmu.edu/~airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf) algorithm does. First we can redefine the original proximal gradient method by defining $p_L(y)=\text{argmin}\{Q_L(x,y):x\in\mathbb{R}^n\}$ which uses a stepsize $L$.
$$
\begin{gather*}
p_L(y)=\text{argmin}_x\left\{g(x)+\frac{L}{2}\left\|x-\left(y-\frac{1}{L}\nabla f(y)\right)\right\|^2\right\}\\
x_k=p_L(x_{k-1})
\end{gather*}
$$
The accelerated proximal gradient method introduces a momentum term $y_k$ which is given its own stepsize $t_k$. For later convergence analysis we define $t^2_k-t_k=t^2_{k-1}$ so that $t^2_{k+1}-t_{k+1}\leq t^2_k$ holds. Using this along with our previously defined $p_L(\cdot)$ gives the update rule for FISTA.
$$
\begin{gather*}
x_k=p_L(y_k)\\
t_{k+1}=\frac{1+\sqrt{1+4t^2_k}}{2}\\
y_{k+1}=x_k+\left(\frac{t_k-1}{t_{k+1}}\right)(x_k-x_{k-1})
\end{gather*}
$$
While this may feel unintuitive, the same intuition can be used for this as nesterov acceleration itself. Rather than solely relying on the current information, the use of $y_k$ allows the method to look ahead before taking a step, allowing it to correct its trajectory faster. These properties are also shown in the improved convergence guarantees.

### Convergence:
If $f$ is an $L$-Smooth convex function and $g$ is a proper, lower-semicontinuous convex function, then we know that the method has a convergence rate of $O(1/\sqrt{\epsilon})$ to reach an error $\epsilon$.
$$
F(x_k)-F(x^*)\leq\frac{2L\|x_0-x^*\|^2}{(k+1)^2}
$$
In order to start the proof, we need to reintroduce three lemmas from the original proximal gradient definition. The first is the standard descent lemma coming from $f$ being $L$-smooth.
$$
f(x)\leq f(y)+\left<x-y,\nabla f(y)\right>+\frac{L}{2}\|x-y\|^2
$$
The second comes from the optimality conditions of $p_L(y)$ and the fact that its a convex problem. We know that one has $z=p_L(y)$ if and only if there exists $\gamma(y)\in\partial g(z)$ such that the following holds.
$$
\nabla f(y)+L(z-y)+\gamma(y)=0
$$
The third comes from careful analysis of the convexity of both $f$ and $g$. For any $x$ if we have a point $y$ that satisfies $F(p_L(y))\leq Q(p_L(y),y)$ then we have the following.
$$
F(x)-F(p_L(y))\geq\frac{L}{2}\|p_L(y)-y\|^2+L\left<y-x,p_L(y)-y\right>
$$
To prove the third lemma, we can first look at the inequalities from the definition of convexity for both functions and then combine them to get a bound on $F(x)$.
$$
\begin{gather*}
f(x)\geq f(y)+\left<x-y,\nabla f(y)\right>\\
g(x)\geq g(p_L(y))+\left<x-p_L(y),\gamma(y)\right>\\
F(x)\geq f(y)+\left<x-y,\nabla f(y)\right>+g(p_L(y))+\left<x-p_L(y),\gamma(y)\right>
\end{gather*}
$$
From the original lemma requirement that we need $F(p_L(y))\leq Q(p_L(y),y)$ we can simplify.
$$
F(x)-F(p_L(y))\geq-\frac{L}{2}\|p_L(y)-y\|^2+\left<x-p_L(y),\nabla f(y)+\gamma(y)\right>
$$
From the second lemma we know that $\nabla f(y)+\gamma(y)=L(y-p_L(y))$ which we use to simplify the rightmost product.
$$
F(x)-F(p_L(y))\geq-\frac{L}{2}\|p_L(y)-y\|^2+L\left<x-p_L(y),y-p_L(y)\right>
$$
After turning the product into $L\left<x-p_L(y)+y-y,y-p_L(y)\right>$ by adding $y-y$, we can simplify into our original statement, proving the lemma.
$$
F(x)-F(p_L(y))\geq\frac{L}{2}\|p_L(y)-y\|^2+L\left<y-x,p_L(y)-y\right>
$$
Now that we have the prep work layed out, we can start on the convergence proof itself. The proof follows by first establishing a recursive relationship between steps and then using it to bound the suboptimality. We start by applying the third lemma to at $x=x_k$ and $x=x^*$ to derive two bounds, where we use $y=y_{k+1}$ and $L=L_{k+1}$ for both.
$$
\begin{gather*}
\frac{2}{L_{k+1}}(F(x_k)-F(x_{k+1}))\geq\|x_{k+1}-y_{k+1}\|^2+2\left<x_{k+1}-y_{k+1},y_{k+1}-x_k\right>\\
\frac{2}{L_{k+1}}(F(x^*)-F(x_{k+1}))\geq\|x_{k+1}-y_{k+1}\|^2+2\left<x_{k+1}-y_{k+1},y_{k+1}-x^*\right>
\end{gather*}
$$
To simplify the notation, we can define $v_k=F(x_k)-F(x^*)$ and also the fact that $F(x_k)-F(x_{k+1})=(F(x_k)-F(x^*))-(F(x_{k+1}-F(x^*)))$ to make both bounds be on the same terms.
$$
\begin{gather*}
\frac{2}{L_{k+1}}(v_k-v_{k+1})\geq\|x_{k+1}-y_{k+1}\|^2+2\left<x_{k+1}-y_{k+1},y_{k+1}-x_k\right>\\
-\frac{2}{L_{k+1}}(v_{k+1})\geq\|x_{k+1}-y_{k+1}\|^2+2\left<x_{k+1}-y_{k+1},y_{k+1}-x^*\right>
\end{gather*}
$$
Then we multiply the first by $t_{k+1}-1$ before adding the inequalities together so that we can derive a relationship weighted by $t_{k+1}$.
$$
\frac{2}{L_{k+1}}((t_{k+1}-1)v_k-t_{k+1}v_{k+1})\geq t_{k+1}\|x_{k+1}-y_{k+1}\|^2+2\left<x_{k+1}-y_{k+1},t_{k+1}y_{k+1}-(t_{k+1}-1)\right>
$$
To simplify, we first use the fact that we defined $t_k$ such that $t^2_k=t^2_{k+1}-t_{k+1}$ and then use the identity $\|b-a\|^2+2\left<b-a,a-c\right>=\|b-a\|^2-\|a-c\|^2$.
$$
\begin{gather*}
\frac{2}{L_{k+1}}(t^2_kv_k-t^2_{k+1}v_{k+1})\geq\|t_{k+1}(x_{k+1}-y_{k+1})\|^2+2t_{k+1}\left<x_{k+1}-y_{k+1},t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^*\right>\\
\frac{2}{L_{k+1}}(t^2_kv_k-t^2_{k+1}v_{k+1})\geq\|t_{k+1}x_{k+1}-(t_{k+1}-1)x_k-x^*\|^2-\|t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^*\|^2
\end{gather*}
$$
To make this a meaningful bound, we need to make the second norm on the right-hand side match the form of the first. From the defined update for $y_{k+1}$ we know that $t_{k+1}y_{k+1}=t_{k+1}x_k+(t_k-1)(x_k-x_{k+1})$, so we can simplify the second norm into the following.
$$
\|[t_{k+1}x_k+(t_k+1)(x_k-x_{k+1})]-t_{k+1}x_k+x_k-x^*\|^2=\|t_kx_k-(t_k-1)x_{k+1}-x^*\|^2
$$
Now we can define $u_k=t_kx_k-(t_k-1)x_{k+1}-x^*$ to finally get the bound into a more workable form.
$$
\frac{2}{L_{k+1}}(t^2_{k+1}v_k-t^2_{k+1}v_{k+1})\geq\|u_{k+1}\|^2-\|u_k\|^2
$$
Using the fact that we know that $L_k\geq L_{k+1}$ (either from a static step size or from the specific backtracking line search method defined in the paper), we can define the left-hand side with two terms, one dealing only with iteration $k$ and the other dealing only with iteration $k+1$. This acts as our main recursive relationship for the rest of the proof.
$$
\frac{2}{L_k}t^2_kv_k-\frac{2}{L_{k+1}}t^2_{k+1}v_{k+1}\geq\|u_{k+1}\|^2-\|u_k\|^2
$$
To simplify the notation of the analysis in the next steps, we define $a_k=\frac{2}{L_k}t^2_kv_k$ and $b_k=\|u_k\|^2$ to rewrite the relationship as follows.
$$
a_k-a_{k+1}\geq b_{k+1}-b_k
$$
Since we know that $a_k,b_k>0$ and that the sequence $a_k+b_k$ is monotonically non-decreasing (from the fact that we can rearrange and see that $a_k+b_k\geq a_{k+1}+b_{k+1}$), we can know that if $a_1+b_1\leq c$ with some constant $c$, then for all $k$ we have $a_k\leq c$. If we define $c=\|x_0-x^*\|^2$ and use the fact that we initialize $t_1=1$, we can get an inequality that would bound the entire sequence $a_k$ if proven correct.
$$
\frac{2}{L_1}v_1=\|x_1-x^*\|^2\leq\|y_1-x^*\|^2
$$
To start the proof of this inequality, we apply the third lemma on $x=x^*$, $y=y_1$, and $L=L_1$, and then swap in $p_{L_1}(y)=x_1$ from the method's definition.
$$
\begin{gather*}
F(x^*)-F(p(y_1))\geq\frac{L_1}{2}\|p(y_1)-y_1\|^2+L_1\left<y_1-x^*,p(y_1)-y_1\right>\\
F(x^*)-F(x_1)\geq\frac{L_1}{2}\|x_1-y_1\|^2+L_1\left<y-x^*,x_1-y_1\right>
\end{gather*}
$$
Using the same norm identity that we used for the recursive relation proof lets us simplify in the same way, proving that the base case we needed holds.
$$
\begin{gather*}
F(x^*)-F(x_1)\geq\frac{L_1}{2}(\|x_1-x^*\|^2-\|y_1-x^*\|^2)\\
-v_1\geq \frac{L_1}{2}(\|x_1-x^*\|^2-\|y_1-x^*\|^2)\\
\frac{2}{L_1}v_1+\|x_1-x^*\|^2\leq\|y_1-x^*\|^2
\end{gather*}
$$
From this, we know that we have $a_k\leq c$ for all $k$, which after isolating $v_k$ and using the fact that we define $t_k$ in a way that makes $t_k\geq\frac{k+1}{2}$ hold can be used to simplify the denominator, proving the convergence theorem.
$$
\begin{gather*}
\frac{2}{L_1}t^2_kv_k\leq\|x_0-x^*\|^2\\
v_k\leq\frac{L_k\|x_0-x^*\|^2}{2t^2_k}\\
v_k\leq\frac{2L_k\|x_0-x^*\|^2}{(k+1)^2}
\end{gather*}
$$

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

The proof follows by first proving that the direction of the iteration is a strict descent direction, and then using the line search to derive a recursive relationship between the function values of iterates.

We can first define some properties of the direction $d_k=\hat{x}_{k+1}-x_k$ by using teh optimality conditions of the subproblem since $f$ and $g$ are convex. We define $v\in\partial(x_k+d_k)$ and $H_k=\nabla^2 f(x_k)$ for simplicity.
$$
\begin{gather*}
0\in\nabla f(x_k)+H_k(\hat{x}_{k+1}-x_k)+\partial g(\hat{x}_{k+1})\\
\nabla f(x_k)+H_kd_k+v=0
\end{gather*}
$$
We can prove that $d_k$ is a descent direction by looking at the directional derivative of the problem towards $d_k$, where we need to use the change in $g$ rather than its derivative due to its non-differentiability.
$$
\Delta_k=\nabla f(x_k)^Td_k+g(x_k+d_k)-g(x_k)
$$
Since $g$ is convex, we know that $g(x_k)\geq g(x_k+d_k)-v^Td_k$, so along with the properties from optimality above we can simplify.
$$
\Delta_k\leq\nabla f(x_k)^Td_k+v^Td_k=(\nabla f(x_k)+v)^Td_k=-d^T_kH_kd_k
$$
Since $f$ is $m$-strongly convex, we know that $H_k\succeq mI$ with $m>0$. This means we can bound the right-hand side, proving that $d_k$ is a strict descent direction as $\Delta_k=0$ only if $\|d_k\|^2=0$.
$$
\Delta_k\leq -m\|d_k\|^2\leq 0
$$
Next, we can use the backtracking line search to show that the algorithm leads to a sufficient decrease in $F(x)$. Since we know that the Armijo Condition is satisfied, we know that the following applies.
$$
F(x_k+t_kd_k)\leq F(x_k)+\alpha t_k\Delta_k
$$
To make this descent meaningful, we need to lower bound $t_k$ to make sure that there is progress being made, which we do through manipulation of the function properties. Since $f$ is $L$-smooth, we can use the descent lemma to know the first bound, and since $g$ is convex and $t\in(0,1]$, we can derive the second bound.
$$
\begin{gather*}
f(x_k+td_k)\leq f(x_k)+t\nabla f(x_k)^Td_k+\frac{L}{2}t^2\|d_k\|^2\\
g(x_k+td_k)=g((1-t)x_k+t(x_k+d_k))\leq(1-t)g(x_k)+tg(x_k+d_k)\\
g(x_k+td_k)\leq g(x_k)+t(g(x_k+d_k)-g(x_k))
\end{gather*}
$$
Combining these two lets us derive a bound on the entire objective.
$$
F(x_k+td_k)\leq f(x_k)+t\nabla f(x_k)^Td_k+\frac{L}{2}t^2\|d_k\|^2+g(x_k)+t(g(x_k+d_k)-g(x_k))
$$
Grouping the first order terms and the previous defined $\Delta_k$, along with the fact that we know $\|d_k\|^2\leq-\frac{1}{m}\Delta_k$ from previous steps lets us simplify.
$$
\begin{gather*}
F(x_k+td_k)\leq F(x_k)+t\Delta_k+\frac{L}{2}t^2\|d_k\|^2\\
F(x_k+td_k)\leq F(x_k)+t\Delta_k+\frac{L}{2}t^2\left(-\frac{1}{m}\Delta_k\right)\\
F(x_k+td_k)\leq F(x_k)+t\Delta_k\left(1-\frac{Lt}{2m}\right)
\end{gather*}
$$
This gives us an upper bound parabola with terms we can work with. To show that the Armijo Condition can be satisfied, we can require that the right-hand side of the Armijo equation is below the parabola we defined here.
$$
\begin{gather*}
F(x_k)+t\Delta_k\left(1-\frac{Lt}{2m}\right)\leq F(x_k)+\alpha t\Delta_k\\
t\Delta_k\left(1-\frac{Lt}{2m}\right)\leq\alpha t\Delta_k\\
1-\frac{Lt}{2m}\geq\alpha\\
t\leq \frac{2m(1-\alpha)}{L}
\end{gather*}
$$
Since $\alpha\in(0,0.5)$, we know that $t\leq \frac{m}{L}$ will satisfy the Armijo Condition. This leads to either $t_k=1$ being accepted, or a lower bound of $\beta\frac{m}{L}$ if we do need to scale by $\beta$, so $t_k\geq\min\left(1,\beta\frac{m}{L}\right)$. Since we know that $m\leq L$ by definition of the properties and $0<\beta<1$, this simplifies to the following.
$$
t_k\geq\beta\frac{m}{L}
$$
Now that we've done the proper setup, we can start working on making the final bound on suboptimality. To start we use the fact that $f(x)$ is strongly convex and that $g(x)$ is convex, where again $v\in\partial g(x_k+d_k)$.
$$
\begin{gather*}
f(x^*)\geq f(x_k)+\nabla f(x_k)^T(x^*-x_k)+\frac{m}{2}\|x^*-x_k\|^2\\
g(x^*)\geq g(x_k+d_k)+v^T(x^*-(x_k+d_k))
\end{gather*}
$$
We can then combine these bounds to derive a bound on the global minimum $F(x^*)$. We can add $\nabla f(x_k)^Td_k+g(x_k)$ to both sides to simplify, in turn introducing our previous $\Delta_k$.
$$
\begin{gather*}
F(x^*)\geq f(x_k)+\nabla f(x_k)^T(x^*-x_k)+\frac{m}{2}\|x^*-x_k\|^2+g(x_k+d_k)+v^T(x^*-x_k-d_k)\\
F(x^*)\geq F(x_k)+\Delta_k+(\nabla f(x_k)+v)^T(x^*-x_k-d_k)+\frac{m}{2}\|x^*-x_k\|^2
\end{gather*}
$$
We can then substitute in $\nabla f(x)+v=-H_kd_k$ from our optimality condition analysis and simplify further.
$$
F(x^*)\geq F(x_k)+\Delta_k-d^T_kH_k(x^*-x_k)+d^T_kH_kd_k+\frac{m}{2}\|x^*-x_k\|^2
$$
Using the the Peter-Paul inequality, which states that for any two vectors $a$ and $b$ and any constant $\gamma>0$, we know $a^Tb\leq\frac{1}{2\gamma}\|a\|^2+\frac{\gamma}{2}\|b\|^2$, we can simplify again. Specifically we use the inequality with $a=H_kd_k$ and $b=x^*-x_k$ to remove the $\|x^*-x_k\|^2$ term.
$$
F(x^*)\geq F(x_k)+\Delta_k-\frac{1}{2m}\|H_kd_k\|^2+d^T_kH_kd_k
$$
Since $f$ is $L$-smooth, we know that $H_k\preceq LI$, so $\|H_kd_k\|^2\leq L(d^T_kH_kd_k)$, which we can use along with $d^T_kH_kd_k\leq-\Delta_k$ to get the following.
$$
\begin{gather*}
F(x_k)-F(x^*)\leq -\Delta_k+\frac{L}{2m}(d^T_kH_kd_k)-d^T_kH_kd_k\\
F(x_k)-F(x^*)\leq-\frac{L}{2m}\Delta_k
\end{gather*}
$$
Finally, we can rearrange to isolate $\Delta_k$ to derive a bound on it, which we can substitute into the Armijo Condition the derive our final inequality. We can define $\gamma=\alpha\beta\frac{2m^2}{L^2}$, which satisfies $0<\gamma<1$, proving the theorem.
$$
F(x_{k+1})\leq F(x_k)-\alpha\left(\beta\frac{m}{L}\right)\left[\frac{2m}{L}(F(x_k)-F(x^*))\right]
$$

### Local Convergence:
The real strength of the method can be seen once the iterate gets close enough to the optimal $x^*$, when the quadratic approximation becomes highly accurate. If $f$ is $m$-strongly convex, $L$-smooth, and has an $M$-Lipschitz hessian, $g$ is convex, and $\|x_k-x^*\|<\epsilon$ where $\epsilon\approx\frac{m}{M}$, then we know the method has a convergence rate $O(\log\log(1/\epsilon))$ which is detailed below.
$$
\|x_{k+1}-x^*\|\leq\frac{M}{2m}\|x_k-x^*\|^2
$$
Once the iterate gets close enough to $x^*$, we also know that the backtracking line search will always return $t_k=1$, leading to the steps not being damped.

The proof follows by first proving that the error between the actual function and the approximation is bounded, and using it to prove the recursive relationship we want.

We start by finding two subgradients $v^*\in\partial g(x^*)$ and $v_{k+1}\in\partial g(x_{k+1})$ from the optimality conditions of the problem as a whole and of the subproblem for each iteration.
$$
\begin{gather*}
-v^*=\nabla f(x^*)\\
-v_{k+1}=\nabla f(x_k)+\nabla^2f(x_k)(x_{k+1}-x_k)
\end{gather*}
$$
We can use these two to define an inequality using the monotone gradient property of the convex $g$, which we can combine with the strong convexity of $f$ to define an inequality with all of the terms we need.
$$
\begin{gather*}
(v_{k+1}-v^*)^T(x_{k+1}-x^*)\geq 0\\
(\nabla f(x_{k+1})-\nabla f(x^*))^T(x_{k+1}-x^*)\geq m\|x_{k+1}-x^*\|^2\\
(\nabla f(x_{k+1})-v_{k+1}-(\nabla f(x^*)-v^*))^T\geq m\|x_{k+1}-x^*\|^2
\end{gather*}
$$
This can be simplified using $\nabla f(x^*)-v^*=0$, and then by applying Cauchy-Schwarz and dividing by $m\|x_{k+1}-x^*\|^2$.
$$
\begin{gather*}
(\nabla f(x_{k+1})+v_{k+1})^T(x_{k+1}-x^*)\geq m\|x_{k+1}-x^*\|^2\\
\|x_{k+1}-x^*\|^2\leq\frac{1}{m}\|\nabla f(x_{k+1})+v_{k+1}\|
\end{gather*}
$$
The term $\nabla f(x_{k+1})+v_{k+1}$ represents the error between the expected gradient at the next iterate and the real gradient, which we can use the $M$-Lipschitz contuity of the hessian to bound. This is where the importance of the use of the hessian becomes important, as it makes this error much smaller than in typical gradient descent methods.
$$
\nabla f(x_{k+1})+v_{k+1}=\nabla f(x_{k+1})-[\nabla f(x_k)+\nabla^2f(x_k)(x_{k+1}-x_k)]
$$
To denote the difference $\nabla f(x_{k+1})-\nabla f(x_k)$ in terms that can be combined with the hessian term, we can use the Fundamental Theorem of Calculus to denote the term using a path integral over $p(\tau)=x_k+\tau(x_{k+1}-x_k)$, since we know that $\frac{dp}{\tau}=x_{k=1}-x_k$.
$$
\begin{gather*}
\nabla f(x_{k+1})-\nabla f(x_k)=\int^1_0\frac{d}{d\tau}\nabla f(p(\tau))d\tau\\
\nabla f(x_{k+1})-\nabla f(x_k)=\int^1_0\nabla^2f(x_k+\tau(x_{k+1}-x_k))(x_{k+1}-x_k)d\tau
\end{gather*}
$$
Substituting this into the original statement allows us to simplify.
$$
\nabla f(x_{k+1})+v_{k+1}=\int^1_0[\nabla^2f(x_k+\tau(x_{k+1}-x_k))-\nabla^2f(x_k)](x_{k+1}-x_k)d\tau
$$
Since the hessian is $M$-Lipschitz, we can take the norm of both sides and bound the terms inside the bracket. This leads to an integral with a constant integrand, so we can simplify using the fact that $C\int^1_0\tau d\tau=\frac{C}{2}$.
$$
\begin{gather*}
\|\nabla f(x_{k+1})-v_{k+1}\|\leq\int^1_0M\tau\|x_{k+1}-x_k\|^2d\tau\\
\|\nabla f(x_{k+1})-v_{k+1}\|\leq\frac{M}{2}\|x_{k+1}-x_k\|^2
\end{gather*}
$$
To finish the proof, we substitute this into our original bound to derive the desired result.
$$
\|x_{k+1}-x_k\|\leq\frac{M}{2m}\|x_{k+1}-x_k\|^2
$$

## Regularized Proximal Quasi-Newton:
One of the main problems with the default proximal newton setup is that in modern applications, the added function calls for a backtracking line search are extremely computationally expensive. This is an annoyance especially in newton's method since we only need the backtracking line search due to the opportunity for the approximation to overfit locally and choose a bad step direction. Regularized [Proximal Quasi-Newton](https://arxiv.org/abs/2210.07644) aims to fix this by updating and reevaluating the proximal subproblem itself, rather than trying to work with the first direction that was obtained.

To follow the syntax of the paper, we redefine the composite optimization problem as one over a continuously differentiable function $f$ and a convex function $\varphi$. This means we redefine the original proximal newton step as a minimization subproblem over an approximation $q_k$ and then perform $x^{k+1}=x^k+d^k$.
$$
\begin{gather*}
\min_x\psi(x)=f(x)+\varphi(x)\\
\min_dq_k(d)\quad\text{with}\quad q_k(d)=f(x^k)+\nabla f(x^k)^Td+\frac{1}{2}d^TB_kd+\varphi(x^k+d)
\end{gather*}
$$
Instead of using this approximation, the regularized proximal quasi-newton method uses a regularized approximation $\hat{q}_k$ with the regularization controlled by a parameter $\mu_k>0$.
$$
\hat{q}_k(d)=q_k(d)+\frac{1}{2}\mu_k\|d\|^2
$$
To help define optimality for the problem, the paper uses the residual of a proximal gradient step at the current iterate $r(x^k)$ as a gauge, since $\|d_k\|=0$ is a sufficient but not necessary property for stationarity if $B_k+\mu_k I$ isn't positive definite (which is not required for anything here).
$$
r(x)=\text{prox}_\varphi(x-\nabla f(x))-x
$$
As well, in order to help quantify the quality of the step $d_k$ the paper also introduces two terms for each iteration, the predicted reduction $\text{pred}_k$ (which uses the nonregularized form as a better gauge of how much confidence we should have in it) and the actual reduction $\text{ared}_k$, where the ratio $\rho_k=\text{ared}_k/\text{pred}_k$ gives a good idea of how accurate the approximation was.
$$
\begin{gather*}
\text{pred}_k=\psi(x^k)-q_k(d^k)\\
\text{ared}_k=\psi(x^k)-\psi(x^k+d^k)
\end{gather*}
$$
The method starts by initializing parameters $x^0\in\mathbb{R}^n$, $\mu_0>0$, $p_\text{min}\in(0,\frac{1}{2})$ (a tolerance parameter), $c_1\in(0,\frac{1}{2})$ (lower threshold for $\rho_k$), $c_2\in(c_1,1)$ (upper threshold for $\rho_k$), $\sigma_1\in(0,1)$ (regularization shrink parameter), $\sigma_2>1$ (regularization growth parameter), and $k=0$. Then the method can be broken up into three steps.

First, we use a suitable termination criterion to check the current $x^k$ and terminate accordingly.

Second, we choose $B_k\in\mathbb{R}^{n\times n}$ and find a solution $d^k$ to $\min_d\hat{q}_k(d)$. If the problem has no solution or if $\text{pred}_k\leq p_\text{min}\|d^k\|\cdot\|r(x^k)\|$ (checking that the predicted drop is a sufficient decrease since $B_k$ is not necesarilly positive definite), we update $x^{k+1}=x^k$ and $\mu_{k+1}=\sigma_2\mu_k$ and move on to the next iteration.

Third, if the previous $d^k$ passed, we set $\rho_k=\text{ared}_k/\text{pred}_k$ and perform the following updates based on how successful the step was. We break down the updates on $\mu_{k+1}$ into unsuccessful if the iteration was skipped or $\rho_k\leq c_1$, successful if $c_1<\rho_k\leq c_2$, and highly successful if $\rho_k>c_2$.
$$
\begin{gather*}
x^{k+1}=\begin{cases}x^k&\text{if }\rho_k\leq c_1\\x^k+d^k&\text{otherwise}\end{cases}\\
\mu_{k+1}=\begin{cases}\sigma_2\mu_k&\text{if }\rho_k\leq c_1\\\mu_k&\text{if }c_1<\rho_k\leq c_2\\\sigma_1\mu_k&\text{otherwise}\end{cases}
\end{gather*}
$$

### Convergence Preliminaries:
A bunch of preliminaries are mentioned in the paper before going into the convergence proofs, so I'll include the four that were novel to me personally here since they are also important for later steps of the proof.

As a refresher for the firs two, we define a scaled proximal operator with some matrix $H$ and a function $\varphi$ as follows, where in the following theorems we assume that $H\in\mathbb{S}^n_{++}$ (symmetric positive definite) and that $\varphi$ is convex.
$$
\text{prox}^H_{\varphi}(v)=\text{arg}\min_x\{\varphi(x)+\frac{1}{2}(y-x)^TH(y-x)\}
$$
First, we can prove that $p=\text{prox}^H_\varphi(x)$ if and only if $p\in x-H^{-1}\partial\varphi(p)$. We start by defining the objective function $J(z)$ of the operator.
$$
J(z)=\varphi(z)+\frac{1}{2}(z-x)^TH(z-x)
$$
Since $\varphi$ is convex and $H$ is positive definite, we know $J(z)$ is strictly convex, thus we know that a point $p$ is the global minimizer of $J(z)$ if and only if $0\in\partial J(p)$. We can then use the sum rule for subdifferentials to simplify this statement into what we need.
$$
\begin{gather*}
\partial J(p)=\partial\varphi(p)+H(p-x)\\
0\in\partial\varphi(p)+H(p-x)\\
H(x-p)\in\partial\varphi(p)
\end{gather*}
$$
Since $H$ is positive definite, it is invertible, which allows us to derive what we originally stated, concluding the proof.
$$
\begin{gather*}
x-p\in H^{-1}\partial\varphi(p)\\
p\in x-H^{-1}\partial\varphi(p)
\end{gather*}
$$

Second, we can prove that the proximal operator is firmly nonexpansive with respect to the norm induced by $H$, ie for every $x,y\in\mathbb{R}^n$ we have the following.
$$
\|\text{prox}^H_\varphi(x)-\text{prox}^H_\varphi(y)\|^2_H\leq\left<\text{prox}^H_\varphi(x)-\text{prox}^H_\varphi(y),x-y\right>_H
$$
For simplicity, we can define $p=\text{prox}^H_{\varphi}(x)$ and $q=\text{prox}^H_\varphi(y)$. From the optimality conditions we showed previously we know the following.
$$
\begin{gather*}
H(x-p)\in\partial\varphi(p)\\
H(y-p)\in\partial\varphi(q)
\end{gather*}
$$
Since $\varphi$ is a convex function, we know its subdifferential operator is monotone, thus for any $p,q$ and their corresponding subgradients $u,v$ we have $\left<u-v,p-q\right>\geq 0$, thus we know the following for our points.
$$
\begin{gather*}
\left<H(x-p)-H(y-q),p-q\right>\geq 0\\
\left<H((x-p)-(y-q)),p-q\right>=\left<(x-y)-(p-q),p-q\right>_H\geq0
\end{gather*}
$$
We can distribute the inner product and simplify, which allows us to derive what we originally stated, concluding the proof.
$$
\begin{gather*}
\left<x-y,p-q\right>_H-\left<p-q,p-q\right>_H\geq 0\\
\left<x-y,p-q\right>_H\geq\|p-q\|^2_H\\
\left<\text{prox}^H_\varphi(x)-\text{prox}^H_\varphi(y),x-y\right>_H\geq \|\text{prox}^H_\varpi(x)-\text{prox}^H_\varphi(y)\|^2_H
\end{gather*}
$$

For the third, we have to prove that if $d^k=0$ then $x^k$ is a stationary point of $\psi$, with the converse of the statement being true if $B_k+\mu_kI$ is positive definite. Keep in mind that the algorithm itself does not require $B_k+\mu_kI$ to be positive definite anywhere, so the more reliable $r(x^k)$ stopping criterion is used instead, we just need this for some assumptions made later.

To prove that $d^k=0$ leads to $x^k$ being a stationary point, then we only need to use the definition of $d^k$ and Fermat's Rule, which trivially simplifies into the statement for the stationarity of $x^k$.
$$
\begin{gather*}
0\in\nabla f(x^k)+(B_k+\mu_kI)d^k+\partial\varphi(x^k+d^k)\\
0\in\nabla f(x^k)+\partial\varphi(x^k)
\end{gather*}
$$
To prove that $x^k$ being a stationary point leads to $d^k=0$ if $B_k+\mu_kI$ is positive definite, we first get the fact that $-\nabla f(x^k)\in\partial\psi(x^k)$ from stationarity. We can then use this subgradient in the definition of $\varphi$ being convex to know $\varphi(x^k+d)\geq\varphi(x^k)-\nabla f(x^k)^Td$. Using this we can analyze the objective for our iteration subproblem.
$$
\hat{q}_k(0)=f(x^k)+\varphi(x^k)\leq f(x^k)+\nabla f(x^k)^Td+\varphi(x^k+d)
$$
We can add the term $\frac{1}{2}d^T(B_k+\mu_kI)d$ to the right-hand side, since by $B_k+\mu_kI\succ 0$ we know that the term will be positive. This leads to the statement that $d^k=0$ is the global minimizer, which is guaranteed to be the only solution by $B_k+\mu_kI$'s positive definiteness causing the subproblem to become strictly convex, thus proving the statement.
$$
\hat{q}_k(0)\leq f(x^k)+\nabla f(x^k)^Td+\frac{1}{2}d^T(B_k+\mu_kI)d+\varphi(x^k+d)=\hat{q}_k(d)
$$
For the fourth, we need to derive an upper bound for the general residual $r_H(x)$, where the one we discussed previously was defined $r(x)=r_I(x)$.
$$
\begin{gather*}
r_H(x)=\text{arg}\min_d\left\{\nabla f(x)^Td+\frac{1}{2}d^THd+\varphi(x+d)\right\}\\
r_H(x)=\text{prox}^H_\varphi(x-H^{-1}\nabla f(x))-x
\end{gather*}
$$
For some $x$, a matrix $H$, and symmetric positive definite matrix $\tilde{H}$, we know the following.
$$
\|r_{\tilde{H}}(x)\leq\left(1+\frac{\lambda_\text{max}(\tilde{H})}{\lambda_\text{min}(H)}\right)\cdot\frac{\lambda_\text{max}(H)}{\lambda_\text{min}(\tilde{H})}\cdot\|r_H(x)\|
$$
The proof borrows a result from a work that will be mentioned later again, Block Coordinate Descent. From a lemma in the paper we know the following, where $Q=H^{-1/2}\tilde{H}H^{-1/2}$. The proof for the lemma follows from the optimality conditions of the residuals, Fermat's Rule, and some careful algebraic manipulation.
$$
\|r_{\tilde{H}}\|(x)\leq\frac{1+\lambda_\text{max}(Q)+\sqrt{1-2\lambda_\text{min}(Q)+\lambda_\text{max}(Q)^2}}{2}\frac{\lambda_\text{max}(H)}{\lambda_\text{max}(\tilde{H})}\cdot\|r_H(x)\|
$$
We can then derive two bounds for the terms within the bound derived above.
$$
\begin{gather*}
1-2\lambda_{\min}(Q)+\lambda_{\max}(Q)^2\leq 1+\lambda_{\max}(Q)^2\leq (1+\lambda_{\max}(Q))^2\\
\lambda_{\max}(Q)\leq \lambda_{\max}(\tilde{H})/\lambda_{\min}(H)
\end{gather*}
$$
These can then be plugged into the bound, which after being simplified gives us the desired result.
$$
\begin{gather*}
\lambda_{\max} (Q)  = \max_{x \neq 0} \frac{x^T H^{-1/2} \tilde{H} H^{-1/2} x}{x^T x} 
= \max_{z \neq 0} \frac{z^T \tilde{H} z}{z^T H z}\\
\lambda_{\max} (Q)= \max_{z \neq 0} \bigg( \frac{z^T \tilde{H} z}{z^T z} \frac{z^T z}{z^T H z} \bigg) 
\leq \bigg( \max_{z \neq 0} \frac{z^T \tilde{H} z}{z^T z} \bigg)\\
\lambda_{\max}(Q)\leq \bigg( \max_{z \neq 0} \frac{1}{\frac{z^T H z}{z^T z}} \bigg) 
= \lambda_{\max} (\tilde{H}) \frac{1}{\min_{z \neq 0} \frac{z^T H z}{z^T z}} 
= \lambda_{\max} (\tilde{H}) \cdot \frac{1}{\lambda_{\min} (H)}
\end{gather*}
$$

### Global Weak Convergence:
If $\{B_k\}$ is a bounded sequence of symmetric matrices and $\psi$ is bounded from below (along with the requirements from the problem definition), then any sequence $\{x^k\}$ generated by the method satisfies $\liminf_{k\rightarrow\infty}\|r(x^k)\|=0$. This ensures that at least one subsequence of the iterations converges towards a stationary point, although this doesn't require that the method itself converges (with that being proven in a later section).

The proof follows through a long series of lemma work to prove through contradiction that there will be an infinite number of successful steps if the algorithm were to never reach termination, and then use that to prove the theorem itself through another contradiction.

To start the prep work for the first contradiction, we establish an inequality that relates $r(x^k)$ and $d^k$ if we assume that $\{x^k\}$ converges to a nonstationary point $\bar{x}$ of $\psi$. We also assume that $\{B_k\}$ is a bounded sequence of symmetric matrices and that $\mu_k\rightarrow\infty$.

Since $\{B_k\}$ is bounded, we know that there is some constant $C$ such that $\lambda_\text{min}(B_k)\geq -C$, which combined with the fact that we know $\mu_k\rightarrow\infty$ means that we can guarantee $\lambda_\text{min}(B_k+\mu_kI)\geq -C+\mu_k>0$ for large enough $k$, meaning we can guarantee that $B_k+\mu_kI$ is positive definite for large enough $k$. This lets us use the lemma we established previous about the weighted residuals with $H=B_k+\mu_kI$ and $\tilde{H}=I$.
$$
\|r_{\tilde{H}}(x)\|\leq\left(1+\frac{\lambda_\text{max}(\tilde{H})}{\lambda_\text{min}(H)}\right)\cdot\frac{\lambda_\text{max}(H)}{\lambda_\text{min}(\tilde{H})}\cdot\|r_H(x)\|
$$
Since  $\|r(x)\|=0$ if and only if $x$ is a stationary point, we know that $\|r_{B_k+\mu_kI}(x^k)\|\neq 0$, so we can divide it from both sides, which along with dividing by $\mu_k$ and taking $k\rightarrow\infty$ lets us derive the bound we need for future steps.
$$
\begin{gather*}
\frac{\|r(x^k)\|}{\|r_{B_k+\mu_kI}(x^k)\|}\leq\left(1+\frac{1}{\lambda_\text{min}(B_k)+\mu_k}\right)\cdot(\lambda_\text{max}(B_k)+\mu_k)\\
\limsup_{k\rightarrow\infty}\frac{\|r(x^k)\|}{\|r_{B_k+\mu_kI}(x^k)\|\cdot\mu_k}\leq 1
\end{gather*}
$$
Second, we show that under high regularization we know that $d^k$ will always converge to $0$ under the assumptions that $\{B_k\}$ and $\{x^k\}$ are bounded sequences, $\mu_k\rightarrow\infty$.

Since $\{B_k\}$ is bounded and $\mu_k\rightarrow\infty$ lead to $B_k+\mu_kI$ being positive definite from previous steps, we know that the subproblem turns into a strictly convex problem for sufficiently large $k$, meaning that $d^k$ is well defined in such cases. As well, since each iteration is only accepted if its successful or highly successful (which for the rest of the proof will be lumped into being successful), we know that the sequence can't worsen in the objective, so $\{\psi(x^k)\}$ is monotonically increasing. From these two we know that we have the following from the definition of $d^k$.
$$
\psi(x^0)\geq\psi(x^k)=\hat{q}_k(0)\geq\hat{q}_k(d^k)
$$
We can then use the convexity of $\varphi$ to know that $\varphi(x^k+d^k)\geq\varphi(x^k)+(u^k)^Td^k$ (for some subderivative $u^k\in\partial\varphi(x^k)$) after expanding the defined $\hat{q}_k$ and $\psi$ to get the bound in more separable terms.
$$
\begin{gather*}
\hat{q}_k(d^k)=f(x^k)+\nabla f(x^k)^Td^k+\frac{1}{2}(d^k)^T(B_k+\mu_kI)d^k+\varphi(x^k+d^k)\\
\hat{q}_k(d^k)\geq f(x^k)+\nabla f(x^k)^Td^k+\frac{1}{2}(d^k)^T(B_k+\mu^kI)d^k+\varphi(x^k)+(u^k)^Td^k
\end{gather*}
$$
Since $\{x^k\}$ and $\{B_k\}$ are bounded, from continuity we know that $\{f(x^k)\}$, $\{\psi(x^k)\}$, and $\{\nabla f(x^k)\}$ are bounded. As well, since the subdifferential maps bounded sets to bounded sets we know that $\{u^k\}$ is bounded as well. Since this entire inequality is bounded by a fixed value $\psi(x^0)$, the right-hand side needs to be bounded as $\mu^k\rightarrow\infty$. This means that $\frac{1}{2}(d^k)^T(B_k+\mu_kI)d^k$ needs to be bounded, so $d^k\rightarrow 0$.

Third, we use this statement on $d^k$ to prove that the directional derivative along the normalized step is strictly negative, which is going to be a key property for the later contradiction.

We assume that $\{B_k\}$ is a bounded sequence of symmetric matrices, $\mu_k\rightarrow\infty$, and $\bar{x}$ is a nonstationary point of $\psi$. We also define $\bar{d}^k=r_{B_k+\mu_kI}(\bar{x})$ and some point $s$ has an accumulation point of the sequence $\{\bar{d}^k/\|\bar{d}^k\|\}$ (which we know has accumulation points since the sequence is by definition bounded). Due to Fermat's Rule we know that we have the following from the iteration subproblem for some $u^k\in\partial\varphi(\bar{x}+\bar{d}^k)$.
$$
0=\nabla f(\bar{x})+(B_k+\mu_kI)\bar{d}^k+u^k
$$
Since $(B_k+\mu_kI)\bar{d}^k$ is exactly equal to $-(\nabla f(\bar{x})+u^k)$, we know that both terms cannot be $0$ due to the nonstationarity of the convergent point. This allows us to know that even though $\bar{d}^k\rightarrow 0$, it's still not zero, so we can know that the sequence $\{\bar{d}^k/\|\bar{d}^k\|\}$ is well-defined, so $s$ is a real point.

Since the sequence of normalized vectors and the sequence of subgradients are both bounded, we can use the Bolzano-Weierstrass Theorem to form a subsequence $K$ that satisfies the following. Since the subdifferential is closed we also know that $\bar{u}\in\partial\varphi(\bar{x})$, which means we have $\nabla f(\bar{x})+\bar{u}\neq 0$ from nonstationarity.
$$
\frac{\bar{d}^k}{\|\bar{d}^k\|}\rightarrow_Ks\quad u^k\rightarrow_K\bar{u}
$$
From the previous proofs on Proximal Newton, we know that for some regularized proximal step $d$ we have $\psi^\prime(x;d)\leq-d^THd$, where $H$ is the hessian of the quadratic model. Since we have $H=B_k+\mu_kI$, we can substitute the value in and then use the eigenvalues to simplify.
$$
\psi^\prime(\bar{x},\bar{d}^k)\leq-(\bar{d}^k)^T(B_k+\mu_kI)\bar{d}^k\leq-(\lambda_\text{min}(B_k)+\mu_k)\|\bar{d}^k\|^2
$$
From the previous suboptimality analysis we know $\|\nabla f(\bar{x})+u^k\|=\|(B_k+\mu_kI)\bar{d}^k\|\leq(\|B_k\|+\mu_k)\|\bar{d}^k\|$, which we can use to further simplify.
$$
\psi^\prime(\bar{x},\bar{d}^k)\leq-\|\nabla f(\bar{x})+u^k\|\cdot\frac{\lambda_\text{min}(B_k)+\mu_k}{\|B_k\|+\mu_k}\cdot\|\bar{d}^k\|
$$
As we have $\varphi$ as a convex function, we know that $\psi^\prime(\bar{x};d)$ is sublinear, so we know it satisfies positive homogeneity (the first line of the following), which we can use bound the derivative in terms of our regularized sequence.
$$
\begin{gather*}
\psi^\prime(x;\alpha d)=\alpha\psi^\prime(x;d)\\
\psi^\prime\left(\bar{x};\frac{\bar{d}^k}{\|\bar{d}^k\|}\right)\leq-\|\nabla f(\bar{x})+u^k\|\cdot\frac{\lambda_\text{min}(B_k)+\mu_k}{\|B_k\|+\mu_k}
\end{gather*}
$$
Since we have $k\rightarrow_K\infty$, we can use our previous analysis of the convergent subsequence to prove that the directional derivative is strictly negative. Since $\nabla f(\bar{x})+\bar{u}$ is nonzero, we know its norm has to be positive, and since we have $\mu_k\rightarrow\infty$, the multiplicative term including it converges to $1$.
$$
\psi^\prime(\bar{x},s)=\lim_{K\ni k\rightarrow\infty}\psi^\prime\left(\bar{x},\frac{\bar{d}^k}{\|\bar{d}^k\|}\right)\leq-\|\nabla f(\bar{x})+u\|<0
$$
Finally for the prep work, we can use this and our previous results to show that the algorithm will perform infinitely many successful steps if it goes on forever. 

We assume for contradiction that there exists some $k_0\in\mathbb{N}$ such that all steps $k\geq k_0$ are unsuccessful. This implies that $x^k=x^{k_0}$ for all $k\geq k_0$, means that we have $\|r(x^k)\|\neq 0$ since we haven't reached the stopping criterion yet (so $x^{k_0}$ is nonstationary), and also $\mu_k\rightarrow\infty$ by the algorithm definition. Since $\{B_k\}$ is a bounded sequence, this means that the matrices converge to being positive definite and $d^k$ is well-defined. Also from previous lemmas is the fact that $d^k\neq 0$ and that we have the following (from the first lemma since we have $p_\text{min}<0.5$ and $d^k=r_{B_k+\mu_kI}(x^k)$ from the regularized matrix being positive definite).
$$
\frac{\|r(x^k)\|}{\|d^k\|\mu_k}<\frac{1}{2p_\text{min}}
$$
From the optimality of the subproblem we have $\hat{q}_k(d^k)\leq\hat{q}_k(0)$, which we can use to bound $\text{pred}_k$ to prove that the failure point is not from the second step of the algorithm, meaning we need to fail in the first.
$$
\begin{gather*}
\text{pred}_k=\psi(x^k)-q_k(d^k)=\psi(x^k)-\hat{q}_k(d^k)+\frac{\mu_k}{2}\|d^k\|^2\\
\psi(x^k)-\hat{q}_k(d^k)+\frac{\mu_k}{2}\|d^k\|^2\geq \psi(x^k)-\hat{q}_k(0)+\frac{\mu_k}{2}\|d^k\|^2=\frac{\mu_k}{2}\|d^k\|^2\\
\psi(x^k)-\hat{q}_k(d^k)+\frac{\mu_k}{2}\|d^k\|^2>p_\text{min}\|r(x^k)\|\cdot\|d^k\|
\end{gather*}
$$
This means we need $\text{ared}_k\leq c_1\text{pred}_k$ for the steps to be unsuccessful.
$$
\psi(x^{k_0}+d^k)-\psi(x^{k_0})\geq c_1(\nabla f(x^{k_0})^Td^k+\varphi(x^{k_0}+d^k)-\varphi(x^{k_0})+\frac{1}{2}(d^k)^TB_kd^k)
$$
We can then divide by $\|d^k\|$ to not only introduce our previous regularized sequence, but also introduce some terms in derivative form, for which we define $t_k=\|d^k\|$ for simplicity in notation.
$$
\frac{\psi(x^{k_0}+t_k\frac{d^k}{\|d^k\|})-\psi(x^{k_0})}{t_k}\geq c_1\left(\nabla f(x^{k_0})^T\frac{d^k}{\|d^k\|}+\frac{\varphi(x^{k_0}+t_k\frac{d^k}{\|d^k\|})-\varphi(x^{k_0})}{t_k}+\frac{1}{2}\frac{(d^k)^T}{\|d^k\|}B_kd^k\right)
$$
From previous results, we can choose a subsequence $K$ such that $d^k/\|d^k\|\rightarrow s$, which along with the fact that $\varphi$ is locally lipschitz continuous (allows the left-hand term to turn into $\psi^\prime(x^{k_0};s)$) gives the following. The term on the right-hand side converges in a similar way since $d^k\rightarrow 0$ and $\{B_k\}$ is bounded.
$$
\begin{gather*}
\psi^\prime(x^{k_0};s)\geq c_1(\nabla f(x^{k_0})^Ts+\varphi^\prime(x^{k_0};s))\\
\psi^\prime(x^{k_0};s)\geq c_1\psi^\prime(x^{k_0};s)
\end{gather*}
$$
Since $c_1\in(0,1)$, the only way for this to be true is for $\psi^\prime(x^{k_0};s)\geq 0$, which leads to a contradiction with our previous proof that $\psi^\prime(x^{k_0};s)<0$, which means that the algorithm can not get stuch at a nonstationary point.

With this, we can finally construct the original convergence theorem through another proof by contradiction. We assume that $\liminf_{k\rightarrow\infty}\|r(x^k)\|>0$ for contradiction along with the assumption that $\psi$ is bounded. We also use the previous theorem to define a set of successful iterations $\mathcal{S}$ (which we use since almost all of the properties we prove for the sequence of steps in $\mathcal{S}$ is trivial for unsuccessful steps since the difference between unsuccessful steps is $0$).

Our assumption means that there exists some $k_0\in\mathbb{N}$ and $\epsilon>0$ such that $\|r(x^k)\|\geq\epsilon$ for all $k\geq k_0$. By definition of successful steps, we get the following for all $k\in\mathcal{S}$.
$$
\psi(x^k)-\psi(x^{k+1})\geq c_1\text{pred}_k\geq p_\text{min}c_1\|d^k\|\cdot\|r(x^k)\|\geq p_\text{min}c_1\epsilon\|d^k\|
$$
We can then sum over this recursive relationship to create a telescoping sum, and since $\psi$ is bounded we know the telescoping sum is bounded. We can then isolate the sum over $\|d^k\|$ to get the following.
$$
\begin{gather*}
\infty>\sum^\infty_{k=0}[\psi(x^k)-\psi(x^{k+1})]=\sum_{k\in\mathcal{S}}[\psi(x^k)-\psi(x^k+d^k)]\geq p_\text{min}c_1\epsilon\sum_{k\in\mathcal{S}}\|d^k\|\\
\infty>\sum_{k\in\mathcal{S}}\|d^k\|=\sum_{k\in\mathcal{S}}\|x^{k+1}-x^k\|=\sum^\infty_{k=0}\|x^{k+1}-x^k\|
\end{gather*}
$$
This means that $\{x^k\}$ is a Cauchy Sequence, and therefore convergent to some $\bar{x}$, which from our assumptions is not a stationary point of $\psi$ since $\|r(\bar{x})\|=\lim_{k\rightarrow\infty}\|r(x^k)\|$.

Since we have infinitely many successful steps, for this to stay bounded we need $\|d^k\|\rightarrow_\mathcal{S}0$. We can use Fermat's Rule to see that if we assume $\{\mu_k\}$ is bounded, it would result in a contradiction.
$$
0=\nabla f(x^k)+(B_k+\mu_kI)d^k+u^k\rightarrow \nabla f(x^k)+u^k
$$
This means that we would have $0=\nabla f(\bar{x})+\bar{u}$, which contradicts with the nonstationarity of $\bar{x}$, so we need $\{\mu^k\}_\mathcal{S}\rightarrow\infty$. That also means that $\{\mu_k\}\rightarrow\infty$ since $\mu_k$ cannot decrease during unsuccessful steps, and also that the algorithm performs infinitely many unsuccesful steps since $\mu_k$ cant grow during successful iterations.

From previous steps we know that the following holds for sufficiently large $k$.
$$
\text{pred}_k\geq p_\text{min}\|d^k\|\cdot\|r(x^k)\|\geq p_\text{min}\epsilon\|d^k\|
$$
By the Mean Value Theorem, for every $k$ there exists some $\xi^k$ on the straight line between $x^k$ and $x^k+d^k$ such that $f(x^k+d^k)-f(x^k)=\nabla f(\xi^k)^Td^k$. By the convergence of $\{x^k\}$ to $\bar{x}$, and since $\{d^k\}\rightarrow 0$, the sequence $\{\xi^k\}$ must also converge to $\bar{x}$, which means we can obtain the following for $k\rightarrow\infty$.
$$
\begin{gather*}
|\rho_k-1|=\left|\frac{\psi(x^k)-\psi(x^k+d^k)}{\psi(x^k)-q_k(d^k)}-1\right|=\left|\frac{\psi(x^k+d^k)-q_k(d^k)}{\psi(x^k)-q_k(d^k)}\right|\\
|\rho_k-1|\leq\frac{1}{p_\text{min}\epsilon}\frac{|f(x^k+d^k)-f(x^k)-\nabla f(x^k)^Td^k|+\frac{1}{2}|(d^k)^TB_kd^k|}{\|d^k\|}\\
|\rho_k-1|\leq \frac{1}{p_\text{min}\epsilon}\frac{|\nabla f(\xi^k)d^k-\nabla f(x^k)^Td^k|}{\|d^k\|}+\frac{1}{2p_\text{min}\epsilon}\left|(d^k)^TB_k\frac{d^k}{\|d^k\|}\right|\rightarrow 0
\end{gather*}
$$
This means that $\{\rho_k\}\rightarrow 1$, which means eventually all steps are successful, which yields a contradiction meaning that we need some eventual $k$ that satisfies $\|r(x^k)\|=0$, which concludes the proof.

### Global Strong Convergence:
If $\{B_k\}$ is a bounded sequence of symmetric matrices, $\psi$ is bounded from below, and $\nabla f$ is uniformly continuous on $X$ satisfying $\{x^k\}\subset X$ (along with the requirements from the problem definition), then any sequence $\{x^k\}$ generated by the method satisfies $\lim_{k\rightarrow\infty}\|r(x^k)\|=0$. This ensures that all of the subsequences mentioned in the global weak convergence theorem all converge towards a stationary point, although they can be different stationary points (with convergence to a single stationary point being handled in a later section).

The proof follows by assuming for contradiction that the residual sequence does not strcitly converge to $0$, then using the uniform continuity of $\nabla f$ to show that the residual cannot change drastically over the shrinking step sizes.

From the previous proof, we know there is an index $\ell(k)> k$ such that $\|r(x^l)\|\geq\delta$ for all $k\leq l<\ell (k)$ and $\|r(x^{\ell(k)})\|<\delta$. If for $k\in K$, an iteration $k\leq l<\ell(k)$ is successful, we get the following (which also holds trivially for unsuccessful iterations so we can generalize to a sum over all iteartions).
$$
\begin{gather*}
\psi(x^l)-\psi(x^{l+1})\geq c_1\text{pred}_l\geq c_1p_\text{min}\|r(x^l)\|\cdot\|d^l\|\geq c_1p_\text{min}\delta\|x^{l+1}-x^l\|\\
c_1p_\text{min}\delta\|x^{\ell(k)}-x^k\|\leq c_1p_\text{min}\delta\sum^{\ell(k)-1}_{l=k}\|x^{l+1}-x^l\|\leq\sum^{\ell(k)-1}_{l=k}\psi(x^l)-\psi(x^{l+1})=\psi(x^k)-\psi(x^{\ell(k)})
\end{gather*}
$$
By assumption we know $\psi$ is bounded and from the algorithm definition we know $\{\psi(x^k)\}$ is monotonically decreasing, so the sequence is convergent. This implies that $\{\psi(x^k)-\psi(x^{\ell(k)})\}\rightarrow_K0$, so we get $\{\|x^{\ell(k)}-x^k\|\}\rightarrow_K0$. 

The uniform continuity of $\nabla f$ and the proximal operator means that $r(\cdot)$ is unformly continuous, so we also get $\{\|r(x^{\ell(k)})-r(x^k)\|\}\rightarrow_K0$, which the following contradiction for our definition of $\ell(k)$, concluding the proof.
$$
\|r(x^k)-r(x^{\ell(k)})\|\geq\|r(x^k)\|-\|r(x^{\ell(k)})\|\geq 2\delta-\delta\geq \delta
$$

### Covergence Under an Error Bound Condition:
The paper rounds out the convergence analysis of the algorithm by using a similar error bound strategy that is used in [Block Coordinate Descent](https://link.springer.com/article/10.1007/s10107-007-0170-0). We assume that $\psi$ is bounded from below, the set of stationary points $\mathcal{X}^*$ of $\psi$ is nonempty, and that two properties are satisfied to ensure that the optimization landscape is regular enough to provide stable convergence. One is a local error bound for stationarity and the other is a separation between stationary points. First, for any $\xi\geq\min_x\psi(x)$, there exists scalars $\tau>0$ and $\epsilon>0$ such that $\text{dist}(x,\mathcal{X}^*)\leq\tau\|r(x)\|$ whenever $\psi(x)\leq\xi,\|r(x)\|\leq\epsilon$ (ensures that the residual is a good measure of distance to stationary points). Second, there exists a scalar $\delta>0$ such that $\|x-y\|\geq\delta$ whenever $x\in\mathcal{X}^*$, $y\in\mathcal{X^*}$, and $\psi(x)\neq\psi(y)$ (ensures that stationary points with different function values are properly separated).

If $f$ has a lipschitz continuous gradient and $MI\succeq H_k\succeq mI$ for some $M\geq m>0$ (along with the above being true), then the sequence $\{x^k\}$ converges to some $\bar{x}$. A byproduct of the proof is also the fact that the algorithm satisfies Q-linear convergence, meaning it satisfies teh following for some stationary point $\bar{\psi}$ and some $\theta\in(0,1)$.
$$
\psi(x^{k+1})-\bar{\psi}\leq\theta(\psi(x^k)-\bar{\psi})
$$

The proof follows a very similar structure to that in Block Coordinate Descent's paper after some initla prep work. We start by assuming the sequence $\{H_k\}$ is uniformly bounded and positive definite, ie for some $0<m\leq M$ we have $mI\preceq H_k\preceq MI$. Since $H_k\succeq mI$, we know that $d^T_kH_kd_k\geq m\|d_k\|^2$, which can also be applied to $H_k+\mu_kI$ giving $d^T_k(H_k+\mu_kI)d_k\geq (m+\mu_k)\|d_k\|^2$.

Using these two facts allows us to bound $\text{pred}_k$. We first use teh convexity of $\varphi$ to know $\varphi(x^k+d^k)+\varphi(x^k)\geq u^Td^k$ and then the optimality conditions of the subproblem to know $-(\nabla f(x^k)+u^k)=(H_k+\mu_kI)d^k$.
$$
\begin{gather*}
\text{pred}_k=-(\nabla f(x^k)^Td^k+\varphi(x^k+d^k)-\varphi(x^k))-\frac{1}{2}(d^k)^TH_kd^k\\
\text{pred}_k\geq (d^k)^T(H_k+\mu_kI)d^k-\frac{1}{2}(d^k)^TH_kd^k\geq\frac{1}{2}(m+2\mu_k)\|d^k\|^2
\end{gather*}
$$
Using the lemma we established about the resdiaul gives us two more inequalities which we will use later if we use $\lambda_\text{max}(H_k+\mu_kI)\leq M+\mu_k$ and $\lambda_\text{min}(H_k+\mu_kI)\geq m+\mu_k$.
$$
\begin{gather*}
\frac{\|r(x^k)\|}{\|d^k\|}\leq\left(1+\frac{1}{m+\mu_k}\right)(M+\mu_k)\leq\frac{m+1}{m}(M+\mu_k)\\
\frac{\|d^k\|}{r(x^k)}\leq\frac{1+M+\mu_k}{m+\mu_k}\leq\frac{1+M}{m}
\end{gather*}
$$
Next, we have to prove that $\{\mu_k\}$ is a bounded sequence by first proving that if $f$ is $L$-smooth, then if in some $x^k$ we have $\mu_k\geq\bar{\mu}=\max\{L-m,0\}$, then we have $\text{ared}_k>c_1\text{pred}_k$.

By the lipschitz continuity of $\nabla f$, we have the following, which is further simplified since we assume that $\mu_k\geq\bar{\mu}$, meaning $\mu_k\geq L-m$ and that we have $H_k+\mu_kI\geq mI+(L-m)I=LI$.
$$
f(x^k+d^k)-f(x^k)\leq\nabla f(x^k)^Td^k+\frac{1}{2}L\|d^k\|^2\leq\nabla f(x^k)^Td^k+\frac{1}{2}(d^k)^T(H_k+\mu_kI)d^k
$$
We can then subtract $\varphi(x^k+d^k)-\varphi(x^k)$ from both sides to get a more workable form of the inequality.
$$
\psi(x^k+d^k)-\psi(x^k)\leq\nabla f(x^k)^Td^k+\psi(x^k+d^k)-\psi(x^k)+\frac{1}{2}(d^k)^T(H_k+\mu_kI)d^k
$$
By definition this leads to $-\text{ared}_k\leq-\text{pred}_k+\frac{\mu_k}{2}\|d^k\|^2$, which when combined with the first bound from the previous steps gives the following by using $\mu_k+m/2\mu_k+m>\frac{1}{2}$ and $c\leq\frac{1}{2}$, concluding the proof for the lemma.
$$
\text{ared}_k\geq\text{pred}_k-\frac{\mu_k}{2}\|d^k\|^2\geq\text{pred}_k\cdot\frac{\mu_k+m}{2\mu_k+m}>\frac{1}{2}\text{pred}_k\geq c_1\text{pred}_k
$$
Although this ensures that a successful step is called if $\mu_k$ is large enough, we still need to ensure that the early exit isnt called each time, so we use the previous result to prove that $\{\mu_k\}$ is bounded if $f$ is $L$-Smooth and $MI\succeq H_k\succeq mI$ through a contradiction.

We assume that $\{\mu_k\}$ is unbounded, meaning that there is a subsequence $K$ such that $\{\mu_k\}_K\rightarrow\infty$, which implies there are infinitely many unsuccessful steps. Without loss of generality, we can assume that all steps $k\in K$ are unsuccessful. From the previous lemma, this is only possible if we have $\text{pred}_k<p_\text{min}\|d^k\|\cdot\|r(x^k)\|$, which along with the first bound from the first lemma gives the following.
$$
\begin{gather*}
\frac{m+2\mu_k}{2}\|d^k\|<p_\text{min}\|r(x^k)\|\\
\frac{\|r(x^k)\|}{\mu_k\|d^k\|}>\frac{m+2\mu_k}{2p_\text{min}\mu_k}
\end{gather*}
$$
We can then combine this with the second bound from the first lemma to get the following for all $k\in K$.
$$
\left(1+\frac{1}{m+\mu_k}\right)\frac{M+\mu_k}{\mu_k}>\frac{m+2\mu_k}{2p_\text{min}\mu_k}
$$
If we take the limit in $K$, this leads to $1/p_\text{min}$, which by definition of $p_\text{min}\in(0,\frac{1}{2})$, which leads to a contradiction, completing the proof of the boundedness of $\{\mu_k\}$.

Finally, we can complete the proof of the original theorem by using these lemmas and a couple of lemmas from previous proofs (the fact that $\|r(x^k)\|\rightarrow 0$ and $d^k\rightarrow 0$ under the assumptions from the previous convergence theorems and that $\{\psi(x^k)\}$ is a monotonically decreasing sequence from the algorithm definition). Since $\{\psi(x^k)\}$ is monotonically decreasing and bounded below, we know it must converge to some finite $\bar{\psi}$ which will be important for future steps.

Since $\|r(x^k)\|\rightarrow 0$ for all sufficiently large $k$, we have $\|r(x^k)\|\leq\epsilon$, so we can apply the first error bound assumption to define some $\bar{x}^k\in\mathcal{X}^*$ that satisfies the following.
$$
\text{dist}(x,\mathcal{X}^*)=\|x^k-\bar{x}^k\|\leq\tau\|r(x^k)\|
$$
Since $\|r(x^k)\|\rightarrow 0$, we can simplify this to $\|x^k-\bar{x}^k\|\rightarrow 0$. We can then use this to get a decomposed bound on the difference between each iterate and $\bar{x}$.
$$
\begin{gather*}
\|\bar{x}^{k+1}-\bar{x}^k\|=\|\bar{x}^{k+1}-x^{k+1}+x^{k+1}-x^k+x^k-\bar{x}^k\|\\\\
\|\bar{x}^{k+1}-\bar{x}^k\|\leq\|\bar{x}^{k+1}-x^{k+1}\|+\|x^{k+1}-x^k\|+\|x^k-\bar{x}^k\|
\end{gather*}
$$
From the above convergence statement we know that the first and third term go to $0$, and the second term $\|x^{k+1}-x^k\|=\|d^k\|$ also goes to $0$, so we have $\|\bar{x}^{k+1}-\bar{x}^k\|\rightarrow 0$.

From the second error bound assumption, we know that any two stationary points in $\mathcal{X}^*$ need to have a minimum distance $\delta$, and since the points are getting infinitely close this cannot be satisfied, so they must share the same objective value $\psi(\bar{x}^k)=\bar{\psi}$.

We can then find the suboptimality bound between current iterate's objective value and our convergent $\bar{\psi}$. We first use the convexity of $\varphi$ and the fact that we have $-\nabla f(\bar{x}^k)\in\partial\varphi(\bar{x}^k)$ to get the following.
$$
\varphi(x^{k+1})-\varphi(\bar{x}^k)\geq-\nabla f(\bar{x}^k)^T(x^{k+1}-\bar{x}^k)
$$
Plugging this inequality into the objective gap we wanted to define lets us simplify along with the descent lemma from the $L$-smoothness of $f$.
$$
\begin{gather*}
\psi(x^{k+1})-\bar{\psi}=f(x^{k+1})-f(\bar{x}^k)+\varphi(x^{k+1})-\varphi(\bar{x}^k)\\
\psi(x^{k+1})-\bar{\psi}\leq f(x^{k+1})-f(\bar{x}^k)-\nabla f(\bar{x}^k)^T(x^{k+1}-\bar{x}^k)\\
\psi(x^{k+1})-\bar{\psi}\leq\frac{L}{2}\|x^{k+1}-\bar{x}^k\|^2
\end{gather*}
$$
To bound $\|x^{k+1}-\bar{x}^k\|$, we can use the triangle inequality, our definition of $d^k$, and the error bound $\|x^k-\bar{x}^k\|\leq \tau\|r(x^k)\|$.
$$
\begin{gather*}
\|x^{k+1}-\bar{x}^k\|\leq\|x^{k+1}-x^k\|+\|x^k-\bar{x}^k\|\\
\|x^{k+1}-\bar{x}^k\|\leq\|d^k\|+\|x^k-\bar{x}^k\|\\
\|x^{k+1}-\bar{x}^k\|\leq\|d^k\|+\tau\|r(x^k)\|
\end{gather*}
$$
From the second inequality of the first lemma of this proof, we know that $\|r(x^k)\|\leq C_r\|d_k\|$, which we can use to bound the previous optimality gap by defining $C_r=\frac{m+1}{m}(M+\mu_k)$ and then $C_L=\frac{L}{2}(1+\tau C_r)^2$ for simplicity.
$$
\begin{gather*}
\|x^{k+1}-\bar{x}^k\|\leq\|d^k\|+\tau C_r\|d^k\|=(1+\tau C_r)\|d^k\|\\
\psi(x^{k+1})-\bar{\psi}\leq\frac{L}{2}(1+\tau C_r)^2\|d^k\|^2=C_L\|d^k\|^2
\end{gather*}
$$
To make this bound meaningful, we need to replace the $d^k$ term. For any successful step we know $\text{ared}_k\geq c_1\text{pred}_k$ holds, which combined with $\text{pred}_k\geq\frac{m}{2}\|d^k\|^2$ from the previous lemma gives the following (which also holds trivially for unsuccessful steps).
$$
\begin{gather*}
\psi(x^k)-\psi(x^{k+1})\geq\frac{c_1m}{2}\|d^k\|^2\\
\|d^k\|^2\leq\frac{2}{c_1m}(\psi(x^k)-\psi(x^{k+1}))
\end{gather*}
$$
We can plug this into the previous suboptimality gap bound we had to get the desired convergence result by defining $K=\frac{2C_L}{c_1m}$. If we define $\theta=\frac{K}{1+K}$, we are guaranteed $\theta\in(0,1)$ since $K>0$, proving the convergence result shown.
$$
\begin{gather*}
\psi(x^{k+1})-\bar{\psi}\leq C_L\left[\frac{2}{c_1m}(\psi(x^k)-\psi(x^{k+1}))\right]\\
(1+K)(\psi(x^{k+1})-\bar{\psi})\leq K(\psi(x^k)-\bar{\psi})\\
\psi(x^{k+1})-\bar{\psi}\leq\frac{K}{1+K}(\psi(x^k)-\bar{\psi})
\end{gather*}
$$
Finally, to prove the convergence to a single iterate $\bar{x}$, we use the bound on $\|d^k\|^2$. Since the difference between consecutive objectives is bounded by the distance in the limit (the objectives of two iterates can't be farther than the distance between initial and fully optimal), we can simplify.
$$
\begin{gather*}
\|x^{k+1}-x^{k}\|=\|d^k\|\leq\sqrt{\frac{2}{c_1m}(\psi(x^k)-\psi(x^{k+1}))}\\
\|d^k\|\leq\sqrt{\frac{2}{c_1m}}\sqrt{\psi(x^k)-\bar{\psi}}
\end{gather*}
$$
Since the term inside the square root shrinks at a rate $\theta$, we know that $\|d^k\|$ shrinks at a rate of $\sqrt{\theta}$, which since $0<\theta<1$, we have $\sqrt{\theta}<1$ as well. This means that a sum of all $d^k$ creates a geometric series with a common ratio less than $1$, which converges to a finite number.
$$
\sum^\infty_{k=0}\|x^{k+1}-x^k\|=\sum^\infty_{k=0}\|d^k\|<\infty
$$
This means that $\{x^k\}$ is a Cauchy Sequence, which must converge to a unique limit point, completing the proof.