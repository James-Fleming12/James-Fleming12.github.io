---
title: Frank-Wolfe and Convergence
description: Projected Gradient Descent, Frank-Wolfe, and their convergence on convex and non-convex problems
pubDate: 2/23/2026
---

This is the third of the quick summaries for the lab review. Going to cover Projected Gradient Descent for some background, Frank-Wolfe for some modern techniques, and their convergence on well-behaved functions. Also going to cover some new-ish convergence proofs for Frank-Wolfe on non-convex settings.

## Prerequisites:
I'm going to assume that properties like convexity, L-smoothness, and standard Gradient Descent terminology are known, however I am going to prove one property of L-smooth functions. Any L-smooth function is upper bounded by some quadratic (shown below for any $x,y\in\mathbb{R}^n$).
$$
f(y)\leq f(x)+\left<\nabla f(x),y-x\right>+\frac{L}{2}\|y-x\|^2
$$
For the proof, we can first define the distance $f(y)-f(x)$ with the line integral of the segment of the function between the two points, which is derived from some basic calculus properties.
$$
\begin{gather*}
g(1)=g(0)+\int^1_0g^\prime(t)dt\quad \phi(t)=x+t(y-x)\quad g(t)=f(\phi(t))\\
g^\prime=\nabla f(\phi(t))\cdot\phi^\prime(t)\quad \phi^\prime(t)=y-x\\
g^\prime(t)=\left<\nabla f(x+t(y-x)),y-x\right>\\
f(y)=f(x)+\int^1_0\left<\nabla f(x+t(y-x)),y-x\right>dt
\end{gather*}
$$
We can then add and subtract $\left<\nabla f(x),y-x\right>$ inside the integral to get it into a form that works well with the L-smooth property.
$$
f(y)=f(x)+\left<\nabla f(x),y-x\right>+\int^1_0\left<\nabla f(x+t(y-x))-\nabla f(x),y-x\right>dt
$$
After this we can now use the Cauchy-Schwarz Inequality and the L-smoothness property to upper bound the integral.
$$
\begin{gather*}
\int^1_0\left<\nabla f(x+t(y-x))-\nabla f(x),y-x\right>dt\leq\dots\\\dots\int^1_0\|\nabla f(x+t(y-x))-\nabla f(x)\|\cdot\|y-x\|dt\\
\|\nabla f(x+t(y-x))-\nabla f(x)\|\leq L\|x+t(y-x)-x\|=Lt\|y-x\|\\
\int^1_0(Lt\|y-x\|)\cdot\|y-x\|dt=L\|y-x\|^2\int^1_0tdt=\frac{L}{2}\|y-x\|^2
\end{gather*}
$$
Using this final bound on the first definition gives the desired property.
$$
\begin{gather*}
\int^1_0\left<\nabla f(x+t(y-x))-\nabla f(x),y-x\right>dt\leq\frac{L}{2}\|y-x\|^2\\
f(y)\leq f(x)+\left<\nabla f(x),y-x\right>+\frac{L}{2}\|y-x\|^2
\end{gather*}
$$

## Projected Gradient Descent:
Consider a constrained optimization problem over a set $\Omega$ defined below.
$$
\min_{x\in\Omega}f(x)
$$
One of the most basic ways to solve this problem is Projected Gradient Descent, which extends Gradient Descent by simply adding a projection step after the gradient step. The projection $\Pi_\Omega(x)$ returns the point within $\Omega$ that is closest to $x$.
$$
\begin{gather*}
\Pi_\Omega(x)=\text{arg }\min_{y\in\Omega}\|x-y\|\\
x^{(t+1)}=\Pi_\Omega(x^{(t)}-\mu\nabla f(x^{(t)}))
\end{gather*}
$$
Under settings where both $f(x)$ and $\Omega$ are convex, this method takes good advantage of the benefits that gradient descent gives.

## PGD Non-Increasing Iterates:
To properly evaluate the convergence properties of the method, we first need to prove the guarantee that under an L-Smooth function, a convex $\Omega$, and a step-size $\mu=\frac{1}{L}$, then we know that the iterates $x^{(t)}$ are non-increasing.
$$
f(x^{(t)})\leq f(x^{(t-1)})
$$
For the proof, we can first see that the algorithm can be redefined as a minimization subproblem for each iterate by using the definition of the projection. Convexity being required for the set not only guarantees that the solution to this subproblem is unique, but also is a general requirement for this property to be meaningful, since it's required to guarantee that the algorithm will reach the true minimum at all. We scale the term by $\frac{L}{2}$ for simplicity later since it does not change the final minimum.
$$
x^{(t+1)}=\text{arg}\min_{z\in\Omega}\left\{\frac{L}{2}\left\|z-\left(x^{(t)}-\frac{1}{L}\nabla f(x^{(t)})\right)\right\|^2\right\}
$$
Using the quadratic upper bounding property from the fact that $f$ is L-smooth, we can define a surrogate function $Q_x(z)$ below for use later. This definition intuitively means that $f(x^{(t+1)})\leq Q_{x^{(t)}}(x^{(t+1)})$ and $f(x^{(t)})=Q_{x^{(t)}}(x^{(t)})$.
$$
Q_x(z)=f(x)+\left<\nabla f(x),z-x\right>+\frac{L}{2}\|z-x\|^2
$$
Expanding the square norm of the original minimization problem and isolating terms with $z$ (which are the only ones that impact the minimization) gives $\left<\nabla f(x),z-x\right>+\frac{L}{2}\|z-x\|^2$, which is equivalent to our surrogate $Q_x(z)$ in the minimization. This means we can redefine the subproblem with the surrogate.
$$
x^{(t+1)}=\text{arg}\min_{x\in\Omega}Q_{x^{(t)}}(z)
$$
By definition, we know that $f(x^{(t+1)})\leq Q_{x^{(t)}}(x^{(t+1)})$ and $f(x^{(t)})=Q_{x^{(t)}}(x^{(t)})$. As well, since $x^{(t+1)}$ is the minimizer of the subproblem over all $\Omega$ and $x^{(t)}$ is within $\Omega$, we know $Q_{x^{(t)}}(x^{(t+1)})\leq Q_{x^{(t)}}(x^{(t)})$. Combining these together gives our desired result.
$$
f(x^{(t+1)})\leq Q_{x^{(t)}}(x^{(t+1)})\leq Q_{x^{(t)}}(x^{(t)})=f(x^{(t)})
$$

## PGD Convex Convergence:
Now that we've proved that iterates never get worse, we can derive what convergence rate PGD has under for well-behaved problems. If $f$ is an L-Smooth Convex Function over a Convex Set $\Omega$ and $\mu=\frac{1}{2}$, we can prove that PGD has a convergence rate of $\mathcal{O}(1/k)$.
$$
f(x^{(k)})-f(x^*)\leq\frac{L\|x^{(0)}-x^*\|^2}{2k}
$$
The proof is a standard one in optimization, simply using the different properties to derive some inequalities, combining them, and then using the derived properties to prove some global behavior.

From L-smoothness and convexity we know that the following hold by definition.
$$
\begin{gather*}
f(x^{(t+1)})\leq f(x^{(t)})+\left<\nabla f(x^{(t)}),x^{(t+1)}-x^{(t)}\right>+\|\frac{L}{2}x^{(t+1)}-x^{(t)}\|^2\\
f(x^*)\geq f(x^{(t)})+\left<\nabla f(x^{(t)}),x^*-x^{(t)}\right>
\end{gather*}
$$
We can rearrange the convexity inequality to replkace $f(x^{(t)})$ in the L-smoothness inequality and the remove like terms to get the following.
$$
f(x^{(t+1)})\leq f(x^*)+\left<\nabla f(x^{(t)}),x^{(t+1)}-x^*\right>+\frac{L}{2}\|x^{(t+1)}-x^{(t)}\|^2
$$
Since $\Omega$ is a convex set, we can use the a property of projections on convex sets that guarantees that for all $y$ $\left<p-z,y-p\right>\geq 0$ with any $z$ and its projection $p$. We can set $z=x^{(t)}-\frac{1}{L}\nabla f(x^{(t)})$ and $p=x^{(t+1)}$ to get the inequality in terms of the algorithm, and then rearrange and set $y=x^*$ to add some more meaning.
$$
\begin{gather*}
\left<x^{(t+1)}-x^{(t)}+\frac{1}{L}\nabla f(x^{(t)}),y-x^{(t+1)}\right>\\
\left<\nabla f(x^{(t)},x^{(t+1)}-x^*)\right>\leq L\left<x^{(t)}-x^{(t+1)},x^{(t+1)}-x^*\right>
\end{gather*}
$$
Using this we can replace the middle term of the convexity and L-smoothness inequality with this new projection inequality.
$$
f(x^{(t+1)})-f(x^*)\leq \frac{L}{2}\left(2\left<x^{(t)}-x^{(t+1)},x^{(t+1)}-x^*\right>+\|x^{(t+1)}-x^{(t)}\|^2\right)
$$
Using the property of squared norms that $\|a+b\|^2=\|a\|^2+2\left<a,b\right>+\|b\|^2$ and rearranging to $2\left<a,b\right>+\|a\|^2=\|a+b\|^2-\|b\|^2$ we can replace the right-hand side of the inequality with another simpler term. Setting $a=x^{(t)}-x^{(t+1)}$ and $b=x^{(t+1)}-x^*$ gives us the following.
$$
f(x^{(t+1)})-f(x^*)\leq\frac{L}{2}\left(\|x^{(t)}-x^*\|^2-\|x^{(t+1)}-x^*\|^2\right)
$$
We can see that adding the inequality of $f(x^{(t)})-f(x^*)$ and that of $f(x^{(t+1)})$ would result in the terms $\|x^{(t+1)}-x^*\|^2$ in both cancelling out. Summing over $k$ iterations gives us a telescoping sum since this continues, giving us the following.
$$
\sum^{k-1}_{t=0}f(x^{(t+1)})-f(x^*)\leq\frac{L}{2}\left(\|x^{(0)}-x^*\|^2-\|x^{(k)}-x^*\|^2\right)
$$
Since we know that all of the iterates are non-increasing, we can simplify the sum on the left-hand side to be only in terms of our final iterate, giving us our desired result.
$$
\begin{gather*}
\sum^{k+1}_{t=0}(f(x^{(t+1)})-f(x^*))\geq\sum^{k-1}_{t=0}(f(x^{(k)})-f(x^*))=k(f(x^{(k)})-f(x^*))\\
k(f(x^{(k)})-f(x^*))\leq\frac{L}{2}\|x^{(0)}-x^*\|^2
\end{gather*}
$$

## Frank-Wolfe:
One problem with Projected Gradient Descent is that for more complex settings the projection step itself can be very expensive since it requires solving a quadratic optimization problem. Frank-Wolfe, also known as Conditional Gradient, is an optimization method over the same type of constrained problem that replaces this quadratic subproblem with a linear subproblem at each iteration. 

Frank-Wolfe works off of the idea of linearizing and minimizing, where a linear approximation of the function is first created and then minimized over. The most basic form of this is the Linear Minimization Oracle, which chooses an $s^{(k)}\in\Omega$ that minimizes over the current iterate's gradient's direction. The iterate is then updated by moving towards the direction of the chosen $s^{(k)}$ with step-size $\gamma^{(k)}$.
$$
\begin{gather*}
s^{(k)}=\text{arg}\min_{s\in\Omega}s^T\nabla f(x^{(k)})\\
x^{(k+1)}=x^{(k)}+\gamma^{(k)}(s^{(k)}-x^{(k)})
\end{gather*}
$$
By definition we can see that $x^{(k+1)}=x^{(k)}+\gamma^{(k)} s^{(k)}-\gamma^{(k)}x^{(k)}=(1-\gamma^{(k)})x^{(k)}+\gamma^{(k)}s^{(k)}$, which means that the new iterate is derived from a convex combination of two points within $\Omega$, thus the new iterate is also in $\Omega$ as long as $\Omega$ is a convex set. This presents a projection-free method of constrained optimization that is much quicker for each iteration when compared to projected gradient descent in complex problems.

## Frank-Wolfe Convex Convergence:
Along with being faster per iteration, we can also prove that the method has the same convergence rate under well-behaved functions as projected gradient descent. On a L-Smooth Convex function over a Compact Convex Set $\Omega$, Frank-Wolfe with the LMO also has a convergence rate of $O(1/k)$ if the stepsize is chosen as $\gamma^{(k)}=\frac{2}{k+2}$. The convergence rate is bounded by the Curvature Constant $C_f$, which conveys how much $f$ deviates from the linear approximation made for each iteration.
$$
\begin{gather*}
f(x^{(k)})-f(x^*)\leq\frac{2C_f}{k+2}\\
C_f=\sup_{x,s\in\Omega,\gamma\in[0,1],y=x+\gamma(s-x)}\frac{2}{\gamma^2}(f(y)-f(x)-\left<y-x,\nabla f(x)\right>)
\end{gather*}
$$
The proof follows a very similar structure as before, where all of the properties are used to bound certain quantities, which are then used to derive a recurrence relation which allows us to derive our desired result.

We can first prove that $C_f$ is finite with the L-smoothness property so that we can make statements using it. We can use the quadratic lower bounding property to show that the following holds with $y=x+\gamma(s-x)$.
$$
\begin{gather*}
f(x+\gamma(s-x))-f(x)-\left<\nabla f(x),\gamma(s-x)\right>\leq\frac{L}{2}\|\gamma(s-x)\|^2\\
f(y)-f(x)-\left<\nabla f(x),y-x\right>\leq\frac{L\gamma^2}{2}\|s-x\|^2
\end{gather*}
$$
We can then substitute this inequality into our definition of $C_f$ to upper bound it.
$$
\begin{gather*}
C_f\leq\sup_{x,s\in\Omega}\frac{2}{\gamma^2}\left(\frac{L\gamma^2}{2}\|s-x\|^2\right)\\
C_f\leq\sup_{x,s\in\Omega}L\|s-x\|^2
\end{gather*}
$$
Since $\Omega$ is compact, we know that $\|s-x\|^2$ is finite, so we know that $C_f$ is finite.

We can then rearrange the definition of $C_f$ as an upper bound, using our previously defined $y=x+\gamma(s-x)$, and rewrite it in terms of our iterates.
$$
\begin{gather*}
C_f\geq \frac{2}{\gamma^2}(f(y)-f(x)-\left<y-x,\nabla f(x)\right>)\\
f(x+\gamma(s-x))\leq f(x)+\gamma\left<s-x,\nabla f(x)\right>+\frac{\gamma^2}{2}C_f\\
f(x^{(k+1)})\leq f(x^{(k)})+\gamma^{(k)}\left<s^{(k)}-x^{(k)},\nabla f(x^{(k)})\right>+\frac{\gamma^{(k)2}}{2}C_f
\end{gather*}
$$
For this to be meaningful, we need a relationship between $s^{(k)}$ and the optimal point $x^*$. Since $s^{(k)}$ is defined as the minimum of $\left<s,\nabla f(x^{(k)})\right>$, we know that this is the upper bound of any point, including $x^*$.
$$
\begin{gather*}
\left<s^{(k)},\nabla f(x^{(k)})\right>\leq\left<x^*,\nabla f(x^{(k)})\right>\\
\left<s^{(k)}-x^{(k)},\nabla f(x^{(k)})\right>\leq\left<x^*-x^{(k)},\nabla f(x^{(k)})\right>
\end{gather*}
$$
Since $f$ is convex we can apply the definition to the above inequality, with $h^{(k)}=f(x^{(k)}-f(x^*))$ being used to defin the optimality gap for simplicity.
$$
\begin{gather*}
\left<x^*-x^{(k)},\nabla f(x^{(k)})\right>\leq f(x^*)-f(x^{(k)})=-h^{(k)}\\
\left<s^{(k)}-x^{(k)},\nabla f(x^{(k)})\right>\leq -h^{(k)}
\end{gather*}
$$
We can now use this to upper bound our previous inequality, deriving the following.
$$
\begin{gather*}
f(x^{(k+1)})\leq f(x^{(k)})-\gamma^{(k)}h^{(k)}+\frac{\gamma^{(k)2}}{2}C_f\\
h^{(k+1)}\leq h^{(k)}(1-\gamma^{(k)})+\frac{\gamma^{(k)2}}{2}C_f
\end{gather*}
$$
Now that we have a recurrence relation, we can use induction on it to prove $h^{(k)}\leq\frac{2C_f}{k+2}$, deriving our desired result.
$$
\begin{gather*}
h^{(k+1)}\leq\frac{2C_f}{k+2}\left(1-\frac{2}{k+2}\right)+\frac{2C_f}{(k+2)^2}\\
h^{(k+1)}\leq\frac{2C_f}{k+2}\left(\frac{k}{k+2}\right)+\frac{2C_f}{(k+2)^2}=2C_f\left(\frac{k}{(k+2)^2}+\frac{1}{(k+2)^2}\right)\\
h^{(k+1)}\leq 2C_f\frac{k+1}{(k+2)^2}=2C_f\frac{(k+1)(k+3)}{(k+2)^2(k+3)}\\
h^{(k+1)}\leq\frac{2C_f}{k+3}
\end{gather*}
$$

## Frank-Wolfe Non-Convex Convergence:
