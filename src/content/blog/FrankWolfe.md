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
A convergence rate for Frank-Wolfe has also been derived in [non-convex](https://arxiv.org/abs/1607.00345) settings. Since we're now dealing with non-convex functions, we need to describe convergence in terms other than our original optimality gap. Instead, we use the Frank-Wolfe gap $g_t$. This describes how much the objective can be decreased by moving towards an element in the feasible set, which allows $g_t=0$ to act as an analogous term to $\|\nabla f(x)\|=0$ in our setting.
$$
g_t=\max_{s\in\Omega}\left<s-x^{(t)},-\nabla f(x^{(t)})\right>
$$
Since we're in a non-convex setting, we don't have the guarantee that the iterates always improve, so we need to denote the convergence in terms of the minimal Frank-Wolfe gap $\tilde{g}_t=\min_{0\leq k\leq t}$ after $t$ iterations. The minimal Frank-Wolfe gap on an L-Smooth, potentially non-convex, $f$ over a colmpact convex set $\Omega$ after $t$ iterations can be bounded by $O(\frac{1}{\sqrt{t}})$ as shown below.
$$
\begin{gather*}
\tilde{g}_t\leq\frac{\max\{2h_0,C\}}{\sqrt{t+1}}\quad t\geq 0\\
h_0=f(x^{(0)})-\min_{x\in\mathcal{M}}f(x)
\end{gather*}
$$
$h_0$ denotes the initial suboptimality and $C$ is some constant that satisfies $C\geq C_f$ that is determined by our stepsize. If a line-search algorithm is used to find the optimal step-size, then $C=C_f$. If an adaptive stepsize is chosen $\min\{\frac{g_t}{C},1\}$ (will be derived later), then $C\geq C_f$.

The proof follows the same general form of using the properties of $f$ to derive a recursive bound and then using it to prove global convergence properties. We can first derive a more practical form of the L-Smooth descent lemma using the definition of $C_f$, which we know is finite from $f$ being L-Smooth and $\Omega$ being compact.
$$
\begin{gather*}
\frac{2}{\gamma^2}[f(y)-f(x)-\left<\nabla f(x),y-x\right>]\leq C_f\\
f(y)\leq f(x)+\left<\nabla f(x),y-x\right>+\frac{\gamma^2}{2}C_f
\end{gather*}
$$

We can then define $x_\gamma=x^{(t)}+\gamma d_t$ (the potential next step without a specific $\gamma$) and $d_t=s^{(t)}-x^{(t)}$ (the direction of the step) to get the bound in terms of our algorithm. We can also further simplify by using the definition of $g_t$ as an upper bound of the middle term and replacing $C_f$ with our previously defined $C$.
$$
\begin{gather*}
f(x_\gamma)\leq f(x^{(t)})+\gamma\left<\nabla f(x^{(t)}),d_t\right>+\frac{\gamma^2}{2}C_f\quad\forall\gamma\in[0,1]\\
f(x_\gamma)\leq f(x^{(t)})-\gamma g_t+\frac{\gamma^2}{2}C
\end{gather*}
$$
To minimize this bound on $f(x_\gamma)$, we can choose a stepsize $\gamma^*=\min\{\frac{g_t}{C},1\}$. This breaks the decrease of the objective at each step into a piecewise based on whether $g_t\leq C$, leading to a stepsize of $\frac{g_t}{C}$, or $g_t>C$, leading to a stepsize of $1$. We can also replace the piecewise with an indicator function as shown below.
$$
\begin{gather*}
f(x^{(t+1)})\leq f(x^{(t)})-\begin{cases}\frac{g^2_t}{2c}&\text{if }g_t\leq C\\g_t-\frac{C}{2}&\text{if }g_t>C\end{cases}\\
f(x^{(t+1)})\leq f(x^{(t)})-\min\left\{\frac{g_t^2}{2C},g_t-\frac{C}{2}\mathbb{1}_{\{g_t>C\}}\right\}
\end{gather*}
$$
In cases where $g_t>C$, we have $\min\left\{\frac{g_t^2}{2C},g_t-\frac{C}{2}\right\}$. We can define the difference between the two as a function $f(g_t)=\frac{g^2_t}{2C}-(g_t-\frac{C}{2})=\frac{g^2_t-2Cg_t+C^2}{2C}=\frac{(g_t-C)^2}{2C}$. Since we know that $(g_t^2-C)^2\geq 0$, we know that the difference is positive so $\frac{g_t^2}{2C}\geq g_t-\frac{C}{2}$, thus the correct min is chosen. In cases where $g_t\leq C$, we have $\min\left\{\frac{g_t^2}{2C},g_t\right\}$. We can first divide by $g_t$, which we know is positive, and get $\min\{\frac{g_2}{2C},1\}$. Since $g_t\leq C$, we know $\frac{g_t}{2C}\leq \frac{1}{2}$, thus the correct min is chosen.

Since this forms a recursive relationship, we can somver over $t$ iterations since it forms a telescoping sum. We can then simplify by replacing the sum over decreaes with the minimal gap $\tilde{g}_t$ since we know that each term is always positive, so replacing the subtracted terms with a smaller subtracted term keeps the $\leq$ bound.
$$
\begin{gather*}
f(x^{(t+1)})\leq f(x^{(0)})-\sum^t_{k=0}\min\left\{\frac{g_k^2}{2C},g_k-\frac{C}{2}\mathbb{1}_{\{g_k>C\}}\right\}\\
f(x^{(t+1)})\leq f(x^{(0)})-(t+1)\min\left\{\frac{\tilde{g}_t^2}{2C},\tilde{g}_t-\frac{C}{2}\mathbb{1}_{\{\tilde{g}_t>C\}}\right\}
\end{gather*}
$$
This means we can bound $\tilde{g}_t$ with these, specifically using the fact that $f(x^{(0)})-f(x^{(t+1)})\leq f(x^{(0)})-f(x^*)=h_0$.
$$
\tilde{g}_t\leq\begin{cases}\frac{h_0}{t+1}+\frac{C}{2}&\tilde{g}_t>C\\\sqrt{\frac{2h_0C}{t+1}}&\tilde{g}_t\leq C\end{cases}
$$
Since we're trying to bound $\tilde{g}_t$, we would like to derive a new set of conditions for the piecewise to get rid of the circular logic. While an equivalent condition is hard to find, we can find a set of conditions that guarantees that the bounds hold. By simple algebraic manipulation, we can know that $\tilde{g}_t>C$ implies $t+1\leq \frac{2h_0}{C}$. If $\tilde{g}_t\leq C$, so any $\tilde{g}_t$ has to be cateogized in the first case. If we have some $\tilde{g}_t\leq C$ and $t+1\leq\frac{2h_0}{C}$, we can show that the first bound also holds.
$$
\tilde{g}_t\leq C\leq \frac{C}{2}+\frac{C}{2}=
\frac{h_0}{\frac{2h_0}{C}}+\frac{C}{2}\leq\frac{h_0}{t+1}+\frac{C}{2}
$$
Thus we can rewrite the conditions based on this.
$$
\tilde{g}_t\leq\begin{cases}\frac{h_0}{t+1}+\frac{C}{2}&\text{if }t+1\leq\frac{2h_0}{C}\\\sqrt{\frac{2h_0C}{t+1}}&\text{otherwise}\end{cases}
$$
Since we have two different convergence rates based on these conditions, we need to get them into a similar form to provide an overall convergence rate. We can first manipulate the first bound to get it into a workable form.
$$
\begin{gather*}
\frac{h_0}{t+1}+\frac{C}{2}=\frac{h_0}{\sqrt{t+1}}\left(\frac{1}{\sqrt{t+1}}+\frac{C}{2h_0}\sqrt{t+1}\right)\\
\frac{h_0}{t+1}+\frac{C}{2}\leq\frac{h_0}{\sqrt{t+1}}\left(\frac{1}{\sqrt{t+1}}+\sqrt{\frac{C}{2h_0}}\right)
\end{gather*}
$$
In order to work with this, we need to prove that $h_0>\frac{C}{2}$ if $t+1\leq\frac{2h_0}{C}$. Consider the case where the condition holds and $h_0\leq\frac{C}{2}$. This would lead to $t+1\leq \frac{2h_0}{C}\leq \frac{2}{C}\cdot\frac{C}{2}=1$, which leads to a contradiction since we know that $t\geq 0$.

Using this fact, we can know that $\frac{C}{2}\cdot\frac{1}{h_0}<1$, so we can substitute and simplify, deriving our desired result.
$$
\frac{h_0}{t+1}+\frac{C}{2}\leq\frac{h_0}{\sqrt{t+1}}\left(\frac{1}{\sqrt{t+1}}+1\right)\leq\frac{2h_0}{\sqrt{t+1}}
$$
This leaves us with the following two bounds.
$$
\tilde{g}_t\leq\begin{cases}\frac{2h_0}{\sqrt{t+1}}&\text{if }t+1\leq\frac{2h_0}{C}\\\frac{\sqrt{2h_0C}}{\sqrt{t+1}}&\text{otherwise}\end{cases}
$$
To further simplify, we can show that both bounds can have their numerators be replaced with $\max\{2h_0,C\}$. In the first case, since we know that $h_0>\frac{C}{2}$, we know that $2h_0>C$, which derives our desired numerator $2h_0=\max\{2h_0,C\}$. For the second, we know that since both $h_0$ and $C$ are nonnegative we have $\sqrt{2h_0C}\leq\max\{2h_0,C\}$ (given some arbitrary $a,b\geq 0$ and that $a\geq b$, we know that $\sqrt{ab}\leq\sqrt{a^2}=a$). Replacing the numerators then combines the conditions and derives our final result, proving the theorem.
$$
\tilde{g}_t\leq\frac{\max \{2h_0,C\} }{\sqrt{t+1}}
$$