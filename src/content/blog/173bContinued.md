---
title: Basic Primal-Dual Optimization Methods
description: Dual Ascent, Augmented Lagrangian Method of Multipliers, ADMM, and their convergence
pubDate: 3/28/2026
---

This is going to be a continuation of the second half of MATH 173B at UCSD (which at the time of writing was Optimization Methods in Data Science II), which covered Dual Ascent, the Augmented Lagrangian Method of Multipliers, and ADMM. The original lecture notes didn't cover the convergence theorems and proofs at a depth that I wanted, so I thought that I'd write something to extend them. The proof of Danskin's Theorem and Finsler's Lemma, which are used in the convergence proofs, are also included at the end to round things out.

## Dual Ascent:
For simplicity we consider an equality constrained problem of the following form.
$$
\begin{gather*}
\min f(x)\\
\text{s.t. }Ax=b
\end{gather*}
$$
For all algorithms shown here, one can simply replace the constraints in each of the parts of the algorithm to use another set of constraints. As a refresher we can define the dual problem for this using the Lagrangian $L(x,y)=f(x)+y^T(Ax-b)$ and the Lagrange Dual $g(y)=\min_xL(x,y)$ below.
$$
\begin{gather*}
\max_y g(y)
\end{gather*}
$$
Since this removes the need to explicitly handle the constraint, we could consider running Gradient Ascent on the dual problem rather than running Gradient Descent on the primal, which allows us to derive the update rule for Dual Ascent.
$$
y^{(t+1)}=y^{(t)}+\alpha_t(\nabla g(y^{(t)}))
$$
This leads to the following algorithm for the problem we considered originally since $\nabla g(y)=Ax^*(y)-b$, where $x^*(y)$ is the optimal $x$ given some $y$.
$$
\begin{gather*}
x^{(t+1)}=\text{arg}\min_xL(x,y^{(t)})\\
y^{(t+1)}=y^{(t)}+\alpha_t(Ax^{(t+1)}-b)
\end{gather*}
$$

One of the main benefits of this method to be taken into consideration in later sections is that it can be parallelized in many real world applications. Consider the optimization of a $f(w)$ of the following form, one that is commonly found in applications of optimization over a dataset (which has $w$ being split into many $x_i$ terms to allow for parallelization).
$$
f(w)=\sum^n_{i=1}f_i(w)=\sum^n_{i=1}f_i(x_i)
$$
Due to the structure of the Lagrangian we can separate the Lagrangian over the variables of $x$.
$$
\begin{gather*}
L(x,y)=\left[\sum^n_{i=1}f_i(x_i)\right]+[y^T(Ax-b)]=\sum^n_{i=1}L_i(x_i,y)\\
L_i(x,y)=f_i(x)+y^Ta_ix_i-\frac{y^Tb}{n}
\end{gather*}
$$
Due to this, we can separate the optimization step over $x$ within the algorithm itself, letting us parallelize it.
$$
\begin{gather*}
x^{(t+1)}_i=\text{arg}\min_{x_i}L_i(x_i,y^{(t)})\\
y^{(t+1)}=y^{(t)}+\alpha_t(Ax^{(t+1)}-b)
\end{gather*}
$$

### Convergence:
Given $f(x)$ is $m$-strongly convex and $L$-smooth and $0<\alpha<\frac{2m}{\|A\|^2}$, we know that $Ax^{(t)}-b\rightarrow 0$, $x^{(t)}\rightarrow x^*$, and $y^{(t)}\rightarrow y^*$.

The proof follows by showing that Dual Ascent makes constant improvements on $g(y)$ and the fact that $x^*=\text{arg}\min_xL(x,y^*)$.

We can first use the assumption that $f$ is strongly convex to prove that $g(y)$ is L-smooth. This can be trivially done with known properties or even the connection of $g(y)$ to the convex conjugate, but for completeness we can derive it from scratch here through algebraic manipulation of two points $x_1=x(y_1)$ and $x_2=x(y_2)$, where $x(y)=\text{arg}\min_xL(x,y)$. Since $x_1$ and $x_2$ minimize the Lagrangian, we know that they satisfy the optimality conditions below.
$$
\nabla f(x_1)+A^Ty_1=0\quad\nabla f(x_2)+A^Ty_2=0
$$
We can subtract these to derive a statement involving both points.
$$
\nabla f(x_1)-\nabla f(x_2)=A^T(y_2-y_1)
$$
Since $f$ is $m$-strongly convex, we know that the following holds from an equivalent condition of strongly convex functions and then simplify using the previous statement.
$$
\begin{gather*}
m\|x_1-x_2\|^2\leq(x_1-x_2)^T(\nabla f(x_1)-\nabla f(x_2))\\
m\|x_1-x_2\|^2\leq (x_1-x_2)^TA^T(y_2-y_1)
m\|x_1-x_2\|^2
\end{gather*}
$$
We can then simplify using the Cauchy-Schwartz Inequality and then divide by $m\|x_1-x_2\|$ to get the definition of $x(y)$ being lipschitz.
$$
\begin{gather*}
m\|x_1-x_2\|^2\leq\|A(x_1-x_2)\|\|y_2-y_1\|\leq\|A\|\|x_1-x_2\|\|y_2-y_1\|\\
\|x_1-x_2\|\leq\frac{\|A\|}{m}\|y_1-y_2\|
\end{gather*}
$$
By Danskin's Theorem we know that $\nabla g(y)=Ax(y)-b$, and since the linear transformation of an L-smooth function is also L-smooth, we have proven that $g(y)$ is L-smooth. This means we can use the Descent Lemma on $g(y)$ between $y^{(t+1)}$ and $y^{(t)}$, and then use $y^{(t+1)}-y^{(t)}=\alpha\nabla g$ to simplify.
$$
\begin{gather*}
g(y^{(t+1)})\geq g(y^{(t)})+\nabla g(y^{(t)})^T(y^{(t+1)}-y^{(t)})-\frac{L}{2}\|y^{(t+1)}-y^{(t)}\|^2\\
g(y^{(k+1)})-g(y^{(k)})\geq\alpha\|\nabla g\|^2-\frac{L\alpha^2}{2}\|\nabla g\|^2=\alpha\left(1-\frac{\alpha L}{2}\right)\|\nabla g\|^2
\end{gather*}
$$
This shows that $g(y^{(t+1)})>g(y^{(t)})$ if we have $\alpha(1-\frac{\alpha L}{2})>0$, which simplifies to $\alpha<\frac{2}{L}$. With this step-size we know that as long as $\|\nabla g(y^{(t)})\|\neq 0$, we are making constant progress on reaching optimality. This also proves $g(y^{(t)})\rightarrow g(y^*)$ which will be used later.

Since $g(y)$ is bounded from above by the primal optimal value $p^*$, the sum of the increases must also be bounded since it forms a telescoping sum, which is detailed below. Since this is still running a standard gradient descent algorithm we know that the iterates will have non-increasing objective values, so the only way for this to be the case is if $\|\nabla g(y^{(t)})\|\rightarrow 0$ (since $g(y^{(t+1)})-g(y^{(t)})\approx\alpha\|\nabla g(y^{(t)})\|^2$), which proves that primal faesibility is reached.
$$
\sum^\infty_{k=0}g(y^{(t+1)})-g(y^{(t)})\leq p^*-g(y^{(0)})<\infty
$$
To prove that $y\rightarrow y^*$, we can first prove that the dual problem is strongly concave given the assumptions made. Since $f$ is strongly convex, we know that $\nabla^2f(x)\succeq mI$ and that $(\nabla^2f)^{-1}\preceq\frac{1}{m}I$ from the inversion property of eigenvalues. Since $A$ is full rank, we know the smallest eigenvalue of $AA^T$ is greater than zero, letting the following prove that $g(y)$ is strongly concave.
$$
\begin{gather*}
\nabla^2 g(y)=-A(\nabla^2f(x(y)))^{-1}A^T\preceq -\frac{1}{m}AA^T\\
\nabla^2g(y)\preceq-\frac{\sigma_\text{min}(AA^T)}{m}I
\end{gather*}
$$
Since $g(y)$ is strongly concave, we know that it satisfies the PL-Condition, which we can use to write the following inequality which along with $g(y^{(t)})\rightarrow g(y^*)$ proves that $y\rightarrow y^*$.
$$
g(y^*)-g(y^{(t)})\geq\frac{\sigma_\text{min}(AA^T)}{2m}\|y^{(t)}-y^*\|^2
$$
Finally, since we know that both primal feasibility and dual optimality are reached, we know that $x\rightarrow x^*$ from the fact that $x^*=\text{arg}\min_xL(x,y^*)$ and the fact that $f$ is strongly convex, leading to one minimum.

## Augmented Lagrangian Method of Multipliers:
The main problem with Dual Ascent is that it has some strict requirements to guarantee convergence. Problems that are not strictly convex or problems where the function can take values of $+\infty$ will not work, as the update for $x$ becomes a problem if the lagrangian does not have a unique minimum. The Augmented Lagrangian $L_\rho(x,y)$ tries to fix this issue by adding a penalty term to the Lagrangian, which allows for more stability under weaker assumptions.
$$
L_\rho(x,y)=f(x)+y^T(Ax-b)+\frac{\rho}{2}\|Ax-b\|^2_2
$$
The Augmented Lagrange Method of Multipliers, shortened ALM, then performs the exact same calculations as Dual Ascent with this new Augmented Lagrangian.
$$
\begin{gather*}
x^{(t+1)}=\text{arg}\min_{x}L_\rho(x,y^{(t)})\\
y^{(t+1)}=y^{(t)}+\rho(Ax^{(t+1)}-b)
\end{gather*}
$$

### Convergence:
The convergence theorem for ALM relies on converging to a local minimum, contrasting with the other two focusing on global minimums. Assume $f(x)$ and $Ax=b$ are twice continuously differentiable, $x^*$ is a local minimum satisfying second-order sufficient condition for the Lagrangian, and $A$ has full rank. With these we know that there exists a threshold $\hat{\rho}$ such that for all $\rho>\hat{\rho}$ we know $x^{(t)}\rightarrow x^*$, $y^{(t)}\rightarrow y^*$, and the convergence of $y^{(t)}$ is linear with a ratio proportional to $\frac{1}{\rho}$. A lot of the proof solely cares about the existence of a $\rho$, but in practical applications the $\rho$ that works best is still reasonable.

The proof follows by first ensuring there is some $\rho$ that can make $L_\rho$ convex, using it to guarantee that the algorithm's iterations are well-defined, and then proving the linear convergence rate by relating iterates with a contraction.

By definition of $x^*$ satisfying second order sufficient conditions, we know that $\nabla^2L(x^*,y^*)$ is positive definite on the null space of $A$. By Finsler's Lemma, we know that for large enough $\rho$ the augmented matrix (which forms the hessian of the augmented lagrangian) is positive definite everywhere, which means the augmented lagrangian is convex everywhere for a large enough $\rho$.
$$
\nabla^2_{xx}L_\rho(x^*,y^*)=\nabla^2_{xx}L(x^*,y^*)+\rho A^TA
$$
Since the Hessian is positive definite, we know that $\nabla^2_{xx}L_\rho(x^*,y^*)$ is non-singular. Along with the fact that $G(x^*,y^*)=\nabla f(x^*)+A^Ty^*=0$ by KKT, we can use the Implicit Function Theorem to define a unique function $x(y)$ (the corresponding minimizing $x$ for any given $y$) such that $\nabla_xL_\rho(x(y),y)=0$ in a neighborhood around $y^*$. The theorem gives us the derivative of the function, and since the size of the neighborhood depends on $\rho$ (as we are able to overcome non-convexity), we are able to make general statements about the problem as a whole using it.
$$
\nabla x(y)=-[\nabla^2_{xx}L_\rho(x(y),y)]^{-1}A^T
$$
Now that we have a derivative of $x(y)$, we can start showing that the update steps of the algorithm converge. To start we can define a surrogate $T(y)$ for the updates as follows.
$$
\begin{gather*}
y^{(t+1)}=y^{(t)}+\rho(Ax(y^{(t)})-b)\\
T(y)=y+\rho(Ax(y)-b)
\end{gather*}
$$
We can then use a first-order Taylor expansion of $T(y^{(t)})$ to find an approximation of the above statement, which can be simplified since $y^{(t+1)}=T(y^{(t)})$ and by definition $y^*=T(y^*)$.
$$
\begin{gather*}
T(y^{(t)})\approx T(y^*)+\nabla T(y^*)(y^{(t)}-y^*)\\
y^{(t+1)}-y^*\approx \nabla T(y^*)(y^{(t)}-y^*)
\end{gather*}
$$
This is where we can use the previous gradients found for $x(y)$ to help define the gradient $\nabla T(y^*)$. Once the previous terms are plugged in we can simplify further using the Woodbury Matrix Identity.
$$
\begin{gather*}
\nabla T(y)=\frac{\partial}{\partial y}[y+\rho(Ax(y)-b)]=I+\rho A\nabla x(y)\\
\nabla T(y^*)=I-\rho A[\nabla^2_{xx}L+\rho A^TA]^{-1}A^T=(I+\rho A(\nabla^2_{xx}L)^{-1}A^T)^{-1}
\end{gather*}
$$
As $\rho$ gets larger, the impact of the added identity matrix becomes negligible in comparison to the other terms, so we can remove it for now.
$$
\begin{gather*}
\nabla T(y^*)=(I+\rho M)^{-1}\approx(\rho M)^{-1}=\frac{1}{\rho}M^{-1}\\
M=A(\nabla^2_{xx}L)A^T
\end{gather*}
$$
This means that the norm of the gradient is bounded by $C/\rho$ for some arbitrary constant $C$, so we know that $T(y)$ forms a contraction for large enough $\rho$.
$$
\begin{gather*}
\|\nabla T(y^*)\|\leq\frac{C}{\rho}\\
y^{(t+1)}-y^*\approx\frac{C}{\rho}(y^{(t)}-y^*)
\end{gather*}
$$
This means that the error is multiplied by a factor of $1/\rho$ at each step, so the convergence to $y^*$ is proven to be linear since there exists some $\rho$ where that would be the case. Since this means that $y^{(t)}\rightarrow y^*$, and we know that $x(y)$ is continuous, we know that $x^{(k)}\rightarrow x^*$ as well, thus concluding the proof.

## Alternating Direction Method of Multipliers:
Although ALM allows us to get much more stable and faster convergence results in comparison to Dual Ascent, by using ALM we miss out on the ability to parallelize the primal update. The Alternating Direction Method of Multipliers, shortened ADMM, is an algorithm that allows for convergence under weak assumptions in the same way as ALM and also is able to achieve parallelization in the same way as Dual Ascent, getting the best of both worlds.

ADMM considers a problem over two decision variables $x$ and $z$, where we will consider a problem of the form below to simplify.
$$
\begin{gather*}
\min f(x)+g(z)\\
\text{s.t. }Ax+Bz=c
\end{gather*}
$$
The method then performs the ALM with this problem, minimizing $x$, $z$, and $y$ at each iteration, with one key distinction in how updates on $x$ and $z$ are made. Instead of updating both jointly like what ALM would do with $(x^{(t+1)},z^{(t+1)})=\text{arg}\min_{x,z}L_\rho(x,z,y^{(t)})$, ADMM uses an alternating update, first updating $x$ and then $z$.
$$
\begin{gather*}
x^{(t+1)}=\text{arg}\min_xL_\rho(x,z^{(t)},y^{(t)})\\
z^{(t+1)}=\text{arg}\min_zL_\rho(x^{(t+1)},z,y^{(t)})\\
y^{(t+1)}=y^{(t)}+\rho(Ax^{(t+1)}+Bz^{(t+1)}-c)
\end{gather*}
$$
This separation of variables actually allows for many use cases that the first two algorithms are not capable of. Parallelization can be achieved in a number of different ways, but one of the most common is for problems in the following form with a consensus variable $z$.
$$
\begin{gather*}
\min\sum^N_{i=1}f_i(x_i)\\
\text{s.t. }x_i-z=0,\text{ for all }i=1,\dots,N
\end{gather*}
$$
Under this we can separate the primal update the same way we did with Dual Ascent.
$$
\begin{gather*}
x^{(t+1)}_i=\text{arg}\min_{x_i}\left(f_i(x_i)+y^{(t)T}_i(x_i-z^{(t)})+\frac{\rho}{2}\|x_i-z^{(t)}\|^2\right)\\
z^{(t+1)}=\frac{1}{N}\sum^N_{i=1}\left(x^{(t+1)}_i+\frac{1}{\rho}y^{(t)}_i\right)
\end{gather*}
$$

### Convergence:
If $f(x)$ and $g(z)$ are convex (updates will be well-behaved) and Slater's Condition holds (for the existence of a proper solution), we know that $Ax^{(t)}+Bz^{(t)}-c\rightarrow0$, $f(x^{(t)})+g(z^{(t)})\rightarrow p^*$, and $y^{(t)}\rightarrow y^*$.

The proof follows by deriving inequalities about iterates to the optimal and to the next iteration, combining them, and then using those to derive relationships.

To derive an inequality about global relationships, we can use the optimality conditions that a given optimal $(x^*,z^*,y^*)$ would need to satisfy. Since $y$ interacts with $x$ and $z$ separately, we can separate the derivative optimality conditions, where $\partial$ is used to denote the subdifferential of the corresponding function.
$$
\begin{gather*}
Ax^*+Bz^*-c=0\\
A^Ty^*\in\partial f(x^*)\text{ and }B^Ty^*\in\partial g(z^*)
\end{gather*}
$$
Since we know that convex functions satisfy $f(x)\geq f(x^*)+\left<g,x-x^*\right>$ for all subderivatives $g$ at $x^*$, by the optimality conditions we know that the following hold as well.
$$
\begin{gather*}
f(x)-f(x^*)+\left<A^Ty^*,x-x^*\right>\geq0\\
g(z)-g(z^*)+\left<B^Ty^*,z-z^*\right>\geq0
\end{gather*}
$$
We can then combine these along with the primal feasibility condition to get the first inequality we want to derive.
$$
(f(x)+g(z))-(f(x^*)+g(z^*))+\left<y^*,Ax+Bz-c\right>
$$
To derive our second set of inequalities, we can instead use the optimality conditions of the ADMM updates themselves.
$$
\begin{gather*}
0\in\partial f(x^{(t+1)})+A^Ty^{(t)}+\rho A^T(Ax^{(t+1)}+Bz^{(t)}-c)\\
0\in\partial g(z^{(t+1)})+B^Ty^{(t)}+\rho B^T(Ax^{(t+1)}+Bz^{(t+1)}-c)
\end{gather*}
$$
We can substitute the dual update $y^{(t+1)}=y^{(t)}+\rho(Ax^{(t+1)}+Bz^{(t+1)}-c)$ to simplify both.
$$
\begin{gather*}
0\in\partial f(x^{(t+1)})+A^Ty^{(t+1)}-\rho A^TB(z^{(t+1)}-z^{(t)})\\
0\in\partial g(z^{(t+1)})+B^Ty^{(t+1)}
\end{gather*}
$$
These both give us subderivatives that we can plug into the same property of convex functions that we did for the first inequality to derive the two new inequalities.
$$
\begin{gather*}
f(x)-f(x^{(t+1)})+\left<A^Ty^{(t+1)}-\rho A^TB(z^{(t+1)}-z^{(t)}),x-x^{(t+1)}\right>\geq0\\
g(z)-g(z^{(t+1)})+\left<y^{(t+1)},B(z-z^{(t+1)})\right>\geq0
\end{gather*}
$$
Now that we have all three, we can combine the inequalities evaluated at $x=x^*$ and $y=y^*$ to derive the following bound.
$$
\left<y^{(t+1)}-y^*,r^{(t+1)}\right>+\rho\left<B(z^{(t+1)}-z^{(t)}),r^{(t+1)}+B(z^{(t+1)}-z^*)\right>\leq 0
$$
We can apply the identity $2\left<a-b,a-c\right>=\|a-b\|^2+\|a-c\|^2-\|b-c\|^2$ to both terms and simplify to get the following inequality which is simplified using the Lyapunov Function $V$, which is found after separating step $t$ and step $t+1$ terms and multiplying by $2$. It is also further simplified by using the fact that $y^{(t+1)}-y^{(t)}=\rho r^{(t+1)}$.
$$
\begin{gather*}
V_t-V_{t+1}\geq\frac{1}{\rho}\|y^{(t+1)}-y^{(t)}\|^2+\rho\|B(z^{(t+1)}-z^{(t)})\|^2\\
V_t-V_{t+1}\geq\rho\|r^{(t+1)}\|^2+\rho\|B(z^{(t+1)}-z^{(t)})\|^2\\
V_t=\frac{1}{\rho}\|y^{(t)}-y^*\|^2+\rho\|B(z^{(t)}-z^*)\|^2
\end{gather*}
$$
Since the right-hand side of the inequality is always non-negative, we know that $V_t\geq V_{t+1}$. As well, the definition of each $V_k$ means that it forms a sequence of non-negative numbers and is monotonically non-increasing, so the sequence must converge to some limit $V_\infty$. If we examine the following sum as $N\rightarrow\infty$, we see that it forms a telescoping sum which leads to the entire sum being bounded by the initial error.
$$
\sum^N_{t=0}\left(\rho\|r^{(t+1)}\|^2+\rho\|B(z^{(t+1)}-z^{(t)})\|^2\right)=\sum^N_{t=0}V_t-V_{t+1}\leq V_0-V_{N+1}\leq V_0
$$
Since this applies for $N\rightarrow\infty$, we know that the terms in the sum themselves must go to zero for the sum to be finite. This means that we need both $\|r^{(t)}\|\rightarrow 0$ and $\|B(z^{(t+1)}-z^{(t)})\|\rightarrow 0$, which proves both feasibility and optimality of $z$. Since $B(z^{(k+1)}-z^{(k)})\rightarrow 0$, we know that the update for $x$ will converge to $0\in\partial f(x^{(t+1)})+A^Ty^{(t+1)}$, which is the optimality condition for $x$, so $x\rightarrow x^*$ arises. Since $y^{(t+1)}=y^{(t)}+\rho r^{(t+1)}$ and $\|r^{(t)}\|\rightarrow 0$, we know that $y^{(t+1)}-y^{(t)}\rightarrow 0$, so $y\rightarrow y^*$ since we know that feasibility is reached. From $\|B(z^{(t+1)}-z^{(t)})\|\rightarrow 0$ we know that $z^{(t+1)}-z^{(t)}\rightarrow 0$, and since the iteration of $z$ is defined so that the optimality condition for $z$ is always satisfied with the current $y$, we know that $z\rightarrow z^*$.

Since Strong Duality holds from Slater's Condition, we know that the KKT conditions are sufficient for optimality, so proving that these points satisfy the optimality conditions we laid out originally is enough to prove that they are converging to their optimal points, which concludes the proof.

## Appendix:
This is going to include proofs for all supporting theorems except for the Implicit Function Theorem, because the proof for that theorem relies on a chain of theorems that would trail a little of course for this.

### Danskin's Theorem:
Consider the function $\phi(x)$ below defined for some continuous function $f(x,z)$, where $z\in\mathbb{R}$ and $z$ is a parameter from a compact set $Z\subset\mathbb{R}^m$ (keep in mind that $z$ being from a compact set is only so that a stable minimizer $z^*$ exists, so it can be replaced by other conditions).
$$
\phi(x)=\max_{z\in Z}f(x,z)
$$
If $f$ is convex in $x$ for every $z\in Z$ and $f$ has continuous partial derivatives with respect to $x$, then the directional derivative of $\phi(x)$ can be defined below with the set of maximizing points for any given $x$ $Z_0(x)$.
$$
\begin{gather*}
\phi^\prime(x;y)=\max_{z\in Z_0(x)}\left<\nabla_xf(x,z),y\right>\\
Z_0(x)=\{z\in Z\mid f(x,z)=\phi(x)\}
\end{gather*}
$$

The proof follows by proving that the maximum of the individual gradients is both an upper and lower bound on the directional derivative and then combining them to prove equality.

Proving the lower bound is a simple manipulation of the definition of each function. By definition of the maximum function, we know that for some $z^*\in Z_0(x)$ the maximum function forms an upper bound for the original function. We can use to form an inequality with a perturbation away from $x$ in direction $y$ and then subtract $\phi(x)=f(x,z^*)$ from both sides to get it into a more workable form.
$$
\begin{gather*}
\phi(x+ty)\geq f(x+ty,z^*)\\
\phi(x+ty)-\phi(x)\geq f(x+ty,z^*)-f(x,z^*)
\end{gather*}
$$
We can then divide both sides by $t>0$ and take the limit as $t\rightarrow0^+$ to turn both sides of the inequality into statements about the derivatives of each function.
$$
\begin{gather*}
\liminf_{t\rightarrow0^+}\frac{\phi(x+ty)-\phi(x)}{t}\geq \lim_{t\rightarrow0^+}\frac{f(x+ty,z^*)-f(x,z^*)}{t}\\
\liminf_{t\rightarrow0^+}\frac{\phi(x+ty)-\phi(x)}{t}\geq \left<\nabla_xf(x,z^*),y\right>
\end{gather*}
$$
Since this applies for all $z^*\in Z_0(x)$, we know it applies for the maximizer among them, completing the lower bound section of the proof.
$$
\liminf_{t\rightarrow0^+}\frac{\phi(x+ty)-\phi(x)}{t}\geq\max_{z\in Z_0(x)}\left<\nabla_xf(x,z),y\right>
$$
The proof for the upper bound follows a similar case of manipulation of definitions with some use of the fact that $Z$ is a closed set. By definition of the maximum function, we know that the following apply for some $t>0$ and $z_t\in Z_0(x+ty)$.
$$
\phi(x+ty)=f(x+ty,z_t)\quad \phi(x)\geq f(x,z_t)
$$
We can subtract these two terms to derive an inequality.
$$
\phi(x+ty)-\phi(x)\leq f(x+ty,z_t)-f(x,z_t)
$$
We can use the Mean Value Theorem on the right-hand side to know that there must be some $\theta_t$ that satisfies the following, which we can plug into the inequality and divide by $t$ to get into the familiar pre limit derivative form.
$$
\begin{gather*}
f(x+ty,z_t)-f(x,z_t)=t\left<\nabla_xf(x+\theta_tty,z_t),y\right>\\
\frac{\phi(x+ty)-\phi(x)}{t}\leq\left<\nabla_xf(x+\theta_tty,z_t),y\right>
\end{gather*}
$$
Since $z_t$ changes as we take the limit of $t$, we need to make sure that $z_t$ has a convergent point as $t\rightarrow0^+$. Consider the sequence of shrinking step sizes $t_k\rightarrow 0^+$, where each is guaranteed by the Extreme Value Theorem to have a corresponding maximizer $z_{t_k}\in Z$. Since $Z$ is a compact set, by the Heine-Borel Theorem we know that the set is both closed and bounded. By the Bolzano-Weierstrass Theorem, we know that $\{z_{t_k}\}$ must contain a convergent subsequence defined as $\{z_{t_{k_j}}\}$ that converges to some $z_0$, which we know satisfies $z_0\in Z$ since $Z$ is closed.

For $z_0$ to be meaningful to the bound we need to prove that $z_0\in Z_0(x)$. For any arbitrary $z\in Z$, the fact that $z_{t_{k_j}}$ is the maximizer of the step $t_k$ means the following is satisfied.
$$
f(x+t_{k_j}y,z_{t_{k_j}})\geq f(x+t_{k_j}y,z)
$$
We can then take the limit as $j\rightarrow\infty$ which leads to $f(x,z_0)\geq f(x,z)$, which along with the fact that $f$ is continuous leads to $z_0$ being the maximizer at $x$, so $z_0\in Z_0(x)$. This lets us take the limit to derive an inequality that we can use to derive the desired bound.
$$
\begin{gather*}
\frac{\phi(x+t_{k_j}y)-\phi(x)}{t_{k_j}}\leq\left<\nabla_xf(x+\theta_{k_j}t_{k_j}y,z_{t_{k_j}})\right>\\
\limsup_{j\rightarrow\infty}\frac{\phi(x+t_{k_j}y)-\phi(x)}{t_{k_j}}\leq\left<\nabla_xf(x,z_0),y\right>
\end{gather*}
$$
As this applies for every convergent subsequence, we know that the limit superior of the sequence itself must be bounded by the maximum among the individual bounds, completing the upper bound section of the proof.
$$
\limsup_{t\rightarrow0^+}\frac{\phi(x+ty)-\phi(x)}{t}\leq\max_{z\in Z_0(x)}\left<\nabla_xf(x,z),y\right>
$$
Since we know that both the $\limsup$ of the sequence is upper bounded by the same term that the $\liminf$ is lower bounded by, we know that the $\limsup$ and $\liminf$ are equal, proving that the limit exists and that it has the following value, proving the theorem.
$$
\lim_{t\rightarrow 0^+}\frac{\phi(x+ty)-\phi(x)}{t}=\phi^\prime(x;y)=\max_{z\in Z_0(x)}\left<\nabla_xf(x,z),y\right>
$$

### Finsler's Lemma:
Given a symmetric matrix $H\in\mathbb{R}^{n\times n}$ and a matrix $A\in\mathbb{R}^{m\times n}$, we know the following statements are equivalent.

1. $d^THd>0$ for all $d\in\mathbb{R}^n\backslash\{0\}$ such that $Ad=0$
2. There exists a scalar $\hat{\rho}\in\mathbb{R}$ such that for all $\rho>\hat{\rho}$, we know $H+\rho A^TA\succ0$

This allows us to know that there exists a threshold $\hat{\rho}$ such that an augmented form of $H$ is positive definite everywhere, making the utility to the convergence proof for ALM clear.

Since the theorem is one of equivalence, we need to prove both that $(1)\implies(2)$ and $(2)\implies (1)$, where the first is proven through contradiction and the second is proven through definition.

To prove $(1)\implies(2)$, we suppose that statement $(1)$ holds but statement $(2)$ is false and then reach a contradiction. If $(2)$ is false then for every $\rho_k=k$, $k=1,2,\dots$, there exists a vector $d_k\in\mathbb{R}^n$ such that the following holds. We can enforce $\|d_k\|=1$ since only the direction of the vector matters for the inequality.
$$
d^T_k(H+kA^TA)d_k=d^T_kHd_k+k\|Ad_k\|^2\leq 0
$$
Since the sequence $\{d_k\}$ is a subset of the unit sphere, which is a compact set, we can use the Bolzano-Weierstrass Theorem to know there exists a convergent subsequence $\{d_{k_j}\}$ that converges to some unit vector $d^*$. Since $\|d_k\|=1$ and $H$ is a fixed matrix, we know that $d^T_kHd_k$ is always bounded. This means as $k\rightarrow\infty$, for $d^T_kHd_k+k\|Ad_k\|^2\leq 0$ to stay true, we need $\|Ad_k\|^2\rightarrow 0$, which means that the limit below needs to hold.
$$
\lim_{k\rightarrow\infty}\|Ad_k\|^2=\|Ad^*\|^2=0\implies Ad^*=0
$$
Since $d^*$ is in the null space of $A$ and we know that the following holds, we reach a contradiction with $(1)$, thus proving that $(1)\implies(2)$.
$$
\begin{gather*}
\lim_{j\rightarrow\infty}(d^T_{k_j}Hd_{k_j}+k_j\|Ad_{k_j}\|^2)\leq 0\\
{d^*}^THd^*\leq 0
\end{gather*}
$$

To prove $(2)\implies(1)$, we assume that $(2)$ holds which means that for some $\rho>\hat{\rho}$ we know that for any non-zero vector $d$ the following holds.
$$
d^T(H+\rho A^TA)d>0
$$
We can distribute and simplify to make the inequality workable with what we need to prove $(1)$.
$$
\begin{gather*}
d^THd+d^T(\rho A^TA)d>0\\
d^THd+\rho(Ad)^T(Ad)>0\\
d^THd+\rho\|Ad\|^2>0
\end{gather*}
$$
If the chosen $d$ satisfies $Ad=0$, then this simplifies into the definition of $(1)$, thus proving that $(2)\implies(1)$.
$$
\begin{gather*}
d^THd+\rho(0)>0\\
d^THd>0
\end{gather*}
$$
Since we have proven both $(1)\implies(2)$ and $(2)\implies(1)$, we know that they are equivalent statements, thus proving the theorem.
