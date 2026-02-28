---
title: Lagrange Multipliers and KKT Conditions
description: The derivation of the method of lagrange multipliers and extending them to the KKT conditions
pubDate: 2/20/2026
---

This is another quick summary of the derivations for the KKT conditions for some lab review.

## Method of Lagrange Multipliers:
Consider a constrained optimization problem with the following form. The function $f(x)$ defines the objective and each $g_i(x)$ defines one of the $m$ equality constraint.
$$
\min_x f(x)\quad\text{s.t.}\quad g_i(x)=0\space i=1,\dots,m
$$
Due to the constraints, if we want to find the critical points of the problem we can't do the typical $\nabla f=0$ check. Instead we need to use an analog to the check that acts on the surface defined by the equality constraints.

Consider the case where $m=1$, so we only have one equality constraint. Any small movement $dx$ from a point $x$ on the constraint surface must not make a change to the value of $g(x)$. This can also be thought as the gradient at $x$ being $0$, since we want to define each direction that maintains $g(x)=0$.
$$
\nabla g(x)\cdot dx=0
$$
If $x^*$ is a critical point of $f(x)$ subject to $g(x)=0$, then it can not change along any of these directions $dx$, else you could move in some direction along the surface to find a smaller value of $f$.
$$
\nabla f(x^*)\cdot dx=0
$$
Since both $\nabla g(x)$ and $\nabla f(x)$ need to be perpendicular to any valid $dx$, they must be parallel to eachother. This means that one must be a scalar multiple of the other. We can define this multiple with $\lambda$.
$$
\nabla f(x^*)=\lambda\nabla g(x^*)\rightarrow\nabla f(x^*)-\lambda\nabla g(x^*)=0
$$
Since $\lambda$ is an unknown multiple, we can simply reformat this condition into $\nabla f(x^*)+\lambda\nabla g(x^*)=0$. This means that any minimum $x^*$ must satisfy this, so any feasible $x$ that satisfies this is a candidate for being the minimum.

We can form a function, the Lagrangian $L(x,\lambda)$, using this definition. Making the scalar multiple act on $g(x)$ allows us to check feasibility within one gradient calculation, thus deriving the method of finding candidates through $\nabla L(x,\lambda)=0$.
$$
\begin{gather*}
L(x,\lambda)=f(x)+\lambda g(x)\\
\nabla L(x,\lambda)=\left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial\lambda}\right)=(\nabla f(x)+\lambda\nabla g(x),g(x))=0
\end{gather*}
$$
We can simply extend this to multiple constraints by redefining the set of possible directions over the surface $dx$. Consider a problem with two constraints $g_1(x)=0$ and $g_2(x)=0$. Any valid movement $dx$ from a feasible $x$ must then satisfy $\nabla g_1\cdot dx=0$ and $\nabla g_2\cdot dx=0$. This means that $dx$ must be perpendicular to the subspace spanned by $\{\nabla g_1,\nabla g_2\}$.
$$
(a\nabla g_1+b\nabla g_2)\cdot dx=a(\nabla g_1\cdot dx)+b(\nabla g_2\cdot dx)=a(0)+b(0)=0
$$
Since we still have the fact that $\nabla f$ must also be perpendicular to every $dx$, we know that $\nabla f$ must also be in this span.
$$
\nabla f=\lambda_1\nabla g_1+\lambda_2\nabla g_2
$$
We can then use this to form a generalized Lagrangian and condition for a problem with $m$ equality constraints.
$$
\begin{gather*}
L(x,\lambda)=f(x)+\sum^m_{i=1}\lambda_ig_i(x)\\
\nabla L(x,\lambda)=(\nabla f(x)+\sum^m_{i=1}\lambda_i\nabla g_i(x),g_1(x),\dots,g_m(x))=0
\end{gather*}
$$

## KKT Conditions:
The KKT Conditions act as an analog to this method for constrained problems that also have inequality constraints as long as the inequalities are well-behaved. Consider an optimization problem of the following form, with an added set of inequality constraint functions $h_j(x)$.
$$
\min_x f(x)\quad\text{s.t.}\quad g_i(x)=0,h_j(x)\leq 0
$$
Since this adds inequality constraints to the previous problem, we can treat the equality constraints basically the same. The inequality constraints allow a feasible point to either be on the surface of the inequalities or inside, meaning we can break down the derivation into whether a point and constraint are active, meaning $h_j(x)=0$, or inactive, meaning $h_j(x)<0$.

If the constraint is active, then the inequality constraint restricts $dx$ like an equality constraint, except we are allowed to move inwards as well. This leads to the same $\nabla f+\mu\nabla h=0$ condition as before (under certain conditions on the constraint functions themselves which stop things like cusps from forming), which can also be extended to a span in the same way. The only change is that we need $\mu_j\geq0$. Since the constraint still allows stepping into the feasible set, we need both gradients to point in opposite directions, so $\nabla f=-\mu\nabla h$.

If the constraint is inactive, then the constraint shouldn't influence the set of possible $dx$ since we can move in any direction locally. The only way to make $\nabla f+\mu\nabla h=0$ true while allowing every $dx$ is to force $\mu=0$, which turns it into the standard $\nabla f=0$. Since we either want to have a constraint be active, meaning $h_j(x)=0$, or we want $u_j=0$, we can derive the property that we want $\mu_jh_j(x^*)=0$, called Complementary Slackness. While the idea behind it is always true, the simplified version that acts in the equation needs certain conditions to stay correct since certain problem formulations can cause multipliers of nonactive constraints to still need to be nonzero.

Combining the properties of these two and the added feasibility requirements, we get the following conditions for critical points of the problem.
$$
\begin{gather*}
\nabla f(x^*)+\sum\lambda^*_i\nabla g_i(x^*)+\sum\mu^*_j\nabla h_j(x^*)=0\\
g_i(x^*)=0\text{ and }h_j(x^*)\leq 0\\
\mu^*_j\geq 0\\
\mu^*_jh_j(x^*)=0
\end{gather*}
$$

## Complementary Slackness:
One of the requirements for using KKT conditions is in how well-behaved the inequality constraint lagrange multipliers are, which allow the Complementary Slackness property $\mu^*_jh_j(x^*)=0$ to hold. This comes in the form of whether the primal and dual have strong duality (technically only requiring local strong duality, which will be important for understanding the purpose of later conditions). Consider a problem where strong duality does not hold. This means that we know there exists some dual gap $\epsilon$ between the minimum of the primal and dual problems.
$$
f(x^*)=F(\lambda^*,\mu^*)+\epsilon
$$
By definition of the dual function $F$ we can simplify this to bound the values of the constraint variables. Since we know that any primal optimum $x^*$ will lead to $g_i(x^*)=0$, we can isolate the sum of all $\mu_jh_j$ and show that it can be nonzero, disproving Complementary Slackness for the minima.
$$
\begin{gather*}
F(\lambda^*,\mu^*)=\inf_xL(x,\lambda^*,\mu^*)\leq L(x^*,\lambda^*,\mu^*)\\
f(x^*)-\epsilon\leq f(x^*)+\sum\lambda^*_ig_i(x^*)+\sum\mu^*_jh_j(x^*)\\
-\epsilon\leq\sum\mu^*_jh_j(x^*)
\end{gather*}
$$
Now consider a problem where strong duality does hold, i.e. one where $f(x^*)=F(\lambda^*,\mu^*)$. We can follow the same chain of logic to prove that $0\leq\sum^m_{i=1}\mu^*_jh_j(x^*)$, and since we know that $\mu^*_i\geq 0$ and $h_j(x^*)\leq 0$, the only way to make this inequality hold is for each element of the sum to be equal to $0$, thus proving Complementary Slackness for our minima.

## Regularity Conditions:
Along with the strong duality condition for Complementary Slackness, we also need a regularity condition to guarantee that the feasible set defined by the inequality constraints is well-behaved so that our stationarity condition is logical. There are a number of regularity conditions for this task, but one of the most common is Slater's Condition, which states that $f$ is convex, all $g_i$ are convex, all $h_j$ are affine, and there exists some $\tilde{x}$ such that $g_i(\tilde{x})<0$ and $h_j(\tilde{x})=0$. These conditions guarantee that the feasible set is well behaved (the first three guaranteeing that the feasible set is convex and the fourth ensuring the feasible set has an interior), and it also guarantees strong duality so it often is also used to prove complementary slackness in cases where it applies. Since only local strong duality is needed for complementary slackness, other regularity conditions exist for other problems that are not as simple.

To prove that this leads to the existence of dual variables that satisfy stationarity, we first construct a set $\mathcal{A}$ of all attainable objective and constraint values. Since the $f$ and $g_i$ are convex and $h_j$ are affine (can not make the set nonconvex), we know $\mathcal{A}$ is convex.
$$
\mathcal{A}=\{(u,v,w)|\exists x,f(x)\leq u,g_i(x)=v_i,h_j(x)\leq w_j\}
$$
Given a primal optimal point and an objective value $p^*=f(x^*)$, we can define a point $(p^*,0,0)$ that is on the boundary of $\mathcal{A}$. Since the point is on the boundary of the objective values we know that it is on the boundary of the set itself. Since $x^*$ is feasible we know that there exists some point in the set $(p^*,0,\text{some value }\leq 0)$ and since we defined $\mathcal{A}$ as all values above any $(f(x),g_i(x),h_j(x))$, we know that $(p^*,0,0)$ exists in the set.

Since $\mathcal{A}$ is convex, by the Supporting Hyperplane Theorem we know that there exists some non-zero vector $(\alpha,\lambda,\mu)$  such that for all $(u,v,w)\in\mathcal{A}$ the following applies.
$$
\begin{gather*}
\alpha u+\sum\lambda_iv_i+\sum\mu_jw_j\geq\alpha p^*+\sum\lambda_i(0)+\sum\mu_j(0)\\
\alpha f(x)+\sum\lambda_ig_i(x)+\sum\mu_jh_j(x)\geq\alpha f(x^*)
\end{gather*}
$$
In order to work with this statement further, we need $\alpha>0$ so that we know it makes some statements about $f(x)$. We can do this through contradiction by supposing that $\alpha=0$. The inequality then becomes $\sum\lambda_ig_i(x)+\sum\mu_jh_j(x)\geq 0$ for all $x$. We can then apply the point $\tilde{x}$ from the condition statement, which by definition has $g_i(\tilde{x})=0$ and $h_j(\tilde{x})<0$. This reduces the sum to $\sum\mu_jh_j(\tilde{x})\geq 0$ and since we know the sum is negative, this leads to a contradiction, thus $\alpha>0$.

Since we know $\alpha$ is non-zero, we can divide by $\alpha$ and see that the left hand side becomes the lagrangian with $\lambda^*_i=\lambda_i/\alpha$ and $\mu^*_j=\mu_j/\alpha$.
$$
\begin{gather*}
f(x)+\sum\lambda^*_ig_i(x)+\sum\mu^*_jh_j(x)\geq f(x^*)\\
L(x,\lambda^*,\mu^*)\geq f(x^*)
\end{gather*}
$$
Since we know that Slater's Condition guarantees strong duality, and thus Complementary Slackness, we know that $f(x^*)=L(x^*,\lambda^*,\mu^*)$, and since we know this inequality holds for all $x$, we know that $L(x^*,\lambda^*,\mu^*)$ is the minimum of the lagrangian. By the first order necessary condition of optimality to hold, we need the following, which proves stationarity and the existence of the dual variables in $(\lambda^*,\mu^*)$.
$$
\begin{gather*}
\nabla_xL(x^*,\lambda^*,\mu^*)=0\\
\nabla f(x^*)+\sum^m_{i=1}\lambda_i^*\nabla g_i(x^*)+\sum^p_{j=1}\mu^*_j\nabla h_j(x^*)=0
\end{gather*}
$$

## Sufficient Condition:
This was all to set up the fact that under certain regularities we know that any optimum will $(x^*,\lambda^*,\mu^*)$ need to satisfy the KKT conditions, but we can also prove that under these same conditions ($f$ and $g_i$ being convex and $h_j$ being affine) we know that any tuple satisfying the conditions has to be the optimum.

Consider a tuple $(x^*,\lambda^*,\mu^*)$ that satisfy the KKT conditions. We have that $x^*$ is primal feasible and $\lambda^*,\mu^*$ are dual feasible. Using these conditions we already proved that stationarity holds, so we can derive the following.
$$
F(\lambda^*,\mu^*)=L(x^*,\lambda^*,\mu^*)=f(x^*)
$$
Combining this with weak duality, which states that $f(x^*\geq F(\lambda^*,\mu^*))$, we see that $\lambda^*,\mu^*$ maximized $F$ and that strong duality holds. Since $\lambda^*,\mu^*$ maximized $F$, the pair is the solution to the dual problem. Since strong duality holds, we can then deduce that $x^*$ is also our optimum, thus showing that any points satisfying the KKT conditions under this setup would be optimum.
