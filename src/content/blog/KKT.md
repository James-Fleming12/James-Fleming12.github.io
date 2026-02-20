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
The KKT Conditions act as an analog to this method for constrained problems that also have inequality constraints. Consider an optimization problem of the following form, with an added set of inequality constraint functions $h_j(x)$.
$$
\min_x f(x)\quad\text{s.t.}\quad g_i(x)=0,h_j(x)\leq 0
$$
Since this adds inequality constraints to the previous problem, we can treat the equality constraints basically the same. The inequality constraints allow a feasible point to either be on the surface of the inequalities or inside, meaning we can break down the derivation into whether a point and constraint are active, meaning $h_j(x)=0$, or inactive, meaning $h_j(x)<0$.

If the constraint is active, then the inequality constraint restricts $dx$ like an equality constraint, except we are allowed to move inwards as well. This leads to the same $\nabla f+\mu\nabla h=0$ condition as before, which can also be extended to a span in the same way. The only change is that we need $\mu_j\geq0$. Since the constraint still allows stepping into the feasible set, we need both gradients to point in opposite directions, so $\nabla f=-\mu\nabla h$.

If the constraint is inactive, then the constraint shouldn't influence the set of possible $dx$ since we can move in any direction locally. The only way to make $\nabla f+\mu\nabla h=0$ true while allowing every $dx$ is to force $\mu=0$, which turns it into the standard $\nabla f=0$. Since we either want to have a constraint be active, meaning $h_j(x)=0$, or we want $u_j=0$, we can derive the property that we want $\mu_jh_j(x^*)=0$.

Combining the properties of these two and the added feasibility requirements, we get the following conditions for critical points of the problem.
$$
\begin{gather*}
\nabla f(x^*)+\sum\lambda^*_i\nabla g_i(x^*)+\sum\mu^*_j\nabla h_j(x^*)=0\\
g_i(x^*)=0\text{ and }h_j(x^*)\leq 0\\
\mu^*_j\geq 0\\
\mu^*_jh_j(x^*)=0
\end{gather*}
$$