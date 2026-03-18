---
title: Non-Convexity and KKT
description: Non-Convex Functions and Duality Gaps, Local Strong Duality, and KKT Regularity Conditions
pubDate: 3/02/2026
---
This is a continuation of the previous KKT blog to not only explain their use in nonconvex settings but also hopefully motivate the changes that are made.

## Non-Convex Functions and Duality Gaps:
As a first step I want to motivate the idea that in some problems, especially nonconvex ones, the global concept of strong duality does not hold even though we still get some guarantees that are significant for ensuring KKT works.

One of the best things about the convex settings that were explored previously is that there is a direct connection between the stationarity of a point and it being a global minimizer. Consider a simple inequality constrained problem where we have a pair $(x^*,\lambda^*)$ that satisfy the KKT conditions, which means we know the following holds (where $f_0$ is the objective and $f_i$ are the constraints).
$$
\begin{gather*}
\nabla_xL(x^*,\lambda^*)=0\\
L(x^*,\lambda^*)=f_0(x^*)+\lambda^*f_i(x^*)=f_0(x^*)
\end{gather*}
$$
Since in a convex setting we know that both the objective and constraints are convex, we know that $L(x,\lambda)$ is also convex. This means that by stationarity we know that $x^*$ is the point chosen in the lagrange dual function $g(\lambda)$. This establishes that the duality gap is zero.
$$
\begin{gather*}
L(x^*,\lambda^*)=\inf_xL(x,\lambda^*)
g(\lambda^*)=L(x^*,\lambda^*)=f_0(x^*)
\end{gather*}
$$
Now consider the existence of the same pair $(x^*,\lambda^*)$ in a setting with a non-convex $f_0$, meaning we assume that $L(x^*,\lambda^*)=f_0(x^*)$ and the other conditions hold. Since this is now a non-convex function, consider that the function has a local feasible minimum $x^*$ and an infeasible global minimum $x_s$, meaning we know $f_0(x_s)<f_0(x^*)$. 

If this gap is large enough, we can run into a situation where $L(x_s,\lambda^*)<L(x^*,\lambda^*)$ even if the constraint penalty term is being added to the Lagrangian at $x_s$ (if you need to be convinced this is possible, look at the example $x^3-3x$ subject to $x\geq 2$). This means that the lagrange dual function will actually lead to a different value then it did previous.
$$
\begin{gather*}
g(\lambda^*)=\inf_xL(x,\lambda^*)=L(x_s,\lambda^*)
\end{gather*}
$$
This means that even though we have a set of points tha satisfy the KKT conditions and that also lead to the correct primal optimum, solving the dual problem actually leads to a duality gap.
$$
\begin{gather*}
d^*=g(\lambda^*)=L(x_s,\lambda^*)\quad p^*=f_0(x^*)\\
d^*<p^*
\end{gather*}
$$
This naturally leads to the idea of reformatting the KKT conditions to act as a tool to prove that a point has significance, rather than a tool that can be used as a test to find the points themselves.

## Local Strong Duality:
If we have an optimization method that uses dual variables in a smarter way than the original dual problem, especially ones designed to be well suited for non-convex settings, we can still use these conditions to validate that a set of primal and dual variables is significant. This is where the concept of Local Strong Duality comes in to play, where a problem will satisfy strong duality-like properties locally around a given $(x^*,\lambda^*)$. We can define a point $x^*$ as satisfying Local Strong Duality if there exists dual variables $\lambda^*,\nu^*$ such that the following are satisfied.
$$
\begin{gather*}
\nabla_xL(x^*,\lambda^*,\mu^*)=0\\
\lambda^*_if_i(x^*)=0\\
L(x^*,\lambda^*,\nu^*)=\inf_{x\in B(x^*,\epsilon)}L(x,\lambda^*,\nu^*)
\end{gather*}
$$
The first and second ensure that the dual variables are chosen accurately and that the dual gap is actually $0$, which could also be seen as they simply restate Stationarity and Complementary Slackness from the KKT conditions. The third acts as a check to make sure that the stationarity point we found is actually a minima since that is no longer guaranteed in a non-convex setting.

Along with the primal and dual feasibility conditions of the original KKT conditions, if these hold we know that the tuple $(x^*,\lambda^*,\nu^*)$ gives meaningful information about the primal. In non-convex settings with well-behaved constraints the KKT conditions are necessary conditions for a minima, but they are no longer sufficient. Using Local Strong Duality turns it into a sufficient condition for optimality.

## KKT Regularity Conditions:
to be continued later...