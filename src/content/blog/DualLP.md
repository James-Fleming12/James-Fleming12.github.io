---
title: Dual Linear Programs
description: Quick Summary for some Lab Work
pubDate: 2/15/2026
---

This is just a quick summary of some proofs and derivations that I needed to do for some lab work review, really really simple stuff. This is more of a place for me to make sure that I have absolutely no holes in my understanding and less of a resource for others.

## Canonical Form Linear Progarm:
We can describe a general form of a linear optimization problem as follows, in what is called the "Canonical Form". The decision variable $x\in\mathbb{R}^n$ forms an objective with a cost vector $c\in\mathbb{R}^n$ and is constrained by a set of constraint variables $A\in\mathbb{R}^{m\times n}$ and $b\in\mathbb{R}^m$, where $\text{s.t.}$ stands for $\text{subject to}$. The constraint $x\geq 0$ is standard for removing impossible solutions in physical problems and also for deriving the things later.
$$
\min_x\space\space c^Tx\quad\text{s.t.}\quad Ax\geq b,x\geq 0
$$

## Lagrangian Form:
Let's say we wanted to convey this optimization without explicit constraints and instead handle them within the function itself. We can do this with the Lagrangian $L(x,y)$ as follows where $y\in\mathbb{R}^m$. In general terms, $f(x)$ is the objective of the problem and $h(x)$ describes each functional constraint (in this case ignoring $x\geq 0$) such that $g_i(x)\leq 0$ satisfies the respective constraint.
$$
\begin{gather*}
L(x,y)=f(x)+y^Th(x)=c^Tx+y^T(b-Ax)\\
\min_{x\geq 0}\max_{y\geq 0}c^Tx+y^T(b-Ax)
\end{gather*}
$$
Since we have $y\geq 0$, for the maximum over $y$ to not collapse to infinity, we need $b-Ax\leq 0$, which naturally gives rise to the constraint $Ax\geq b$. Since the maximum of $y$ would be $0$ when multiplied with a negative term, whenever the constraint is satisfied this leaves the original optimization problem. This idea also naturally gives rise to the necesitty of having $y\geq 0$, since without it we would have degenerate maximums both when the constraint is satisfied and when it is not.

## Lagrangian Dual Form:
If we flip the $\min$ and $\max$ in the problem, we can derive the dual form of the problem. Duality has a very malleable concept in mathematics, being used to define a way that two structures can be paired with some level of intrinsic symmetry. In our case, this is done by turning our minimization problem, our original problem known as the primal, into a maximization problem, known as the dual.

The function $g(y)$ below is often known as the Lagrange Dual Function and will have a couple of notable properties proven later.
$$
\begin{gather*}
g(y)=\min_xL(x,y)=\min_{x\geq0}[c^Tx+y^T(b-Ax)]\\
g(y)=y^Tb+\min_{x\geq0}[c^Tx-y^TAx]=y^Tb+\min_{x\geq0}[(c-A^Ty)^Tx]
\end{gather*}
$$
We can use the same reasoning of avoiding degenerate minimums for this minimization to prove that we need $c-A^Ty\geq 0$ since $x\geq 0$. The same reasoning that if $c-A^Ty\leq 0$ and $x\geq 0$ we can know that the minimum is when $x=0$ can also be used as previous to derive a constraint based optimization problem for the dual problem, along with the fact that we know the transpose of a scalar is equal to the scalar.
$$
\begin{gather*}
\max_{y\geq 0}\space\space b^Ty\quad\text{s.t.}\quad A^Ty\geq c
\end{gather*}
$$
This problem may not seem significant at all, but we can prove that the values of the primal and dual are linked.

## Weak Duality:
Weak Duality guarantees that the solution to any dual problem will always act as a lower bound for the solution to the primal problem. In the case of the simple linear programs here the benefit of optimizing over the dual form instead of the primal may seem restricted to efficiency, but in harder optimization schemes the dual often has a much better optimization landscape. For our cases with Linear Programs, it is often used when a problem has many more decision variables than it does constraints, since the dual has decision variables in line with the number of constraints instead.

For our basic linear program, we can trivially prove weak duality with the bounds of each problem. Since we know that $Ax\geq b$ and $y\geq 0$, we also know that $b^Ty\leq (Ax)^Ty$, which we can combine with the fact that $A^Ty\geq c$ and $x\geq 0$ to prove the following.
$$
b^Ty\leq (Ax)^Ty=x^TA^Ty\leq x^Tc=c^Tx
$$
Since we have $b^Ty\leq c^Tx$, we know that $\max b^Ty\leq \min c^Tx$, proving weak duality for this problem.

We can also prove this for the general Lagrangian case by proving that $g(y)$ is always less than $f(x)$. This can be proven below using the solution of the primal $x^*$, which we know satisfies $h(x^*)\leq 0$ since it has to satisfy the constraints.
$$
\begin{gather*}
L(x^*,y)=f(x^*)+yg(x^*)
\end{gather*}
$$
Since $x^*$ has to satisfy all of the constraints, we know that $h(x^*)\leq 0$, which combined with $y\geq 0$ can remove $yg(x^*)$ from the Lagrangian with the same reasoning as before. By definition of the minimum, we know that the following is satisfied, thus proving weak duality for all dual problems.
$$
\begin{gather*}
g(y)=\min_{x\geq 0}L(x,y)\leq L(x^*,y)=f(x^*)
\end{gather*}
$$

## Strong Duality:
Strong Duality is a condition of an optimization problem that guarantees that value of the solution to the dual problem will give exactly the value of the primal's solution. This does not hold for all primal dual pairs, but we can prove that it holds for our specific case of linear programs, which means that there exists some $z=c^Tx^*$ and $z=b^Ty^*$.

This can be proven by setting up a linear system that would find such a $y^*$, assuming that no such $y^*$ exists, and then finding a contradiction using Farkas' Lemma, specifically the Inequality Form, which states that exactly one of the following is true for a matrix $A$ and a vector $b$.

1. There exists $x\geq 0$ such that $Ax\leq b$
2. There exists $y\geq 0$ such that $A^Ty\geq 0$ and $b^Ty<0$

Since we want a $y^*$ that satisfies the constraints, we know that $y^*$ needs to satisfy both $A^Ty^*\leq c$ to be feasible and $b^Ty^*\geq z$. This lets us set up a system of inequalities as follows.
$$
\begin{bmatrix}A^T\\-b^T\end{bmatrix}y\leq\begin{bmatrix}c\\-z\end{bmatrix}
$$

If we assume that this system has no solutions, then we know that there has to be some vector $[x;\lambda]\geq 0$ such that $Ax-\lambda b\geq 0$ and $c^Tx-\lambda z<0$, which can be rewritten as $Ax\geq\lambda b$ and $c^Tx<\lambda z$. We can break this up into cases based on the value of $\lambda$ to show that this is not possible.

In the case where $\lambda>0$, we can show a contradiction by dividing by $\lambda$, giving $A(x/\lambda)=b$ and $c^T(x/\lambda)<z$. The first inequality shows that $(x/\lambda)$ is feasible for the primal, which combined with the second would show that there is a feasible $x$ that leads to a minimum strictly less than $z$, which leads to a contradiction because we defined $z$ as the minimum.

In the case where $\lambda=0$, this implies that $Ax\geq 0$ and $c^Tx<0$. We can add this point to the optimal $x^*$ to get $A(x+x^*)=Ax+Ax^*\geq b+0$ and $c^T(x+x^*)=c^Tx+c^Tx^*=(\text{some value }<0) + c^Tx^*$, which means that $(x+x^*)$ would be a feasible solution leading to a lower value than $z$, which again leads to a contradiction.

Since this leads to an impossible situation, we know that there must exist some $y$ that satisfies both inequalities. Since this $y$ satisfies $b^Ty^*\geq z$, and we have already proven $b^Ty\leq z$ from weak duality, we know that $b^Ty=z$, thus proving strong duality for this linear program pair.