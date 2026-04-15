---
title: Derivation of Quasi-Newton Methods
description: The definition of Newton and Quasi-Newon and the derivation of Broyden's, SR1, DFP, and BFGS
pubDate: 4/01/2026
---

This is a quick review that I needed to do before some composite optimization review. It's going to cover the basic definitions of Newton's Method, Quasi-Newton Methods, and give derivations for Broyden's Method, SR1, PSB, DFP, and BFGS. Might cover the convergence of each in a later article, but this was long enough as is.

## Newton's Method:
One interpretation of Gradient Descent that is going to important for the content here is the interpretation that the method approximates $\nabla^2 f(\xi)$ of the second-order taylor expansion (where $\xi$ is a point between $x$ and $x^{(t)}$ which makes the approximation exact) below using $\frac{1}{\mu}I$ (where $\mu$ is the stepsize).
$$
\begin{gather*}
f(z)=f(x^{(t)})+\nabla f(x^{(t)})(z-x^{(t)})+\frac{1}{2}(z-x^{(t)})\nabla^2f(\xi)(z-x^{(t)})\\
f(z)\approx f(x^{(t)})+\nabla f(x^{(t)})(z-x^{(t)})+\frac{1}{2\mu}\|z-x^{(t)}\|^2
\end{gather*}
$$
We can see that this leads to the right-hand side of the approximation being a convex function, which if we define as $g(z)$ and use the fact that the minimum will be at $\nabla g(z^*)=0$, then we can derive the standard gradient descent algorithm.
$$
\begin{gather*}
\nabla g(z)=\nabla f(x^{(t)})+\frac{1}{\mu}(z-x^{(t)})=0\\
z^*=x^{(t)}-\mu\nabla f(x^{(t)})
\end{gather*}
$$
Newton's Method arises when we approximate $\nabla^2f(\xi)$ with the real hessian of the function at $x^{(t)}$ rather than using $\frac{1}{\mu}I$.
$$
f(x)\approx f(x^{(t)})+\nabla f(x^{(t)})(x-x^{(t)})+\frac{1}{2}(x-x^{(t)})^T\nabla^2f(x^{(t)})(x-x^{(t)})
$$
Since this also gives rise to a convex function approximation, we can use the same trick of finding when the gradient is $0$ and using it as our update rule.
$$
\begin{gather*}
0=\nabla f(x^{(t)})+\nabla^2f(x^{(t)})(x-x^{(t)})\\
x^{(t+1)}=x^{(t)}-[\nabla^2f(x^{(t)})]^{-1}\nabla f(x^{(t)})
\end{gather*}
$$
Since this approximation of the function captures much more information about the curvature of the function it is able to converge extraordinarily fast when it can, although it is more limited to the types of functions it can converge in comparison to gradient descent. It is also much more computationally expensive in comparison to other methods, since not only is calculating the hessian often computationally expensive when parameter counts get large, but inverting it can take even longer. This leads into the idea of approximating the hessian rather than using the exact hessian, which leads into the next set of methods.

## Quasi-Newton Methods:
If we replace $\nabla^2f(x^{(t)})$ with a close approximation $B^{(t)}$, we create the family of Quasi-Newton methods.
$$
x^{(t+1)}=x^{(t)}-[B^{(t)}]^{-1}\nabla f(x^{(t)})
$$
For methods in this family we want $B^{(t)}$ and $\nabla^2 f(x)$ to have very similar properties, one of which will which acts as the backbone for all of the methods here will be reviewed in the next subsection, but for a quick proof we can look at a property that won't reappear until much later.

If the approximation $B$ is positive definite, then we know that the direction $-[B^{(t)}]^{-1}\nabla f(x^{(t)})$ will be a descent direction. By definition we know that for a differentiable function $f$, a direction $p$ is only a descent direction if $\nabla f(x)^Tp<0$. Given that we define $p=-B^{-1}\nabla f(x)$, we can plug our definition into the descent direction check.
$$
\nabla f(x)^Tp=\nabla f(x)^T\left(-B^{-1}\nabla f(x)\right)=-\nabla f(x)^TB^{-1}\nabla f(x)
$$
If $B$ is positive definite then we know that its inverse $B^{-1}$ is also positive definite, and by definition we know that for any vector $v$ we have $v^TB^{-1}v>0$, so we know that the following holds, proving that it leads to a descent direction.
$$
-\nabla f(x)^TB^{-1}\nabla f(x)<0
$$
This is not necessarily true if $B$ is not positive definite, so it's a property that we would like to have in an ideal scenario, but is not a requirement for a functioning algorithm.

### Secant Equation:
To properly approximate the hessian from only gradient and iterate information, we can use what is known as the secant equation. Since we know that the hessian is the gradient of the gradient, we know that the hessian describes the change in the gradient over the change in $x$, so we know that the following is a good approximation of what the real hessian would be if the underlying function is not too hectic.
$$
\begin{gather*}
B^{(t+1)}s^{(t)}=y^{(t)}\\
s^{(t)}=x^{(t+1)}-x^{(t)}\quad y^{(t)}=\nabla f^{(t+1)}-\nabla f^{(t)}
\end{gather*}
$$
The only reason I mention this early is because I mention it by name a lot both in the definitions and derivations, and the fact that it is used in the very first method we discuss.

## Broyden's Method:
Since the system defined by the secant equation has $n^2$ variables and only $n$ equations, it has infinitely many solutions. This means we have to define some way to which is the best. One way is to limit the change between the approximations in neighboring steps, which also leads to much more stable iterations since neighboring steps with very different approximations often become unstable. Broyden's Method frames this as an optimization problem, which is defined below.
$$
\min_B\|B-B^{(t)}\|^2_F\quad\text{s.t.}\quad Bs^{(t)}-y^{(t)}
$$
This leads to a rank-1 update where the error of the secant equation is projected onto the direction $s^{(t)}$, which gives the update rule for the hessian approximation for the method.
$$
B^{(t+1)}=B^{(t)}+\frac{(y^{(t)}-B^{(t)}s^{(t)})(s^{(t)})^T}{(s^{(t)})^Ts^{(t)}}
$$
Even though this gives an approximation for the hessian, which would leave us with the task of inverting it still, we can use the Woodbury Matrix Identity to turn the rank-1 update into an update for the inverse hessian directly.
$$
(A+UCV)^{-1}=A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}
$$

### Derivation:
Since we are solving for a minimization, we can use the Lagrangian $L(B,\lambda)$ and then use optimality conditions to find the form of the minimum. We can first define the Lagrangian with the objective scaled by $\frac{1}{2}$ to simplify the notation for the final update as scaling it by a constant term won't change the final minimizer.
$$
L(B,\lambda)=\frac{1}{2}\sum_{i,j}(B_{ij}-B^{(t)}_{ij})+\lambda^T(Bs^{(t)}-y^{(t)})
$$
We can then take the derivative and set it equal to $0$ to find the optimal $B$ in terms of our unknowns, which leads to the form of a rank-1 update with the outer product of $\lambda$ and $s^{(t)}$.
$$
\begin{gather*}
\frac{\partial L}{\partial B_{ij}}=(B_{ij}-B^{(t)}_{ij})+\lambda_is^{(t)}_j=0\\
B-B^{(t)}+\lambda(s^{(t)})^T=0\rightarrow B=B^{(t)}-\lambda(s^{(t)})^T
\end{gather*}
$$
We can then solve for $\lambda$ by substituting the secant equation back into the equation and isolating $\lambda$.
$$
\begin{gather*}
(B^{(t)}-\lambda(s^{(t)})^T)s^{(t)}=y^{(t)}\\
B^{(t)}s^{(t)}-\lambda((s^{(t)})^Ts^{(t)})=y^{(t)}\\
-\lambda((s^{(t)})^Ts^{(t)})=y^{(t)}-B^{(t)}s^{(t)}\\
\lambda=-\frac{y^{(t)}-B^{(t)}s^{(t)}}{(s^{(t)})^Ts^{(t)}}
\end{gather*}
$$
Plugging this $\lambda$ back into the derived rank-1 update gives the update rule of Broyden's Method, completing the derivation.

## SR1:
One big problem with Broyden's Method is that it ignores that all hessian matrices by definition are symmetric. SR1, standing for Symmetric Rank-1, fixes this by adding a symmetry constraint to the secant equation, along with an additional constraint to the update requiring it to be rank-1. This was done at the time not only for efficiency due to limited compute, but also because it led to an update derived entirely from the residual $y-Bs$. Updates of higher ranks have much more wiggle room to perform updates that find significance from other things, which was not deemed ideal. This leads to a set of three constraints defined below.
$$
\quad B=B^T,\space Bs^{(t)}=y^{(t)},\space\text{rank}(B-B^{(t)})=1
$$
This leads to a unique solution that satisfies all three, which when solved for gives the update rule for SR1.
$$
B^{(t+1)}=B^{(t)}+\frac{(y^{(t)}-B^{(t)}s^{(t)})(y^{(t)}-B^{(t)}s^{(t)})^T}{(y^{(t)}-B^{(t)}s^{(t)})^Ts^{(t)}}
$$
Removing the attempt to make sure that neighboring hessians are as similar as possible actually allows SR1 to converge remarkably fast in scenarios where making rapid changes to the hessian is preferred, but it comes with heaps of stability issues.

### Derivation:
Since we need the update to be rank-1 and lead to a symmetric matrix, we know that the update itself must be symmetric, meaning we can describe the update with only one vector $v$ and a scalar $\omega$.
$$
B^{(t+1)}=B^{(t)}+\sigma vv^T
$$
We can then apply the secant equation to include the last necessary constraint into derivation and simplify.
$$
\begin{gather*}
(B^{(t)}+\sigma vv^T)s^{(t)}=y^{(t)}\\
B^{(t)}s^{(t)}+\sigma v(v^Ts^{(t)})=y^{(t)}\\
\sigma v(v^Ts^{(t)})=y^{(t)}-B^{(t)}s^{(t)}
\end{gather*}
$$
Since $v^Ts^{(t)}$ is a scalar, and thus $\sigma v^Ts^{(t)}$ is also a scalar, so we know that $v$ must have the same direction as the residual. To keep the derivation clean, we can set this to become an equivalence rather than scaling it since we can scale it in the next step.
$$
v=y^{(t)}-B^{(t)}s^{(t)}
$$ 
We can substitute this back into $\sigma v(v^Ts^{(t)})=y^{(t)}-B^{(t)}s^{(t)}$ to find the value for $\sigma$.
$$
\begin{gather*}
\sigma(v^Ts^{(t)})v=v\\
\sigma(v^Ts^{(t)})=1\rightarrow \sigma=\frac{1}{v^Ts^{(t)}}
\end{gather*}
$$
Plugging these values for $v$ and $\sigma$ into the original rank-1 update gives the update rule for SR1, completing the derivation.

## PSB:
Although the reliance on a rank-1 update is sound in theory and in practice, the update rule derived from it was not. If there was ever a situation where $(y^{(t)}-B^{(t)}s^{(t)})^Ts^{(t)}\approx 0$ (which often happened when the two vectors were nearly orthogonal), the update term would explode leading to a lot of numerical instability. This is where PSB, standing for Powell-Symmetric-Broyden, reintroduced the use of the minimization problem to define a unique $B$ without the rank-1 constraint.
$$
\min_B\|B-B^{(t)}\|^2_F\quad\text{s.t.}\quad B=B^T,Bs^{(t)}=y^{(t)}
$$
Solving the minimization actually gives a rank-2 update for PSB, defined below where the residual $r^{(t)}=y^{(t)}-B^{(t)}s^{(t)}$ is used.
$$
B^{(t+1)}=B^{(t)}+\frac{r^{(t)}(s^{(t)})^T+s^{(t)}(r^{(t)})^T}{(s^{(t)})^Ts^{(t)}}-\frac{(r^{(t)})^Ts^{(t)}s^{(t)}(s^{(t)})^T}{((s^{(t)})^Ts^{(t)})^2}
$$
Even though this leads to an update in a form that is not naturally compatible with our previous Woodbury Identity trick for getting the update of the inverse hessian, we can simply treat the update as a single rank-2 block.
$$
B^{(t+1)}=B^{(t)}+\begin{bmatrix}u\mid v \end{bmatrix}I\begin{bmatrix}u\mid v \end{bmatrix}^T
$$
Due to the minimization ensuring that neighboring approximations are similar, while PSB was much more numerically stable it does converge slower than SR1 in cases where making rapid changes to the hessian was preferred.

### Derivation:
Since we are moving back to a minimization problem, we derive the update rule using the lagrangian and optimality conditions. To simplify the notation we can reformat the minimization problem in terms of the update itself $E$.
$$
\min_E\frac{1}{2}\|E\|^2_F\quad\text{s.t.}\quad E=E^T,Es=y-B^{(t)}s
$$
We can then write the Lagrangian $L(E,\lambda,\Omega)$ of this new problem, where we simplify since we know that $\text{Tr}(\Omega^T E-\Omega^TE^T)=\text{tr}(\Omega^TE-\Omega E^T)$.
$$
\begin{gather*}
L(E,\lambda,\Omega)=\frac{1}{2}\text{Tr}(E^TE)-\lambda^T(Es-r)-\text{Tr}(\Omega^T(E-E^T))\\
L(E,\lambda,\Omega)=\frac{1}{2}\text{Tr}(E^TE)-\lambda^T Es+\lambda^Tr-\text{Tr}(E(\Omega-\Omega^T))
\end{gather*}
$$
We can then take the gradient and set it equal to zero to find the form of the optimal $E$.
$$
\begin{gather*}
\nabla_EL=E-\lambda s^T-(\Omega-\Omega^T)=0\\
E=\lambda s^T+(\Omega-\Omega^T)
\end{gather*}
$$
To remove the $\Omega$ terms from the equation we can use the fact that we need $E=E^T$ to define them in terms of $s$. Since $\lambda$ is still an unknown we scale them by $2$ before plugging them back into the equation for $E$ for notational simplicity.
$$
\begin{gather*}
\lambda s^T+(\Omega-\Omega^T)=s\lambda^T+(\Omega^T-\Omega)\\
2(\Omega-\Omega^T)=s\lambda^T-\lambda s^T\\
(\Omega-\Omega^T)=\frac{1}{2}(s\lambda^T-\lambda s^T)\\
E=\lambda s^T+s\lambda^T
\end{gather*}
$$
We can then use this new definition of $E$ to plug in to the secant equation $Es=r$ to find the value for $\lambda$ using $\alpha=\lambda^Ts$ which will be solved in future steps.
$$
\begin{gather*}
(\lambda s^T+s\lambda^T)s=r\\
\lambda(s^Ts)+s(\lambda^Ts)=r\\
\lambda=\frac{r-\alpha s}{s^Ts}
\end{gather*}
$$
We can multiply both sides by $s^T$ to get $\alpha$ in terms of only $s$ and $r$, which derives the final $\lambda$.
$$
\begin{gather*}
s^T\lambda=\frac{s^Tr-\alpha(s^Ts)}{s^Ts}\\
\alpha=\frac{s^Tr}{s^Ts}-\alpha\rightarrow \alpha=\frac{s^Tr}{2s^Ts}\\
\lambda=\frac{r-\left(\frac{s^Tr}{2s^Ts}\right)s}{s^Ts}
\end{gather*}
$$
This $\lambda$ can be plugged into the rank-2 update for $E$ we had to derive the update rule for PSB, completing the derivation.

## DFP:
To try and reach a middle ground between the speed of SR1 and the stability of PSB, we can recall the benefit that having a positive definite matrix can have on the iteration. If we can keep the hessian positive definite, even in situations where the real hessian is not, we can improve the convergence of the Quasi-Newton Method. This is where DFP, standing for Davidon-Fletcher-Powell, comes in. The method redefines the minimization problem and uses the Mahalanobis norm rather than the Frobenius. We can define the minimization problem with the matrix $W$ defined such that $Wy^{(t)}=s^{(t)}$.
$$
\min_B\|B-B^{(t)}\|^2_W\quad\text{s.t.}\quad B=B^T,Bs^{(t)}=y^{(t)}
$$
To motivate the use of the weighted norm, we can prove one of the new properties of the updates made by DFP, that the update is scale invariant. Consider a scaled $\tilde{x}$ defined with some invertible transformation matrix $M$.
$$
x=M\tilde{x}
$$
With this definition we know that $\tilde{s}=M^{-1}s$ and $\tilde{y}=M^Ty$, and consequently that $\tilde{B}=M^TBM$ and $\tilde{W}=M^{-1}WM^{-T}$. We can then define the weighted norm between the new $\tilde{B}$ and the old $\tilde{B}^{(t)}$.
$$
\|\tilde{B}-\tilde{B}^{(t)}\|^2_W=\text{Tr}\left(\tilde{W}(\tilde{B}-\tilde{B}^{(t)})\tilde{W}(\tilde{B}-\tilde{B}^{(t)})^T\right)
$$
If we define $\Delta B=B-B^{(t)}$ for notation simplicity and plug in the definitions of each scaled variable with the originals, we see that a lot of the terms cancel.
$$
\begin{gather*}
\text{Tr}([M^{-1}WM^{-T}]\cdot[M^T\Delta BM]\cdot[M^{-1}WM^{-T}]\cdot[M^T\Delta B^{T}M])\\
\text{Tr}(M^{-1}W\Delta BW\Delta B^TM)
\end{gather*}
$$
Using the cycle property that $\text{Tr}(ABC)=\text{Tr}(CAB)$, we can cancel one more term and see that the objective on the scaled $\tilde{x}$ and the original $x$ are the exact same.
$$
\begin{gather*}
\text{Tr}(MM^{-1}W\Delta BW\Delta B^T)=\text{Tr}(W\Delta BW\Delta B^T)\\
\|\tilde{B}-\tilde{B}^{(t)}\|^2_W=\|B-B^{(t)}\|^2_W
\end{gather*}
$$
In practical applications this means that changes in units and other things that should not change the nature of the final result will not actually change the nature of the final result, unlike in previous methods.

The update rule for $B$ can be derived from this minimization and is defined using a rank-2 update below, which can be expanded and simplified to showcase another important, if not the most important property of the method.
$$
\begin{gather*}
B^{(t+1)}=\left(I-\frac{y^{(t)}(s^{(t)})^T}{(y^{(t)})^Ts^{(t)}}\right)B^{(t)}\left(I-\frac{s^{(t)}(y^{(t)})^T}{(y^{(t)})^Ts^{(t)}}\right)+\frac{y^{(t)}(y^{(t)})^T}{(y^{(t)})^Ts^{(t)}}\\
B^{(t+1)}=B^{(t)}+\frac{(y-B^{(t)}s)y^T+y(y-B^{(s)}s)^T}{y^Ts}-\frac{s^T(y-B^{(t)}s)}{(y^Ts)^2}yy^T
\end{gather*}
$$
Using the first definition, we can see that under certain conditions both terms are either positive-definite or positive-semidefinite. If $y^Ts>0$, then we know that the second term is PSD, and we can see that the first term inherits the definiteness of $B^{(t)}$ through some case analysis. This means that if we have a positive definite $B^{(t)}$ and $y^Ts>0$, then we have a positive definite $B^{(t+1)}$, which means we can always have a descent direction for our Quasi-Newton step. The condition $y^Ts>0$ can be trivially guaranteed by using a Backtracking Line-Search stepsize, so this can apply always in the method. 

This is the main distinction between DFP and previous methods, acting as the main difference in terms of convergence. Although theoretically an approximation that is always positive definite would seem bad, as it can lie about the curvature of the function if the real hessian is indefinite or even negative definite, it ends up having better convergence than when the approximation is as realistic as possible. This gives the method much better convergence results compared to PSB, although the bottleneck of being limited in how much it can change the hessian still keeps it slightly slower than SR1.

### Derivation:
We again use the lagrangian $L(B,\lambda)$ where we scale the objective by $\frac{1}{2}$ to simplify the notation of the final update.
$$
L(B,\lambda)=\frac{1}{2}\|B-B^{(t)}\|^2_W+\lambda^T(Bs-y)
$$
We can then take the derivative and set it equal to $0$ to find the general form for $B$. Since we know that $Ws=y$, we also know that $Ws^{-1}=y$ which is used for simplification.
$$
\begin{gather*}
\nabla_BL=W(B-B^{(t)})W+\lambda s^T+s\lambda ^T=0\\
B=B^{(t)}-W^{-1}(\lambda s^T+s\lambda^T)W^{-1}\\
B=B^{(t)}-(\lambda y^T+y\lambda^T)
\end{gather*}
$$
To solve for $\lambda$ we apply this definition for $B$ to the secant equation and use the same trick as the previous derivation by defining $\alpha=\lambda^Ts$ and seeing that we can solve for it later.
$$
\begin{gather*}
(B^{(t)}-\lambda y^T-y\lambda^T)s=y\\
B^{(t)}s-\lambda(y^Ts)-y(\lambda^Ts)=y\\
\lambda(y^Ts)=B^{(t)}s-y-\alpha y\\
\lambda=\frac{B^{(t)}s-(1+\alpha y)}{y^Ts}
\end{gather*}
$$
To solve for $\alpha$ we can plug in the definition of $\lambda$ into the definition we set for $\alpha$ and simplify.
$$
\begin{gather*}
\alpha=\left(\frac{B^{(t)}s-(1+\alpha)y}{y^Ts}\right)^Ts=\frac{s^TB^{(t)}s-(1+\alpha)y^Ts}{y^Ts}\\
\alpha(y^Ts)=s^TB^{(t)}s-y^Ts\\
2\alpha(y^Ts)=s^TB^{(t)}s-y^Ts\\
\alpha=\frac{s^TB^{(t)}s}{2y^Ts}-\frac{1}{2}
\end{gather*}
$$
Plugging this definition for $\alpha$ back into our definition for $\lambda$ then gives us the final definition of $\lambda$.
$$
\begin{gather*}
(1+\alpha)=\frac{s^TB^{(t)}s}{2y^Ts}+\frac{1}{2}=\frac{s^TB^{(t)}s+y^Ts}{2y^Ts}\\
\lambda=\frac{B^{(t)}s}{y^Ts}-\frac{(s^TB^{(t)}s+y^Ts)y}{2(y^Ts)^2}
\end{gather*}
$$
Plugging this $\lambda$ into the rank-2 update we had previously gives the update rule for DFP, completing the derivation.

## BFGS:
Although DFP has an approximation that gets all of the properties we want, it still does not have as fast convergence as SR1. BFGS, standing for Broyden-Fletcher-Goldfarb-Shanno, further improves on DFP by changing the minimization from dealing with the hessian approximation to the inverse hessian, which is the matrix that is actually used in the method.
$$
\min_H\|H-H^{(t)}\|^2_W\quad\text{s.t.}\quad H=H^T,Hy^{(t)}=s^{(t)}
$$
Solving the optimization problem leads to the following update for the inverse hessian, which can be inverted using the same Woodbury Identity to see what the update means for the hessian itself. It can also be seen that this forms the dual of DFP, where the equation for the inverse is used for the hessian and vice versa, which is why the derivation won't be repeated.
$$
\begin{gather*}
H^{(t+1)}=\left(I-\frac{s^{(t)}(y^{(t)})^T}{(s^{(t)})^Ty^{(t)}}\right)H^{(t)}\left(I-\frac{y^{(t)}(s^{(t)})^T}{(s^{(t)})^Ty^{(t)}}\right)+\frac{s^{(t)}(s^{(t)})^T}{(s^{(t)})^Ty^{(t)}}\\
B^{(t+1)}=B^{(t)}+\frac{y^{(t)}(y^{(t)})^T}{(y^{(t)})^Ts^{(t)}}-\frac{B^{(t)}s^{(t)}(s^{(t)})^TB^{(t)}}{(s^{(t)})^TB^{(t)}s^{(t)}}
\end{gather*}
$$
This leads to an update that has all of the same properties as DFP, but has much more stable iterations because minimizing the difference between the hessians and the inverse hessians lead to two different matrices. In minimizing the difference between the matrices that are actually used in the algorithm, the method is able to converge just as fast as SR1 while having all of the new benefits that have been found in the methods following it. BFGS also has a nice property of being self-correcting, where a bad initial approximation or a poor line search can be fixed, where DFP often stalls.
