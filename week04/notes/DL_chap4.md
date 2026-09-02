# Newton's Method, Lipschitz Continuity, and Convex Optimization

## 1. First-Order vs. Second-Order Optimization

### First-Order Optimization

First-order methods use only the **gradient**:

$$
\nabla f(x)
$$

Example: **Gradient Descent**

$$
x_{\text{new}} = x - \eta \nabla f(x)
$$

- Gradient tells us the direction and rate of change.
- It does not directly use curvature information.

### Second-Order Optimization

Second-order methods also use the **Hessian matrix**:

$$
H(x)
$$

Example: **Newton's Method**

$$
x_{\text{new}} = x - H(x)^{-1}\nabla f(x)
$$

- Gradient describes the slope.
- Hessian describes the curvature in different directions.

---

## 2. Newton's Method

Newton's method locally approximates a function using a **second-order Taylor expansion**.

It then finds the critical point of this quadratic approximation.

The resulting update rule is:

$$
x_{\text{new}} = x - H(x)^{-1}\nabla f(x)
$$

### Quadratic Functions

For a quadratic function:

$$
f(x) = \frac{1}{2}x^T A x + b^T x + c
$$

the second-order Taylor approximation is exactly the original function.

Therefore, if the quadratic function is **positive definite**, Newton's method can reach the minimum in **one step**.

---

## 3. Newton's Method and Saddle Points

Newton's method searches for a nearby **critical point**:

$$
\nabla f(x) = 0
$$

However, a critical point is not necessarily a minimum. It can also be:

- a local maximum
- a saddle point

If the Hessian has both positive and negative eigenvalues, the critical point is a saddle point.

Therefore:

> Newton's method can be attracted to critical points that are not minima.

When the Hessian is positive definite:

$$
\lambda_i(H) > 0
$$

the nearby critical point is a local minimum.

---

## 4. Lipschitz Continuity

A function is Lipschitz continuous if:

$$
|f(x) - f(y)| \leq L\|x-y\|
$$

where $L$ is the **Lipschitz constant**.

Intuition:

> The change in output is bounded by a constant times the change in input.

In other words, the function cannot change arbitrarily fast.

Lipschitz continuity is useful for analyzing and proving the behavior of optimization algorithms.

---

## 5. Convex Optimization

For a convex function, the Hessian is positive semidefinite everywhere:

$$
H(x) \succeq 0
$$

Important properties:

- There are no saddle points.
- Every local minimum is also a global minimum.

$$
\boxed{\text{Local Minimum} = \text{Global Minimum}}
$$

This makes convex functions much easier to optimize.

However, most deep learning optimization problems are **non-convex**, so these guarantees generally do not apply.

---

## Key Takeaways

- **Gradient Descent** is a first-order method because it uses the gradient.
- **Newton's Method** is a second-order method because it uses both the gradient and Hessian.
- The **Hessian** describes curvature in different directions.
- Newton's method minimizes a local quadratic approximation.
- For positive definite quadratic functions, Newton's method can reach the minimum in one step.
- Newton's method may converge to saddle points because it searches for critical points, not necessarily minima.
- **Lipschitz continuity** means the rate of change is bounded.
- For **convex functions**, every local minimum is a global minimum.
- Most deep learning problems are non-convex.


# 4.5 Linear Least Squares

## 1. Least Squares Problem

We want to find `x` that minimizes:

$$
f(x) = \frac{1}{2}\|Ax-b\|_2^2
$$

where:

- `A` = known matrix
- `x` = parameters to optimize
- `b` = target
- `Ax` = prediction
- `Ax - b` = residual (prediction error)

The goal is to make `Ax` as close to `b` as possible.

---

## 2. Gradient of the Least Squares Objective

The gradient is:

$$
\nabla_x f(x)
= A^T(Ax-b)
= A^TAx-A^Tb
$$

The gradient points in the direction of the fastest increase of the objective.

Therefore, gradient descent moves in the opposite direction:

$$
x \leftarrow x - \epsilon \nabla_x f(x)
$$

or:

$$
x \leftarrow x-\epsilon(A^TAx-A^Tb)
$$

where `epsilon` is the step size (learning rate).

---

## 3. Stopping Criterion

Gradient descent continues while:

$$
\|\nabla_x f(x)\|_2 > \delta
$$

where `delta` is a small positive tolerance.

When:

$$
\|\nabla_x f(x)\|_2 \leq \delta
$$

the gradient is sufficiently close to zero, so the algorithm stops.

---

## 4. Newton's Method

The least squares objective is a quadratic function.

Its Hessian is:

$$
H = A^TA
$$

Newton's method uses a quadratic approximation of the objective.

Because the true objective is already quadratic, this approximation is exact.

Therefore, under suitable invertibility conditions, Newton's method can reach the global minimum in a single step.

---

## 5. Constrained Least Squares

Now add the constraint:

$$
x^Tx \leq 1
$$

which is equivalent to:

$$
\|x\|_2 \leq 1
$$

The solution must therefore lie inside or on the unit sphere.

### Case 1: Unconstrained Solution Is Feasible

If the unconstrained solution satisfies:

$$
x^Tx \leq 1
$$

the constraint is inactive.

The unconstrained solution is also the constrained solution.

### Case 2: Unconstrained Solution Is Infeasible

If:

$$
x^Tx > 1
$$

the constraint becomes active.

The constrained optimum occurs on the boundary:

$$
x^Tx = 1
$$

The optimum is on the boundary because moving toward the unconstrained optimum would continue decreasing the objective until the constraint prevents further movement.

---

## 6. Lagrangian

Define the constraint function:

$$
g(x) = x^Tx - 1
$$

The Lagrangian is:

$$
L(x,\lambda)
=
f(x)+\lambda(x^Tx-1)
$$

with:

$$
\lambda \geq 0
$$

The constrained problem can be expressed as:

$$
\min_x \max_{\lambda \geq 0} L(x,\lambda)
$$

Therefore:

- perform gradient **descent** with respect to `x`
- perform gradient **ascent** with respect to `lambda`

---

## 7. Solving for x

Differentiate the Lagrangian with respect to `x`:

$$
\nabla_x L
=
A^TAx-A^Tb+2\lambda x
$$

At the optimum:

$$
A^TAx-A^Tb+2\lambda x=0
$$

Therefore:

$$
(A^TA+2\lambda I)x=A^Tb
$$

and:

$$
x=(A^TA+2\lambda I)^{-1}A^Tb
$$

Increasing `lambda` generally reduces the norm of `x`.

Thus, `lambda` controls the strength of the constraint.

---

## 8. Updating the Lagrange Multiplier

Differentiate the Lagrangian with respect to `lambda`:

$$
\frac{\partial L}{\partial\lambda}
=
x^Tx-1
$$

If:

$$
x^Tx>1
$$

then:

$$
\frac{\partial L}{\partial\lambda}>0
$$

Gradient ascent increases `lambda`.

This increases the penalty on large values of `x`, causing:

$$
\|x\|_2 \downarrow
$$

The process continues until the active constraint is satisfied:

$$
x^Tx=1
$$

and therefore:

$$
\frac{\partial L}{\partial\lambda}=0
$$

---

## 9. Connection to KKT Conditions

For the constraint:

$$
x^Tx-1\leq0
$$

complementary slackness gives:

$$
\lambda(x^Tx-1)=0
$$

Therefore:

- If the constraint is inactive:

$$
x^Tx<1
\quad\Rightarrow\quad
\lambda=0
$$

- If the constraint is active:

$$
\lambda>0
\quad\Rightarrow\quad
x^Tx=1
$$

---

## Key Takeaways

1. Linear least squares minimizes squared prediction error:

$$
\frac{1}{2}\|Ax-b\|^2
$$

2. Its gradient is:

$$
\nabla f(x)=A^T(Ax-b)
$$

3. Gradient descent updates:

$$
x \leftarrow x-\epsilon\nabla f(x)
$$

4. Newton's method is especially effective because the objective is quadratic.

5. Adding `x^T x <= 1` turns the problem into constrained optimization.

6. The Lagrange multiplier `lambda` controls the strength of the constraint.

7. If the unconstrained optimum violates the constraint, the constrained optimum lies on the boundary:

$$
x^Tx=1
$$

8. We minimize over `x` but maximize over `lambda`:

$$
\min_x\max_{\lambda\geq0}L(x,\lambda)
$$

9. This example connects least squares, gradient descent, Newton's method, convex optimization, Lagrangians, and KKT conditions.