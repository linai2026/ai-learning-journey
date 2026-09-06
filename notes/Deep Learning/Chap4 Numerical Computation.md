# 4.1 Overflow and Underflow

## Overflow

### Definition

Overflow occurs when the absolute value of a number exceeds the maximum value that can be represented by a floating-point format. The value is rounded to **Infinity (`∞`)**.

### Why Does It Happen?

Computers have a limited number of bits to represent numbers.

For example, IEEE 754 floating-point numbers have a finite maximum value. Any computation that exceeds this limit results in **overflow**.

### Where Does It Commonly Occur in AI?

Overflow is common in exponential computations, such as:

- `softmax`
- `exp(x)`
- `e^x`
- `sigmoid` (internally computes `exp`)
- `log-sum-exp`

### How Does PyTorch Handle Overflow?

PyTorch cannot completely prevent overflow.

Instead, many built-in functions use **numerically stable algorithms** to reduce the risk.

For example, instead of implementing Softmax manually, use:

```python
torch.softmax(x)
```

PyTorch internally applies numerically stable techniques.

During training, the following functions are also implemented with numerical stability in mind:

- `CrossEntropyLoss`
- `LogSoftmax`

These implementations are much safer than writing the equations manually.

---

## Underflow

### Definition

Underflow occurs when the absolute value of a number becomes smaller than the minimum value that can be represented by a floating-point format. The value is rounded to **0**.

### Why Does It Happen?

Floating-point numbers have both:

- a maximum representable value
- a minimum nonzero representable value

Any value smaller than the minimum representable value becomes **0**.

### Why Is Underflow Common in Neural Network Training?

Neural network training involves many operations such as:

- repeated multiplication
- exponential computations
- backpropagation

If many very small numbers are multiplied together, the result rapidly approaches zero, eventually causing **underflow**.

When gradients become zero, earlier layers can no longer receive gradient information.

As a result:

- parameters stop updating
- learning slows down or completely stops

This phenomenon is closely related to the **Vanishing Gradient** problem.

Therefore, **underflow is one of the major causes of vanishing gradients.**

---

## Key Takeaway

- **Overflow:** Numbers become too large and are rounded to **Infinity (`∞`)**.
- **Underflow:** Numbers become too small and are rounded to **0**.
- Deep learning frameworks such as **PyTorch** provide numerically stable implementations (e.g., `torch.softmax`, `CrossEntropyLoss`, and `LogSoftmax`) to reduce these problems.

---

# 4.2 Poor Conditioning

## Poor Conditioning

### Why is a Nearly Singular Matrix Problematic?

**Answer:**

A nearly singular matrix has a very small **singular value**, making its **condition number** very large.

A large condition number means that even a tiny error in the input can produce a much larger error in the output.

The matrix is still invertible, but numerical computations become unstable.

**Key Idea:**

> Poor conditioning means that small input errors are greatly amplified.

### Why Do Errors Keep Growing?

**Answer:**

Computers store numbers with limited precision, so every floating-point computation introduces a small **rounding error**.

If a matrix is poorly conditioned, these small errors are amplified during computation.

The amplified output error becomes the input of the next computation, causing errors to accumulate over many operations.

**Key Idea:**

> Numerical errors are not only propagated—they are amplified.

### Why Is This Important for Neural Networks?

**Answer:**

Training a neural network involves a large number of matrix operations.

If some matrices are poorly conditioned, small numerical errors can be amplified repeatedly during forward and backward computations.

As computations continue, these errors accumulate and may reduce numerical stability, making training more difficult.

**Key Idea:**

> Deep neural networks perform many matrix operations, so numerical stability is critical.

---

# 4.3 Gradient-Based Optimization

## Why do we need the gradient?

- A gradient is a vector of partial derivatives of the loss function.
- It tells us the direction in which the loss increases the fastest.
- By moving in the opposite direction (the negative gradient), we can reduce the loss efficiently.
- This is much more effective than trying random directions.

## Why not search randomly?

- Random search ignores information about the shape of the loss function.
- The gradient uses local information to find the steepest descent direction.
- In deep learning, models often have millions of parameters, making random search extremely inefficient.

## Why does the loss become smaller?

- Parameters are updated in the direction of the negative gradient.
- Each update usually reduces the loss.
- Repeating this process gradually moves the parameters toward a local minimum, a global minimum, or a point where the gradient is close to zero.

---

## Local Minimum

A point where the loss is lower than every nearby point.

## Global Minimum

A point where the loss is the lowest over the entire parameter space.

## Saddle Point

A critical point where the gradient is zero, but the point is neither a local minimum nor a local maximum.

## Critical Point

A point where the gradient is zero.

## Why Are Saddle Points a Bigger Problem?

Deep neural networks usually have millions of parameters.

To be a **Local Minimum**, **every** direction must have no descending path.

If **even one** direction can still decrease the loss, the point becomes a **Saddle Point**.

As the number of dimensions increases:

- The probability of a Local Minimum decreases.
- The number of Saddle Points increases dramatically.

Therefore, modern deep neural networks are more likely to encounter **Saddle Points** than **Local Minima** during training.

---

## 1. Gradient

For a multivariable function:

$$
L(w_1, w_2, \dots, w_n)
$$

the gradient is:

$$
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\vdots \\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
$$

The gradient tells us:

- how sensitive the loss is to each parameter
- the direction of the fastest increase of the loss
- the negative gradient points toward the fastest local decrease

Therefore, Gradient Descent moves in the direction:

$$
-\nabla L
$$

## 2. Why Is the Gradient a Vector?

For a function with many parameters, each parameter has its own partial derivative.

For example:

$$
L(w_1, w_2, w_3)
$$

has:

$$
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\frac{\partial L}{\partial w_3}
\end{bmatrix}
$$

So:

> One parameter -> one derivative  
> Many parameters -> many derivatives -> gradient vector

## 3. Second Derivative and Hessian

The first derivative describes the **rate of change**.

The second derivative describes the **rate of change of the rate of change**.

In one dimension:

$$
f'(x) \rightarrow \text{slope}
$$

$$
f''(x) \rightarrow \text{curvature}
$$

For multivariable functions, second-order derivative information is represented by the **Hessian matrix**.

Therefore:

> Gradient -> slope  
> Hessian -> curvature

## 4. Why Does Curvature Matter in Optimization?

Curvature tells us how quickly the gradient changes.

High curvature means the gradient can change rapidly.

If the learning rate is too large in a high-curvature direction, Gradient Descent may:

- overshoot the minimum
- oscillate around the minimum
- become unstable

Different curvatures in different directions can also cause Gradient Descent to zig-zag and converge slowly.

Therefore:

> High curvature often requires a smaller learning rate.

---

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

---

# 4.4 Constrained Optimization

## 1. Basic Idea

**Unconstrained optimization** searches for the best solution over all possible values:

`min f(x)`

**Constrained optimization** searches only within a feasible set `S`:

`min f(x), where x ∈ S`

- **Feasible set / region**: the set of allowed solutions.
- **Feasible point**: a point inside the feasible set.
- The unconstrained optimum may not be feasible, so the constrained optimum can be different.

## 2. Projected Gradient Descent

A normal gradient descent step may move `x` outside the feasible region.

One solution is:

1. Take a normal gradient descent step.
2. If the new point is outside `S`, project it back to the nearest feasible point.

**Point projection:**

> Move first, then pull the point back into the feasible region.

Another approach is to project the **gradient** onto the tangent space before moving.

**Gradient projection:**

> Remove the part of the gradient that points in an infeasible direction, then move.

## 3. Equality vs. Inequality Constraints

### Equality Constraint

`g(x) = 0`

Geometrically, it usually reduces the dimension of the feasible set.

### Inequality Constraint

`h(x) ≤ 0`

Geometrically, it acts like a **wall** that separates feasible and infeasible regions.

## 4. Lagrangian and KKT Multipliers

The generalized Lagrangian combines the objective and constraints:

`L(x, λ, α) = f(x) + Σ λᵢgᵢ(x) + Σ αⱼhⱼ(x)`

Main idea:

> Convert a constrained optimization problem into an optimization problem where the effects of the constraints are encoded in the Lagrangian.

- `λ`: multipliers for equality constraints
- `α`: multipliers for inequality constraints

## 5. Active and Inactive Constraints

For an inequality constraint:

`h(x) ≤ 0`

At the solution `x*`:

- `h(x*) = 0` → **active constraint**
  - The solution touches the boundary ("hits the wall").

- `h(x*) < 0` → **inactive constraint**
  - The solution does not touch the boundary.

## 6. Complementary Slackness

For each inequality constraint:

`αⱼ hⱼ(x) = 0`

This means:

- If the constraint is **active**, `hⱼ(x) = 0`.
- If the constraint is **inactive**, its multiplier `αⱼ = 0`.

Intuition:

> Either the solution hits the wall, or the wall has no influence on the solution.

---

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

## 4. Newton's Method

The least squares objective is a quadratic function.

Its Hessian is:

$$
H = A^TA
$$

Newton's method uses a quadratic approximation of the objective.

Because the true objective is already quadratic, this approximation is exact.

Therefore, under suitable invertibility conditions, Newton's method can reach the global minimum in a single step.

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