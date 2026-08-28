# Gradient, Hessian, and Curvature

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

---

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

---

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

---

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

## 5. Why Does Gradient = 0 Not Mean Global Minimum?

If:

$$
\nabla L = 0
$$

we only know that the point is a **critical point**.

It could be:

- a local minimum
- a local maximum
- a saddle point
- another flat/degenerate point

Second-order information from the Hessian can help determine the local curvature around the critical point.

However:

> A local minimum is not necessarily a global minimum.

---

## Key Idea

The main relationship is:

> **Gradient:** How is the loss changing?  
> **Hessian:** How is the gradient changing?  
> **Curvature:** How does the shape of the loss surface affect optimization?


# Constrained Optimization

## 1. Basic Idea

**Unconstrained optimization** searches for the best solution over all possible values:

`min f(x)`

**Constrained optimization** searches only within a feasible set `S`:

`min f(x), where x ∈ S`

- **Feasible set / region**: the set of allowed solutions.
- **Feasible point**: a point inside the feasible set.
- The unconstrained optimum may not be feasible, so the constrained optimum can be different.

---

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

---

## 3. Equality vs. Inequality Constraints

### Equality Constraint

`g(x) = 0`

Geometrically, it usually reduces the dimension of the feasible set.

### Inequality Constraint

`h(x) ≤ 0`

Geometrically, it acts like a **wall** that separates feasible and infeasible regions.

---

## 4. Lagrangian and KKT Multipliers

The generalized Lagrangian combines the objective and constraints:

`L(x, λ, α) = f(x) + Σ λᵢgᵢ(x) + Σ αⱼhⱼ(x)`

Main idea:

> Convert a constrained optimization problem into an optimization problem where the effects of the constraints are encoded in the Lagrangian.

- `λ`: multipliers for equality constraints
- `α`: multipliers for inequality constraints

---

## 5. Active and Inactive Constraints

For an inequality constraint:

`h(x) ≤ 0`

At the solution `x*`:

- `h(x*) = 0` → **active constraint**
  - The solution touches the boundary ("hits the wall").

- `h(x*) < 0` → **inactive constraint**
  - The solution does not touch the boundary.

---

## 6. Complementary Slackness

For each inequality constraint:

`αⱼ hⱼ(x) = 0`

This means:

- If the constraint is **active**, `hⱼ(x) = 0`.
- If the constraint is **inactive**, its multiplier `αⱼ = 0`.

Intuition:

> Either the solution hits the wall, or the wall has no influence on the solution.

---

## 7. Key Intuition

Gradient descent gives the best local direction for decreasing the objective **without considering constraints**.

With constraints, that direction may be infeasible.

Therefore, constrained optimization searches for:

> The best solution among all feasible solutions.