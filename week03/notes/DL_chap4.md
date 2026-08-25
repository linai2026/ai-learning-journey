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