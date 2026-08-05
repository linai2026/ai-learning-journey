# Numerical Computation Notes (Deep Learning Book - Chapter 4)

## Overflow

### Definition

Overflow occurs when the absolute value of a number exceeds the maximum value that can be represented by a floating-point format. The value is rounded to **Infinity (`∞`)**.

### My Understanding

Computers do not store exact real numbers. Instead, they store approximate values using a finite number of bits.

When a number becomes larger than the maximum representable floating-point value, it is stored as **Infinity**.

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

### My Understanding

When a number becomes extremely close to zero, the computer can no longer distinguish it from zero.

As a result, it is stored as **0**.

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

## Why is a Nearly Singular Matrix Problematic?

**Answer:**

A nearly singular matrix has a very small **singular value**, making its **condition number** very large.

A large condition number means that even a tiny error in the input can produce a much larger error in the output.

The matrix is still invertible, but numerical computations become unstable.

**Key Idea:**

> Poor conditioning means that small input errors are greatly amplified.


---

## Why Do Errors Keep Growing?

**Answer:**

Computers store numbers with limited precision, so every floating-point computation introduces a small **rounding error**.

If a matrix is poorly conditioned, these small errors are amplified during computation.

The amplified output error becomes the input of the next computation, causing errors to accumulate over many operations.

**Key Idea:**

> Numerical errors are not only propagated—they are amplified.


---

## Why Is This Important for Neural Networks?

**Answer:**

Training a neural network involves a large number of matrix operations.

If some matrices are poorly conditioned, small numerical errors can be amplified repeatedly during forward and backward computations.

As computations continue, these errors accumulate and may reduce numerical stability, making training more difficult.

**Key Idea:**

> Deep neural networks perform many matrix operations, so numerical stability is critical.


---

## Why do we need the gradient?

- A gradient is a vector of partial derivatives of the loss function.
- It tells us the direction in which the loss increases the fastest.
- By moving in the opposite direction (the negative gradient), we can reduce the loss efficiently.
- This is much more effective than trying random directions.

---

## Why not search randomly?

- Random search ignores information about the shape of the loss function.
- The gradient uses local information to find the steepest descent direction.
- In deep learning, models often have millions of parameters, making random search extremely inefficient.

---

## Why does the loss become smaller?

- Parameters are updated in the direction of the negative gradient.
- Each update usually reduces the loss.
- Repeating this process gradually moves the parameters toward a local minimum, a global minimum, or a point where the gradient is close to zero.