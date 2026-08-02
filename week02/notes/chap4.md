# Numerical Computation Notes (Deep Learning Book - Chapter 4.1)

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