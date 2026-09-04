# Feature Scaling

## Why Feature Scaling Matters

Gradient Descent is sensitive to differences in feature scales.

For example:

```text
Feature 1: 0 ~ 1
Feature 2: 0 ~ 100000
```

Large differences in scale can cause **unbalanced gradient updates**:

- A learning rate may be too large in one direction → **overshooting**
- The same learning rate may be too small in another direction → **slow convergence**

Feature scaling makes the optimization problem better conditioned, allowing Gradient Descent to converge **faster and more stably**.

---

## Standardization

A common feature scaling method is **standardization**:

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

where:

- $x_j$: original feature
- $\mu_j$: mean of the feature
- $\sigma_j$: standard deviation of the feature
- $x'_j$: standardized feature

After standardization:

- Mean ≈ `0`
- Standard deviation ≈ `1`
- Variance ≈ `1`

Standardized values are **dimensionless** because the original physical unit is removed.

For example, a standardized value of `-0.8` means that the value is **0.8 standard deviations below the mean**.

---

## Standardization with NumPy

Suppose:

```python
X.shape
# (150, 2)
```

We can standardize each feature independently:

```python
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

Here:

```python
X.mean(axis=0).shape
# (2,)

X.std(axis=0).shape
# (2,)
```

`axis=0` means calculating the mean and standard deviation **across all samples for each feature**.

NumPy broadcasting allows:

```text
(150, 2) - (2,) → (150, 2)
```

Therefore, every feature is automatically centered using its own mean and scaled using its own standard deviation.

---

## Effect on Gradient Descent

Without feature scaling:

```text
Different feature scales
        ↓
Unbalanced gradient updates
        ↓
Difficult learning-rate selection
        ↓
Overshooting or slow convergence
```

With standardization:

```text
Similar feature scales
        ↓
More balanced gradient updates
        ↓
Easier learning-rate selection
        ↓
Faster and more stable convergence
```

---

## Key Takeaway

> **Feature scaling improves the optimization geometry, helping Gradient Descent find the minimum more efficiently.**





# Stochastic Gradient Descent (SGD)

## 1. Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the **entire training set** before updating the parameters.

$$
w \leftarrow w - \eta \nabla L
$$

For a dataset with $N$ samples:

- Samples per update: $N$
- Updates per epoch: **1**
- Gradient: stable and accurate
- Disadvantage: computationally expensive for large datasets

---

## 2. Stochastic Gradient Descent (SGD)

SGD updates the parameters using **one training sample at a time**.

For each sample:

$$
w \leftarrow w - \eta \nabla L_i
$$

For a dataset with $N$ samples:

- Samples per update: **1**
- Updates per epoch: **N**
- Parameter updates are much more frequent than Batch GD.

### Key Idea

The gradient from one sample is only a **noisy estimate** of the full gradient:

$$
\nabla L_i \approx \nabla L
$$

Individual estimates may be inaccurate, but over many randomly selected samples, they approximate the full-dataset gradient.

---

## 3. Why SGD Is Noisy

Different samples produce different gradient estimates.

Therefore, the optimization path and loss curve fluctuate more than with Batch GD.

### Advantage of Noise

Noise is not always bad.

Random fluctuations can sometimes help the optimizer escape:

- shallow local minima
- flat regions

---

## 4. Shuffling

Training samples should usually be shuffled before each epoch.

```python
X, y = shuffle(X, y)
```

Shuffling:

- prevents systematic bias from fixed sample ordering
- prevents repetitive update cycles
- makes gradient estimates more random

`X` and `y` must always be shuffled **together** so that each sample keeps its correct label.

---

## 5. Mini-Batch Gradient Descent

Mini-batch GD uses a **small group of samples** for each parameter update.

Example:

```text
Dataset size = 10,000
Batch size = 100
```

Then:

$$
\frac{10000}{100}=100
$$

So there are approximately **100 parameter updates per epoch**.

### Comparison

| Method | Samples per Update | Updates per Epoch |
|---|---:|---:|
| Batch GD | Entire dataset | 1 |
| SGD | 1 | $N$ |
| Mini-batch GD | Small batch | $N / \text{batch size}$ |

---

## 6. Why Mini-Batch Is Common in Deep Learning

Mini-batch training combines the advantages of Batch GD and SGD:

- frequent parameter updates
- more stable gradients than one-sample SGD
- efficient matrix operations
- vectorization
- GPU parallelism

Therefore, modern neural networks are usually trained with **mini-batch SGD-family optimizers**.

---

## 7. Epoch vs. Parameter Update

An **epoch** means:

> The entire training dataset has been processed once.

An epoch is **not** the same as one parameter update.

Example:

```text
Dataset size = 10,000
Batch size = 100
```

One epoch contains approximately:

```text
100 mini-batches
→ 100 parameter updates
```

In general:

$$
\text{updates per epoch}
\approx
\frac{\text{number of samples}}
{\text{batch size}}
$$

---

## 8. SGD in Adaline

Batch GD processes the entire dataset before updating:

```python
output = self.activation(self.net_input(X))
```

SGD processes one sample at a time:

```python
for xi, target in zip(X, y):
    self._update_weights(xi, target)
```

For every sample:

```text
sample
  ↓
prediction
  ↓
error
  ↓
gradient
  ↓
parameter update
```

The update still follows the basic Gradient Descent idea:

$$
w \leftarrow w - \eta \nabla L
$$

The main difference is that the gradient is estimated from **one sample instead of the entire dataset**.

---

## 9. Online Learning

SGD naturally supports **online learning**.

New training data can be used to update an existing model without restarting training.

```python
model.partial_fit(X_new, y_new)
```

`partial_fit()`:

- keeps the existing weights
- learns from newly arriving data
- does not reinitialize the model

---

## 10. Learning Rate in SGD

Because SGD is noisy, a large learning rate can cause excessive fluctuations.

The learning rate is often decreased during training:

$$
\eta_t \downarrow
$$

General idea:

```text
Early training
→ larger learning rate
→ faster movement

Later training
→ smaller learning rate
→ more precise convergence
```

This idea later appears in neural networks as **learning-rate scheduling**.

---

# Key Takeaways

- **Batch GD:** entire dataset → one update.
- **SGD:** one sample → one update.
- **Mini-batch GD:** small group of samples → one update.
- SGD uses a **noisy estimate of the full gradient**.
- Frequent updates can make SGD learn faster.
- Training data should usually be **shuffled every epoch**.
- An **epoch** means processing the entire training set once.
- Mini-batches enable efficient **matrix operations, vectorization, and GPU computation**.
- `partial_fit()` enables **online/incremental learning**.
- Modern neural-network training mainly uses **mini-batch SGD-family methods**.