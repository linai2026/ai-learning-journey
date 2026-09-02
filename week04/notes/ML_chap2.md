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