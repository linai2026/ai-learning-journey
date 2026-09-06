# Perceptron: Core Concepts

## 1. Features, Labels, and Predictions

- **Feature (`x`)**: an input variable used by the model to make a prediction.
- **Label (`y`)**: the true target associated with a training sample.
- **Prediction (`ŷ`)**: the output produced by the model for an input.

Key distinction:

- $y$ = true label
- $\hat{y}$ = model prediction

## 2. What Does a Perceptron Learn?

A perceptron learns its **parameters**:

- weights (`w`)
- bias (`b`)

Training means adjusting these parameters based on prediction errors.

## 3. Net Input

The perceptron calculates:

$$
z = w^T x + b
$$

where:

- $x$ = input features
- $w$ = weights
- $b$ = bias
- $z$ = net input

### ⚠️ Important: $z$ is NOT the final prediction.

The correct flow is:

$$
x \rightarrow z \rightarrow \hat{y}
$$

The model first calculates $z$, then uses a decision function to produce the prediction $\hat{y}$.

## 4. Weights

Each feature has a corresponding weight:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

### ⚠️ Important: A weight is not simply "feature importance."

A weight represents both:

- **strength** of the feature's influence
- **direction** of the feature's influence

A positive weight tends to increase $z$, while a negative weight tends to decrease $z$.

## 5. Decision Function

For binary classification, a decision function converts $z$ into a class prediction.

For example:

$$
\hat{y} =
\begin{cases}
1, & z \geq 0 \\
0, & z < 0
\end{cases}
$$

Therefore:

$$
x
\rightarrow w^T x + b
\rightarrow z
\rightarrow \text{decision function}
\rightarrow \hat{y}
$$

## 6. How Does a Perceptron Learn?

During training:

1. Calculate the prediction $\hat{y}$.
2. Compare $\hat{y}$ with the true label $y$.
3. Calculate the classification error.
4. Update the weights and bias.

The error is:

$$
\text{error} = y - \hat{y}
$$

A typical weight update is:

$$
\Delta w_j = \eta (y-\hat{y})x_j
$$

where $\eta$ is the **learning rate**.

### ⚠️ Important: The goal is NOT to make $\hat{y}$ gradually "closer" to the label.

For a perceptron, the prediction is discrete:

$$
\hat{y} \in \{0,1\}
$$

Instead, the model adjusts its parameters so that incorrectly classified samples are more likely to fall on the correct side of the decision boundary.

## 7. Training vs. Prediction

### Training

$$
x
\rightarrow z
\rightarrow \hat{y}
\rightarrow \text{compare with } y
\rightarrow \text{error}
\rightarrow \text{update } w,b
$$

### Prediction

For unseen data:

$$
x_{\text{new}}
\rightarrow w^T x_{\text{new}} + b
\rightarrow z
\rightarrow \text{decision function}
\rightarrow \hat{y}
$$

During prediction, the learned parameters are used **without updating them**.

---

# Perceptron Training

## 1. What does `fit()` do?

`fit()` trains the Perceptron by repeatedly going through the training data.

For each training example:

1. Make a prediction.
2. Compare it with the true label.
3. Update the weights and bias if the prediction is wrong.

The main trainable parameters are:

- **Weights (`w`)**
- **Bias (`b`)**

## 2. What does `predict()` do?

`predict()` first calculates the net input:

`z = w^T x + b`

Then it applies a threshold to convert the net input into a class label.

Flow:

`input → net input → threshold → predicted class`

## 3. What are weights?

Each feature has a corresponding weight.

A weight controls the **strength and direction** of that feature's influence on the prediction.

- Large positive weight → stronger positive influence
- Large negative weight → stronger negative influence
- Weight near zero → weaker influence

## 4. What happens after a wrong prediction?

The Perceptron calculates:

`update = learning_rate × (target - prediction)`

Then it updates:

`w = w + update × x`

`b = b + update`

A wrong prediction also increases the error count for that epoch.

If the prediction is correct, `update = 0`, so the parameters are not changed.

## 5. Why do we need multiple epochs?

One pass through the training set may not be enough to find suitable weights and bias.

An **epoch** means one complete pass through the training set.

Updating the parameters for one example can also affect predictions for other examples, so the model may need multiple epochs.

**Important:** More epochs do not always guarantee fewer errors.

If the data is linearly separable, the Perceptron can eventually converge to zero classification errors. If the data is not linearly separable, it may never reach zero errors.

---

# Perceptron on the Iris Dataset

## 1. Features vs. Classes

- **Features** are the input variables used for prediction.
- **Classes** are the possible output categories.

In the Iris dataset:

- Features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Classes:
  - Setosa
  - Versicolor
  - Virginica

The book uses only:

- **2 features** (sepal length and petal length) for easy 2D visualization.
- **2 classes** (Setosa and Versicolor) because the basic perceptron is a binary classifier.

> A perceptron is **not limited to two features**.

## 2. Preparing the Data

### Target Labels (`y`)

```python
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
```

This extracts the class labels and converts them into numbers:

- Setosa -> `0`
- Versicolor -> `1`

Shape:

```text
y.shape = (100,)
```

So `y` contains the correct class label for each sample.

### Input Features (`X`)

```python
X = df.iloc[0:100, [0, 2]].values
```

This extracts:

- 100 samples
- 2 features: sepal length and petal length

Shape:

```text
X.shape = (100, 2)
```

General machine learning convention:

```text
X.shape = (n_samples, n_features)
y.shape = (n_samples,)
```

## 3. Training the Perceptron

```python
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
```

During training:

```text
Correct prediction -> no parameter update
Wrong prediction   -> update weights and bias
```

If an entire epoch has **zero updates**, all training samples are classified correctly.

Therefore, the perceptron has **converged**.

## 4. Linear Separability

A basic perceptron can converge if the two classes are **linearly separable**.

```text
Linearly separable
        |
        v
A linear decision boundary exists
        |
        v
Perceptron can converge
```

If the classes are **not linearly separable**:

```text
Some samples remain misclassified
        |
        v
Weights and bias continue to update
        |
        v
The perceptron does not converge
```

This is why we usually set a maximum number of epochs.

## 5. Decision Boundary

For an input `x`, the perceptron calculates:

```text
z = w^T x + b
```

Then it applies a threshold:

```text
z > 0  -> Class 1
z <= 0 -> Class 0
```

The **decision boundary** is where:

```text
w^T x + b = 0
```

It separates the feature space into two classification regions.

Depending on the number of features:

```text
2 features -> line
3 features -> plane
n features -> hyperplane
```

## 6. Prediction Workflow

For a new sample:

```text
Input features x
        |
        v
Compute z = w^T x + b
        |
        v
Apply threshold
        |
        v
Predict Class 0 or Class 1
```

## Key Takeaways

- `X` contains the **input features**.
- `y` contains the **correct class labels**.
- `X.shape = (n_samples, n_features)`.
- A basic perceptron is a **binary classifier**.
- A perceptron can use more than two features.
- Two features are used here mainly for **2D visualization**.
- The perceptron updates its parameters when it makes a mistake.
- Zero updates in an epoch means the model has converged on the training data.
- Perceptron convergence requires **linear separability**.
- The decision boundary is defined by `w^T x + b = 0`.

---

# Adaline and Gradient Descent

## 1. Perceptron vs. Adaline

### Perceptron

The Perceptron uses the predicted class label to update its parameters.

```text
Input
  ↓
Net Input: z = w^T x + b
  ↓
Threshold Function
  ↓
Class Label
  ↓
Parameter Update
```

### Adaline

Adaline uses the continuous activation value to calculate the loss.

```text
Input
  ↓
Net Input: z = w^T x + b
  ↓
Linear Activation
  ↓
MSE Loss
  ↓
Gradient Descent
  ↓
Parameter Update
```

The activation function in Adaline is the identity function:

```text
σ(z) = z
```

The threshold function is only used to obtain the final class prediction.

### Key Difference

- **Perceptron:** updates parameters based on predicted class labels.
- **Adaline:** updates parameters based on continuous activation values.
- This allows Adaline to use a differentiable loss function and Gradient Descent.

## 2. Gradient Descent

Gradient Descent minimizes a loss function by updating the model parameters:

- weights `w`
- bias `b`

General update rule:

```text
w = w - η * gradient
```

where:

- `η` = learning rate
- `gradient` = direction of the steepest increase in loss
- `-gradient` = direction of the steepest decrease in loss

For Adaline, the loss function is **Mean Squared Error (MSE)**.

The goal is:

```text
Update w and b
     ↓
Reduce MSE
     ↓
Approach a minimum of the loss function
```

## 3. Learning Rate

The learning rate `η` controls the size of each parameter update.

### Learning Rate Too Large

- Updates are too large.
- The optimizer may overshoot the minimum.
- Loss may oscillate or diverge.

### Learning Rate Too Small

- Updates are very small.
- Training can converge very slowly.
- More epochs are required.

## 4. Epoch

One epoch means:

> **One complete pass through the entire training dataset.**

An epoch does **not** necessarily mean one parameter update.

### Batch Gradient Descent

Uses the entire training dataset to calculate the gradient.

```text
All Training Samples
        ↓
Calculate Gradient
        ↓
Update Parameters Once
```

Therefore:

```text
1 epoch = 1 parameter update
```

### Stochastic Gradient Descent (SGD)

Updates the parameters after each training sample.

```text
Sample 1 → Update
Sample 2 → Update
Sample 3 → Update
...
```

Therefore, if there are 100 samples:

```text
1 epoch = 100 parameter updates
```

## 5. Vectorized Weight Update in Adaline

Adaline uses the entire training set in Batch Gradient Descent.

```python
errors = y - output
self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
```

`errors` contains the prediction error for every training sample.

`X.T.dot(errors)` performs matrix-vector multiplication and calculates the update information for all weights at once.

Example shapes:

```text
X.shape       = (100, 2)
X.T.shape     = (2, 100)
errors.shape  = (100,)

X.T.dot(errors)
        ↓
shape = (2,)
        ↓
one value for each weight
```

This is an example of **vectorization**.

## 6. NumPy `dot()`

The behavior of `dot()` depends on the dimensions of its inputs.

| Input | Operation | Output |
|---|---|---|
| vector · vector | Dot product | Scalar |
| matrix · vector | Matrix-vector multiplication | Vector |
| matrix · matrix | Matrix multiplication | Matrix |

For example:

```python
X.T.dot(errors)
```

is **matrix-vector multiplication**.

In this case:

```text
(2, 100) dot (100,)
        ↓
       (2,)
```

The two output values correspond to the two weights.

## Key Takeaways

- Adaline uses continuous activation values to calculate MSE.
- Adaline's activation function is `σ(z) = z`.
- The threshold function is used for the final class prediction.
- Gradient Descent updates `w` and `b` to reduce the loss.
- A large learning rate may cause overshooting or divergence.
- A small learning rate causes slow convergence.
- One epoch means one complete pass through the training dataset.
- The number of parameter updates per epoch depends on the Gradient Descent method.
- Batch Gradient Descent updates once per epoch.
- SGD updates once per training sample.
- `X.T.dot(errors)` calculates weight-update information using vectorization.

---

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

## Key Takeaway

> **Feature scaling improves the optimization geometry, helping Gradient Descent find the minimum more efficiently.**

---

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

## 3. Why SGD Is Noisy

Different samples produce different gradient estimates.

Therefore, the optimization path and loss curve fluctuate more than with Batch GD.

### Advantage of Noise

Noise is not always bad.

Random fluctuations can sometimes help the optimizer escape:

- shallow local minima
- flat regions

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

## 6. Why Mini-Batch Is Common in Deep Learning

Mini-batch training combines the advantages of Batch GD and SGD:

- frequent parameter updates
- more stable gradients than one-sample SGD
- efficient matrix operations
- vectorization
- GPU parallelism

Therefore, modern neural networks are usually trained with **mini-batch SGD-family optimizers**.

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