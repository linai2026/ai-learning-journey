# Chapter 2 — Perceptron: Core Concepts

## 1. Features, Labels, and Predictions

- **Feature (`x`)**: an input variable used by the model to make a prediction.
- **Label (`y`)**: the true target associated with a training sample.
- **Prediction (`ŷ`)**: the output produced by the model for an input.

Key distinction:

- $y$ = true label
- $\hat{y}$ = model prediction

---

## 2. What Does a Perceptron Learn?

A perceptron learns its **parameters**:

- weights (`w`)
- bias (`b`)

Training means adjusting these parameters based on prediction errors.

---

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

---

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

---

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

---

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

---

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

## Key Mental Model

### Forward Prediction

$$
\text{features } x
\rightarrow
\text{parameters } (w,b)
\rightarrow
\text{net input } z
\rightarrow
\text{decision function}
\rightarrow
\text{prediction } \hat{y}
$$

### Learning

$$
\hat{y}
\rightarrow
\text{compare with } y
\rightarrow
\text{error}
\rightarrow
\text{update } w,b
$$