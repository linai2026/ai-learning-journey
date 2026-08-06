# Chapter 1 Review

What is Machine Learning?

Machine Learning is a method that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

⸻

What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses multi-layer neural networks to learn hierarchical representations of data.

⸻

Types of Machine Learning

Supervised Learning

* Uses labeled data
* Learns the mapping from inputs to outputs

Unsupervised Learning

* Uses unlabeled data
* Discovers hidden patterns or structures in data

Reinforcement Learning

* Learns by interacting with an environment
* Maximizes cumulative rewards through trial and error

⸻

Classification vs. Regression

Classification

* Predicts discrete categories
* Example: Cat or Dog

Regression

* Predicts continuous values
* Example: House Price

⸻

Clustering

Clustering groups similar data points into clusters without labels.

⸻

Dimensionality Reduction

Dimensionality Reduction reduces the number of features while preserving as much important information as possible.

Main purposes:

* Visualization
* Faster computation
* Noise reduction

---

## Training, Validation, and Testing

### Training
- Learn the model parameters from the training dataset.
- The goal is to minimize the loss by updating the model parameters (e.g., weights and biases).

### Validation
- Compare different models.
- Tune hyperparameters (e.g., learning rate, batch size, number of layers).
- Cross-validation is one common validation method.

### Testing
- Evaluate the final model on unseen data.
- The test set should only be used once after model selection.
- Do **not** use the test set for training or model tuning.

---

# Why Do We Need Data, Model, and Loss?

### Data
- Provides patterns for the model to learn.

### Model
- Represents the relationship between the input and the output.

### Loss
- Measures how wrong the model's predictions are.
- Guides the model to improve during training.

**Summary**

Data → Provides knowledge

Model → Learns the knowledge

Loss → Tells the model how wrong it is

---

# Why Python?

- Simple and easy to read.
- Large and active ecosystem.
- Most heavy computations are performed by optimized C/C++/CUDA libraries.

---

# Scikit-Learn

- A Python library for traditional machine learning.
- Provides algorithms for:
  - Classification
  - Regression
  - Clustering
  - Dimensionality Reduction
  - Data Preprocessing
  - Model Selection

---

# PyTorch

- A deep learning framework.
- Provides:
  - Tensor operations
  - Automatic differentiation (Autograd)
  - Neural network modules
  - GPU acceleration

---

# Machine Learning Pipeline

```text
Raw Data
    ↓
Preprocessing
    ↓
Training
    ↓
Validation
    ↓
Testing
    ↓
Prediction
```

### Preprocessing
- Clean the data.
- Handle missing values.
- Scale numerical features.
- Encode categorical features.
- Create useful features (Feature Engineering).

---

# Why Not Train on the Test Set?

- The test set represents unseen data.
- Using it for training or model tuning causes **Data Leakage**.
- The test accuracy becomes overly optimistic and no longer reflects real-world performance.

---

# Validation vs. Test Set

**Validation Set**
- Used for model selection.
- Used for hyperparameter tuning.

**Test Set**
- Used only for the final evaluation.
- Must remain completely unseen during model development.

---

# Problems with Raw Data

Without preprocessing, the model may face:

- Different feature scales.
- Missing values.
- Categorical data that cannot be processed directly.
- Noisy or duplicated data.
- Poor feature representation.
