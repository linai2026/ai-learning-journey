# Goodfellow Chapter 5 — Machine Learning Basics

## 1. Learning Algorithm

A machine learning algorithm learns from **experience \(E\)** to improve its performance on a **task \(T\)**, measured by a **performance measure \(P\)**.

\[
\boxed{\text{Machine Learning} = T + P + E}
\]

- \(T\): Task
- \(P\): Performance measure
- \(E\): Experience

---

## 2. Task \(T\)

A task describes **what the machine learning system should do**.

Examples:

- Classification
- Regression
- Machine translation
- Anomaly detection
- Denoising

Learning itself is **not** the task.

Example:

    Task: walking
    Learning: the process used to acquire the ability to walk

---

## 3. Common Machine Learning Tasks

### Classification

Predict a category.

\[
f:\mathbb{R}^n \rightarrow \{1,\dots,k\}
\]

Example:

    image → cat / dog

### Regression

Predict a numerical value.

\[
f:\mathbb{R}^n \rightarrow \mathbb{R}
\]

Example:

    house features → house price

Main difference:

    Classification → categorical output
    Regression     → numerical output

---

## 4. Performance Measure \(P\)

A performance measure is a **quantitative measure of how well the model performs the task**.

Examples:

    Classification → Accuracy / Error Rate
    Regression     → MSE

For classification:

\[
\text{Error Rate} = 1 - \text{Accuracy}
\]

The appropriate performance measure depends on the task.

---

## 5. Generalization

The goal is not only to perform well on training data, but also on **unseen data**.

    Training set
        ↓
    Learn parameters

    Test set
        ↓
    Evaluate generalization

The test set should not be used to learn model parameters.

Otherwise, the test data is no longer truly unseen and may cause **data leakage**.

---

## 6. Experience \(E\)

Experience means the information available to the learning algorithm during learning.

For many machine learning problems:

    Experience ≈ Training Data

However, experience is a broader concept than just a dataset.

### Supervised Learning

The model learns from inputs and labels:

\[
(X, y)
\]

    features + correct targets

### Unsupervised Learning

The model usually receives only:

\[
X
\]

and tries to discover patterns or structure in the data.

---

## 7. Design Matrix

A dataset is often represented by a design matrix:

\[
X \in \mathbb{R}^{m \times n}
\]

where:

- \(m\) = number of samples
- \(n\) = number of features

In NumPy:

    X.shape == (n_samples, n_features)

Example:

    X.shape == (200, 5)

means:

    200 samples
    5 features per sample

---

## 8. Matrix Element

\[
X_{ij}
\]

means:

    feature j of sample i

For example:

\[
X_{37,2}
\]

means:

    the 2nd feature of the 37th sample

### NumPy Indexing Difference

Mathematical indexing usually starts from 1, while Python indexing starts from 0.

Therefore:

    X[36, 1]

corresponds to:

\[
X_{37,2}
\]

---

## Key Summary

\[
\boxed{\text{Machine Learning} = T + P + E}
\]

    T → What should the model do?
    P → How do we measure performance?
    E → What does the model learn from?

Design matrix:

\[
\boxed{X \in \mathbb{R}^{n_{\text{samples}} \times n_{\text{features}}}}
\]

In NumPy:

    X.shape == (n_samples, n_features)