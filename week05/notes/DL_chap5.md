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


---

# Goodfellow Chapter 5 — Machine Learning Basics

## 5.1.4 Linear Regression

### Definition

Linear regression predicts a scalar output from an input vector.

### Model

\[
\hat{y} = w^T x + b
\]

- \(x\): input features
- \(w\): weights
- \(b\): bias
- \(\hat{y}\): prediction

### Parameters

The model learns:

- weights \(w\)
- bias \(b\)

Each weight controls how strongly a feature affects the prediction.

### Training Objective

Linear regression minimizes the training error.

A common loss is Mean Squared Error:

\[
MSE =
\frac{1}{m}
\sum_{i=1}^{m}
(\hat{y}^{(i)} - y^{(i)})^2
\]

The goal is to find \(w\) and \(b\) that minimize the training MSE.

---

## 5.2 Capacity, Overfitting, and Underfitting

### Generalization

Generalization is the ability of a model to perform well on new, unseen data.

### Training Error

Training error measures performance on the training data.

### Generalization Error

Generalization error measures expected performance on unseen data.

In practice, it is estimated using held-out data.

### Main Goal

A good machine learning model should:

- have low training error
- have a small gap between training error and test error

The main goal is good generalization, not only low training error.

---

## Underfitting

### Definition

Underfitting occurs when the model cannot fit the training data well.

### Typical Pattern

- training error: high
- test error: high

### Main Cause

- model capacity is too low

---

## Overfitting

### Definition

Overfitting occurs when the model fits the training data very well but performs poorly on unseen data.

### Typical Pattern

- training error: very low
- test error: relatively high
- generalization gap: large

### Generalization Gap

\[
\text{Generalization Gap}
=
\text{Test Error}
-
\text{Training Error}
\]

A large generalization gap often indicates overfitting.

---

## Model Capacity

### Definition

Capacity is the ability of a model to represent a wide variety of functions.

Higher capacity allows the model to represent more complex functions.

### Example

Degree-1 polynomial:

\[
\hat{y}
=
b + w_1x
\]

Degree-2 polynomial:

\[
\hat{y}
=
b + w_1x + w_2x^2
\]

Degree-9 polynomial:

\[
\hat{y}
=
b + \sum_{i=1}^{9} w_i x^i
\]

### Capacity and Fitting

- low capacity → underfitting
- appropriate capacity → good generalization
- very high capacity → higher risk of overfitting

High capacity does not always cause overfitting.

Overfitting also depends on:

- amount of training data
- task complexity
- noise
- regularization

---

## Capacity and Error

### Training Error

As capacity increases, training error usually decreases.

\[
\text{Capacity} \uparrow
\Rightarrow
\text{Training Error} \downarrow
\]

### Generalization Error

Generalization error usually follows a U-shaped curve.

- low capacity → high generalization error
- appropriate capacity → lowest generalization error
- excessive capacity → generalization error increases

### Key Idea

The best model is not necessarily the model with the lowest training error.

The goal is to minimize generalization error.

---

## Representational Capacity

### Definition

Representational capacity describes the set of functions that a model family can theoretically represent.

---

## Effective Capacity

### Definition

Effective capacity describes how much of the model's representational capacity the learning algorithm can actually use.

### Key Idea

Effective capacity can be lower than representational capacity because optimization may not find the best possible function.

---

## Parametric Models

### Definition

A parametric model has a fixed-size parameter vector before seeing the training data.

### Example

- linear regression

If the number of features is fixed, the number of model parameters is also fixed.

---

## Nonparametric Models

### Definition

A nonparametric model does not require model complexity to be fixed before seeing the training data.

Its complexity may grow with the amount of training data.

### Example

- nearest neighbor methods

### Important

Nonparametric does not mean parameter-free.

The key difference is whether model complexity is limited by a fixed-size parameter vector.

---

## Training Data and Capacity

### Key Relationship

More training data usually allows the use of higher-capacity models.

\[
\text{More Training Data}
\Rightarrow
\text{Lower Overfitting Risk}
\Rightarrow
\text{Higher Capacity Can Often Be Used}
\]

### Intuition

With a small dataset, a high-capacity model can easily fit accidental patterns or noise.

With more data, this becomes harder.

---

## Bayes Error

### Definition

Bayes error is the minimum possible prediction error caused by inherent uncertainty or noise in the data-generating process.

### Key Idea

Even an ideal model may not achieve zero error.

Bayes error is also called irreducible error.

---

## 5.2.1 No Free Lunch Theorem

### Definition

The No Free Lunch theorem states that no machine learning algorithm is universally best over all possible data-generating distributions.

### Important

It does not mean that all algorithms perform equally well on real-world problems.

### Key Idea

Real-world data has structure.

Therefore, suitable assumptions and model preferences can improve performance on specific tasks.

### Main Relationship

- no universally best algorithm
- real-world problems have structure
- algorithms use assumptions and preferences
- suitable algorithms perform better on suitable tasks

---

## 5.2.2 Regularization

### Definition

Regularization modifies the learning algorithm in order to improve generalization.

### Main Goal

The goal of regularization is to reduce generalization error.

It does not necessarily reduce training error.

### Important

Regularization may:

- increase training error
- decrease generalization error
- improve performance on unseen data

---

## L2 Regularization

### Objective Function

\[
J(w)
=
MSE_{\text{train}}
+
\lambda \|w\|_2^2
\]

where:

\[
\|w\|_2^2
=
w^Tw
=
\sum_i w_i^2
\]

### Training Error Term

\[
MSE_{\text{train}}
\]

This term encourages the model to fit the training data.

### Regularization Term

\[
\lambda \|w\|_2^2
\]

This term penalizes large weights.

### Main Trade-Off

The model must balance:

- fitting the training data
- keeping weights small

---

## Effect of Lambda

### Definition

\(\lambda\) controls the strength of regularization.

### Small Lambda

When \(\lambda\) is small:

- regularization is weak
- larger weights are allowed
- effective capacity is higher
- overfitting risk is higher

### Large Lambda

When \(\lambda\) is large:

- regularization is strong
- large weights are penalized more
- weights tend to become smaller
- effective capacity decreases
- overfitting risk decreases

### Lambda Too Large

If \(\lambda\) is too large:

- the model becomes too simple
- underfitting may occur

### Main Relationship

\[
\lambda \uparrow
\Rightarrow
\|w\| \downarrow
\Rightarrow
\text{Effective Capacity} \downarrow
\Rightarrow
\text{Overfitting Risk} \downarrow
\]

---

## 5.3 Hyperparameters and Validation Sets

## Parameters

### Definition

Parameters are learned from training data by the training algorithm.

### Examples

- weights \(w\)
- bias \(b\)

### Main Relationship

\[
\text{Training Data}
\Rightarrow
\text{Learn Parameters}
\]

---

## Hyperparameters

### Definition

Hyperparameters control the behavior of the model or learning algorithm.

They are selected outside the basic parameter-learning process.

### Examples

- learning rate
- regularization strength \(\lambda\)
- SVM \(C\)
- polynomial degree
- number of neighbors \(k\)
- model architecture

### Main Relationship

\[
\text{Validation Data}
\Rightarrow
\text{Select Hyperparameters}
\]

---

## Parameters vs Hyperparameters

| Type | Examples | How They Are Determined |
|---|---|---|
| Parameters | \(w, b\) | Learned from training data |
| Hyperparameters | \(\lambda, C, k\), learning rate | Selected using validation performance |

---

## Why Hyperparameters Need Validation Data

### Problem

Hyperparameters that control model capacity should not be selected only by minimizing training error.

If training error alone is used:

- higher capacity often gives lower training error
- the model may prefer maximum capacity
- overfitting may occur

### Solution

Use a separate validation set to select hyperparameters.

---

## Training Set

### Purpose

The training set is used to learn model parameters.

### Examples

- \(w\)
- \(b\)

### Main Relationship

\[
\text{Training Set}
\Rightarrow
\text{Parameters}
\]

---

## Validation Set

### Purpose

The validation set is used to:

- select hyperparameters
- compare models
- perform model selection

### Examples

- \(\lambda\)
- \(C\)
- \(k\)
- learning rate
- model architecture

### Main Relationship

\[
\text{Validation Set}
\Rightarrow
\text{Hyperparameters}
\]

---

## Test Set

### Purpose

The test set is used to estimate final generalization performance.

### Important

The test set should not be used to:

- learn parameters
- select hyperparameters
- choose model architecture
- repeatedly modify the model

### Main Relationship

\[
\text{Test Set}
\Rightarrow
\text{Final Evaluation}
\]

---

## Why the Test Set Must Stay Unseen

### Problem

If test performance is repeatedly checked while modifying the model, information from the test set enters the model-development process.

### Result

The model may overfit to the test set.

### Consequence

The reported test performance may become optimistically biased.

### Key Rule

Use the test set only for final evaluation.

---

## Validation Error

### Important

The validation set is used to select hyperparameters.

Therefore, the model-development process indirectly adapts to the validation set.

### Consequence

Validation performance may also become slightly optimistic.

### Solution

Use a separate untouched test set for final evaluation.

---

## 5.3.1 Cross-Validation

### Motivation

Cross-validation is especially useful when the dataset is small.

A single train-validation split may:

- waste limited data
- produce an unstable estimate
- depend too much on one random split

---

## k-Fold Cross-Validation

### Procedure

The dataset is divided into \(k\) non-overlapping folds.

For each trial:

1. Use one fold as the held-out set.
2. Use the remaining \(k-1\) folds for training.
3. Train the model.
4. Evaluate the model on the held-out fold.

Repeat this process \(k\) times.

---

## 5-Fold Cross-Validation

### Number of Trials

\[
k = 5
\]

The model is trained 5 times.

### Data Used in Each Trial

Training data:

\[
\frac{4}{5}
=
80\%
\]

Held-out data:

\[
\frac{1}{5}
=
20\%
\]

### Example

- Trial 1 → Fold 1 held out, Folds 2–5 used for training
- Trial 2 → Fold 2 held out, Folds 1, 3, 4, 5 used for training
- Trial 3 → Fold 3 held out
- Trial 4 → Fold 4 held out
- Trial 5 → Fold 5 held out

---

## Cross-Validation Error

### Formula

\[
CV_{\text{error}}
=
\frac{1}{k}
\sum_{i=1}^{k}
Error_i
\]

### Key Idea

Each example gets a chance to appear in the held-out fold.

---

## Advantages of Cross-Validation

Cross-validation:

- makes better use of small datasets
- reduces dependence on one train-validation split
- provides a more stable performance estimate
- allows every example to participate in evaluation

---

## Core Relationships

\[
\text{Low Capacity}
\Rightarrow
\text{Underfitting}
\]

\[
\text{High Capacity}
\Rightarrow
\text{Higher Overfitting Risk}
\]

\[
\text{More Training Data}
\Rightarrow
\text{Lower Overfitting Risk}
\]

\[
\lambda \uparrow
\Rightarrow
\text{Effective Capacity} \downarrow
\]

\[
\text{Training Set}
\Rightarrow
\text{Learn Parameters}
\]

\[
\text{Validation Set}
\Rightarrow
\text{Select Hyperparameters}
\]

\[
\text{Test Set}
\Rightarrow
\text{Estimate Final Generalization Performance}
\]

\[
\text{Small Dataset}
\Rightarrow
\text{Cross-Validation}
\]

---

## Key Takeaways

- Low training error does not guarantee good generalization.
- Underfitting occurs when model capacity is too low.
- Overfitting occurs when training performance is good but generalization is poor.
- Capacity describes the complexity of functions a model can represent.
- More training data usually allows higher-capacity models.
- Nonparametric does not mean parameter-free.
- Bayes error is irreducible error.
- No machine learning algorithm is universally best.
- Regularization can improve generalization by reducing effective capacity.
- Larger \(\lambda\) usually means stronger regularization.
- Parameters are learned from training data.
- Hyperparameters are selected using validation data.
- The test set should be reserved for final evaluation.
- Repeatedly using the test set can cause test-set overfitting.
- Cross-validation is especially useful for small datasets.