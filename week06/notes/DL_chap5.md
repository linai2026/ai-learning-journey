# Chapter 5.4 — Estimators, Bias and Variance

## 1. Point Estimation

Point estimation uses sample data to estimate an unknown quantity.

The true parameter is denoted by:

$$
\theta
$$

An estimate of the parameter is denoted by:

$$
\hat{\theta}
$$

A point estimator is a function of the sample data:

$$
\hat{\theta}_m = g(x^{(1)}, \ldots, x^{(m)})
$$

Key distinction:

- **Estimator**: the rule or function used to calculate an estimate.
- **Estimate**: the specific value produced from a particular dataset.

Because the training data is randomly sampled, different datasets can produce different estimates.

Therefore, $\hat{\theta}$ can be treated as a random variable.

---

## 2. Function Estimation

Point estimation can also estimate an entire function.

Suppose the true relationship is:

$$
y = f(x) + \epsilon
$$

Machine learning uses training data to learn an approximation:

$$
\hat{f}(x) \approx f(x)
$$

Therefore:

```text
Training data
→ Learning algorithm
→ Estimated function f̂(x)
```

---

## 3. Bias

Bias measures whether an estimator systematically deviates from the true parameter.

$$
\operatorname{Bias}(\hat{\theta})
=
E[\hat{\theta}] - \theta
$$

Here, $E[\hat{\theta}]$ means the average estimate over repeated sampling.

An estimator is **unbiased** if:

$$
E[\hat{\theta}] = \theta
$$

Therefore:

$$
\operatorname{Bias}(\hat{\theta}) = 0
$$

Unbiased does **not** mean every estimate equals the true value.

It means:

```text
Average estimate over repeated sampling
=
True parameter
```

Core intuition:

```text
Bias = systematic error
Bias = "Is the estimator systematically off target?"
```

---

## 4. Sample Mean as an Unbiased Estimator

The sample mean is:

$$
\hat{\mu}
=
\frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

If:

$$
E[x^{(i)}] = \mu
$$

then:

$$
E[\hat{\mu}]
=
\frac{1}{m}\sum_{i=1}^{m}E[x^{(i)}]
=
\mu
$$

Therefore:

$$
\operatorname{Bias}(\hat{\mu}) = 0
$$

The sample mean is an unbiased estimator of the population mean.

---

## 5. Sample Variance

Consider:

$$
\hat{\sigma}_m^2
=
\frac{1}{m}
\sum_{i=1}^{m}
(x^{(i)}-\hat{\mu})^2
$$

Its expectation is:

$$
E[\hat{\sigma}_m^2]
=
\frac{m-1}{m}\sigma^2
$$

Therefore, dividing by $m$ systematically underestimates the true population variance.

Using $m-1$ instead:

$$
\tilde{\sigma}_m^2
=
\frac{1}{m-1}
\sum_{i=1}^{m}
(x^{(i)}-\hat{\mu})^2
$$

gives:

$$
E[\tilde{\sigma}_m^2] = \sigma^2
$$

Therefore:

```text
Divide by m
→ biased downward

Divide by m - 1
→ unbiased estimator of population variance
```

In NumPy:

```python
np.var(x, ddof=0)  # divide by m
np.var(x, ddof=1)  # divide by m - 1
```

---

## 6. Variance of an Estimator

Variance measures how much an estimator changes when the training dataset changes.

$$
\operatorname{Var}(\hat{\theta})
$$

Low variance:

```text
Different training sets
→ similar estimates
→ stable estimator
```

High variance:

```text
Different training sets
→ very different estimates
→ unstable estimator
```

Core intuition:

```text
Variance = instability
Variance = "How sensitive is the estimator to the training data?"
```

---

## 7. Standard Error

Standard error is the square root of the variance of an estimator.

$$
SE(\hat{\theta})
=
\sqrt{\operatorname{Var}(\hat{\theta})}
$$

For the sample mean:

$$
SE(\hat{\mu})
=
\frac{\sigma}{\sqrt{m}}
$$

Therefore:

$$
m \uparrow
\quad\Rightarrow\quad
SE(\hat{\mu}) \downarrow
$$

More data makes the sample mean more stable.

Because:

$$
SE \propto \frac{1}{\sqrt{m}}
$$

if the sample size increases by a factor of 4:

$$
m \times 4
\quad\Rightarrow\quad
SE \times \frac{1}{2}
$$

---

## 8. Confidence Interval

Under appropriate assumptions, an approximate 95% confidence interval for the mean is:

$$
\hat{\mu}
\pm
1.96SE(\hat{\mu})
$$

Smaller standard error generally produces a narrower confidence interval.

---

## 9. Bias vs. Variance

Bias and variance measure different sources of error.

| Concept | Meaning |
|---|---|
| Bias | Systematic deviation from the true value |
| Variance | Sensitivity to different training datasets |

A useful mental model is:

```text
Bias
= "Is it systematically off target?"

Variance
= "Is it stable when the training data changes?"
```

An unbiased estimator is not necessarily a good estimator.

For example:

```text
Bias = 0
Variance = very large
→ estimator can still have large error
```

---

## 10. Mean Squared Error

Mean squared error is:

$$
MSE
=
E[(\hat{\theta}-\theta)^2]
$$

For an estimator of a fixed parameter:

$$
\boxed{
MSE
=
\operatorname{Bias}(\hat{\theta})^2
+
\operatorname{Var}(\hat{\theta})
}
$$

Core intuition:

```text
MSE
=
systematic error²
+
instability
```

A good estimator should generally keep both bias and variance reasonably small.

---

## 11. Bias-Variance Trade-Off

Bias and variance are closely related to model capacity.

As model capacity increases, a common tendency is:

```text
Bias ↓
Variance ↑
```

A simple model often has:

```text
High bias
Low variance
→ tendency toward underfitting
```

A complex model often has:

```text
Low bias
High variance
→ tendency toward overfitting
```

These are common tendencies, not absolute rules.

The goal is to choose model capacity that gives good generalization.

---

## 12. Connection to Underfitting and Overfitting

### Underfitting

The model is too simple to capture the underlying relationship.

Typical pattern:

```text
High bias
+
Low variance
```

### Overfitting

The model adapts too strongly to the training data, including noise.

Typical pattern:

```text
Low bias
+
High variance
```

A complex model can be more sensitive to changes in the training data, leading to higher variance.

---

## 13. Consistency

Consistency describes what happens as the amount of training data increases.

An estimator is consistent if:

$$
m \rightarrow \infty
$$

causes the estimator to approach the true parameter:

$$
\hat{\theta}_m \rightarrow \theta
$$

Core intuition:

```text
More and more data
→ estimate gets closer to the true parameter
```

Consistency asks:

```text
If training data keeps increasing,
does the estimator eventually approach the true parameter?
```

Unbiasedness and consistency are different concepts.

```text
Unbiasedness
→ Is the estimator correct on average?

Consistency
→ Does the estimator approach the true value as data → ∞?
```

An estimator can be unbiased without being consistent.

---

## 14. Key Takeaways

```text
θ
= true unknown parameter

θ̂
= estimate from sample data

Estimator
= rule/function that produces an estimate

Bias
= systematic error
= "Is the estimator systematically off target?"

Variance
= instability
= "Is the estimator stable across different training sets?"

Standard Error
= sqrt(Variance)

More data
→ usually lower variance / standard error
→ more stable estimates

MSE
= Bias² + Variance

Simple model
→ often higher bias
→ underfitting tendency

Complex model
→ often higher variance
→ overfitting tendency

Consistency
= estimator approaches the true parameter as data → ∞
```

## 15. Core Relationships

$$
\operatorname{Bias}(\hat{\theta})
=
E[\hat{\theta}] - \theta
$$

$$
SE(\hat{\theta})
=
\sqrt{\operatorname{Var}(\hat{\theta})}
$$

$$
SE(\hat{\mu})
=
\frac{\sigma}{\sqrt{m}}
$$

$$
MSE
=
\operatorname{Bias}(\hat{\theta})^2
+
\operatorname{Var}(\hat{\theta})
$$

```text
Model Capacity ↑
        ↓
Bias tends to ↓
Variance tends to ↑

Low capacity
→ high-bias / underfitting tendency

High capacity
→ high-variance / overfitting tendency

Goal
→ balance bias and variance
→ good generalization
```


---


# Maximum Likelihood Estimation

## 1. Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation chooses the model parameters that make the observed training data as likely as possible.

Given training data:

$begin:math:display$
X \= \\\{x\^\{\(1\)\}\, x\^\{\(2\)\}\, \\dots\, x\^\{\(m\)\}\\\}
$end:math:display$

the maximum likelihood estimator is:

$begin:math:display$
\\theta\_\{ML\}
\=
\\arg\\max\_\{\\theta\} p\_\{\\text\{model\}\}\(X\;\\theta\)
$end:math:display$

The observed data $begin:math:text$X$end:math:text$ is fixed, while the parameters $begin:math:text$\\theta$end:math:text$ are optimized.

### Key Idea

$begin:math:display$
\\boxed\{
\\text\{Choose parameters that make the observed data most plausible\}
\}
$end:math:display$

---

## 2. Likelihood for i.i.d. Data

If the training examples are independent and identically distributed (i.i.d.):

$begin:math:display$
p\(X\;\\theta\)
\=
\\prod\_\{i\=1\}\^\{m\}
p\(x\^\{\(i\)\}\;\\theta\)
$end:math:display$

Therefore:

$begin:math:display$
\\theta\_\{ML\}
\=
\\arg\\max\_\{\\theta\}
\\prod\_\{i\=1\}\^\{m\}
p\(x\^\{\(i\)\}\;\\theta\)
$end:math:display$

---

## 3. Log-Likelihood

Instead of maximizing the product of probabilities, we usually maximize the log-likelihood:

$begin:math:display$
\\theta\_\{ML\}
\=
\\arg\\max\_\{\\theta\}
\\sum\_\{i\=1\}\^\{m\}
\\log p\(x\^\{\(i\)\}\;\\theta\)
$end:math:display$

This works because:

$begin:math:display$
\\log\(ab\)\=\\log a\+\\log b
$end:math:display$

so products become sums.

### Why Use Log-Likelihood?

- It prevents numerical underflow when multiplying many small probabilities.
- Sums are easier to compute and differentiate than products.
- The logarithm is strictly increasing, so it does not change the location of the maximum.

Therefore:

$begin:math:display$
\\arg\\max\_\{\\theta\} L\(\\theta\)
\=
\\arg\\max\_\{\\theta\} \\log L\(\\theta\)
$end:math:display$

---

## 4. Probability vs. Likelihood

The mathematical expression may be the same, but the viewpoint is different.

### Probability

Fix the parameters and consider different possible data:

$begin:math:display$
p\(x\;\\theta\)
$end:math:display$

### Likelihood

Fix the observed data and compare different parameters:

$begin:math:display$
L\(\\theta\;X\)
$end:math:display$

In MLE:

$begin:math:display$
\\boxed\{
X \\text\{ is fixed\, while \} \\theta \\text\{ changes\}
\}
$end:math:display$

---

## 5. MLE and KL Divergence

Ideally, we want the model distribution to match the true data distribution:

$begin:math:display$
p\_\{\\text\{model\}\}
\\approx
p\_\{\\text\{data\}\}
$end:math:display$

The training data defines an empirical distribution:

$begin:math:display$
\\hat p\_\{\\text\{data\}\}
$end:math:display$

The difference between the empirical distribution and the model distribution can be measured using KL divergence:

$begin:math:display$
D\_\{KL\}
\(
\\hat p\_\{\\text\{data\}\}
\\Vert
p\_\{\\text\{model\}\}
\)
$end:math:display$

Maximum likelihood estimation can be interpreted as minimizing this difference:

$begin:math:display$
\\boxed\{
\\text\{maximize likelihood\}
\\Longleftrightarrow
\\text\{minimize KL divergence\}
\}
$end:math:display$

---

## 6. Negative Log-Likelihood

Maximizing log-likelihood is equivalent to minimizing negative log-likelihood (NLL):

$begin:math:display$
\\max\_\{\\theta\} \\log L\(\\theta\)
\\Longleftrightarrow
\\min\_\{\\theta\} \[\-\\log L\(\\theta\)\]
$end:math:display$

Therefore:

$begin:math:display$
\\boxed\{
\\text\{MLE\}
\\rightarrow
\\text\{maximize log\-likelihood\}
\\rightarrow
\\text\{minimize NLL\}
\}
$end:math:display$

This connects probability modeling directly to loss minimization in machine learning.

---

## 7. MLE and Cross-Entropy

Minimizing negative log-likelihood is closely related to minimizing cross-entropy.

For binary classification with a Bernoulli distribution:

$begin:math:display$
\\boxed\{
\\text\{Binary Cross\-Entropy\}
\=
\\text\{Negative Log\-Likelihood\}
\}
$end:math:display$

Therefore, for Logistic Regression:

$begin:math:display$
\\text\{maximize likelihood\}
\\Longleftrightarrow
\\text\{minimize BCE\}
$end:math:display$

This explains why Binary Cross-Entropy is a natural loss function for binary Logistic Regression.

---

## 8. Conditional Maximum Likelihood

In supervised learning, we want to predict $begin:math:text$Y$end:math:text$ from $begin:math:text$X$end:math:text$.

Therefore, we model the conditional probability:

$begin:math:display$
P\(Y\|X\;\\theta\)
$end:math:display$

The conditional maximum likelihood estimator is:

$begin:math:display$
\\theta\_\{ML\}
\=
\\arg\\max\_\{\\theta\}
P\(Y\|X\;\\theta\)
$end:math:display$

For i.i.d. training examples:

$begin:math:display$
\\theta\_\{ML\}
\=
\\arg\\max\_\{\\theta\}
\\sum\_\{i\=1\}\^\{m\}
\\log
P\(y\^\{\(i\)\}\|x\^\{\(i\)\}\;\\theta\)
$end:math:display$

### Key Idea

$begin:math:display$
\\boxed\{
X \\text\{ is observed\}
\\rightarrow
\\text\{model the distribution of \} Y\|X
\}
$end:math:display$

---

## 9. Linear Regression as Maximum Likelihood

Linear Regression predicts:

$begin:math:display$
\\hat y \= Xw\+b
$end:math:display$

Assume that the target $begin:math:text$y$end:math:text$, given $begin:math:text$x$end:math:text$, follows a Gaussian distribution:

$begin:math:display$
p\(y\|x\)
\=
\\mathcal\{N\}\(y\;\\hat y\,\\sigma\^2\)
$end:math:display$

The mean of this Gaussian distribution is the model prediction:

$begin:math:display$
\\hat y
$end:math:display$

Under this Gaussian assumption, maximizing the conditional log-likelihood is equivalent to minimizing the squared prediction error:

$begin:math:display$
\\sum\_\{i\=1\}\^\{m\}
\\\|y\^\{\(i\)\}\-\\hat y\^\{\(i\)\}\\\|\^2
$end:math:display$

The Mean Squared Error is:

$begin:math:display$
MSE
\=
\\frac\{1\}\{m\}
\\sum\_\{i\=1\}\^\{m\}
\\\|y\^\{\(i\)\}\-\\hat y\^\{\(i\)\}\\\|\^2
$end:math:display$

Therefore:

$begin:math:display$
\\boxed\{
\\text\{Gaussian assumption \+ MLE\}
\\Longleftrightarrow
\\text\{MSE minimization\}
\}
$end:math:display$

This provides a probabilistic justification for using MSE in Linear Regression.

---

## 10. Properties of Maximum Likelihood

### Consistency

Under appropriate conditions, as the number of training examples increases:

$begin:math:display$
m \\rightarrow \\infty
$end:math:display$

the maximum likelihood estimate approaches the true parameter:

$begin:math:display$
\\hat\\theta\_\{ML\}
\\rightarrow
\\theta\_\{\\text\{true\}\}
$end:math:display$

Therefore:

$begin:math:display$
\\boxed\{
\\text\{More data\}
\\rightarrow
\\text\{MLE approaches the true parameter\}
\}
$end:math:display$

### Statistical Efficiency

Different consistent estimators may require different amounts of data to achieve the same estimation accuracy.

A statistically efficient estimator can estimate the true parameters accurately using fewer examples.

Maximum likelihood estimators have desirable asymptotic efficiency properties under appropriate conditions.

---

## 11. Core Connections

### Logistic Regression

$begin:math:display$
X
\\rightarrow
z\=Xw\+b
\\rightarrow
\\sigma\(z\)
\\rightarrow
P\(Y\|X\)
\\rightarrow
BCE
$end:math:display$

From the MLE perspective:

$begin:math:display$
\\boxed\{
\\text\{Bernoulli likelihood\}
\\rightarrow
\\text\{Negative Log\-Likelihood\}
\\rightarrow
\\text\{BCE\}
\}
$end:math:display$

### Linear Regression

$begin:math:display$
X
\\rightarrow
\\hat y\=Xw\+b
\\rightarrow
MSE
$end:math:display$

From the MLE perspective:

$begin:math:display$
\\boxed\{
\\text\{Gaussian likelihood\}
\\rightarrow
\\text\{Negative Log\-Likelihood\}
\\rightarrow
\\text\{MSE\}
\}
$end:math:display$

---

## 12. Final Summary

The central idea of Maximum Likelihood Estimation is:

$begin:math:display$
\\boxed\{
\\text\{Choose parameters that make the observed data most likely\}
\}
$end:math:display$

The main optimization relationship is:

$begin:math:display$
\\boxed\{
\\text\{maximize likelihood\}
\\Longleftrightarrow
\\text\{maximize log\-likelihood\}
\\Longleftrightarrow
\\text\{minimize negative log\-likelihood\}
\}
$end:math:display$

Important machine learning connections:

$begin:math:display$
\\boxed\{
\\text\{Logistic Regression \+ Bernoulli\}
\\rightarrow
\\text\{BCE\}
\}
$end:math:display$

$begin:math:display$
\\boxed\{
\\text\{Linear Regression \+ Gaussian\}
\\rightarrow
\\text\{MSE\}
\}
$end:math:display$

MLE therefore provides a probabilistic foundation for many common machine learning loss functions.