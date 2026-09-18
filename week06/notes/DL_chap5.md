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