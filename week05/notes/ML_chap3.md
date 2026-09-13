Scikit-Learn Workflow — Chapter 3 Notes
1. Train-Test Split
The dataset is divided into:
* Training set → used to learn model parameters
* Test set → used to evaluate performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)
Important Parameters
test_size=0.3
* 30% test data
* 70% training data
random_state=1
* Makes the random split reproducible.
stratify=y
* Preserves the class distribution in the training and test sets.
 
⸻
 
2. Feature Scaling with 
StandardScaler
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
fit()
StandardScaler.fit() learns the preprocessing parameters for each feature:
\mu_j = \text{mean}
\sigma_j = \text{standard deviation}
transform()
The data is standardized using:
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
The same \mu and \sigma learned from the training set must also be used for the test set.
X_train
   ↓
fit scaler
   ↓
learn μ and σ
   ↓
transform X_train
   ↓
transform X_test using the same μ and σ
Do not fit the scaler on the test set because this causes data leakage.
 
⸻
 
3. Scikit-Learn Model Workflow
A typical scikit-learn workflow is:
create model
     ↓
fit
     ↓
learn parameters
     ↓
predict
     ↓
evaluate
Example:
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
 
⸻
 
4. 
fit()
vs 
predict()
Training
ppn.fit(X_train_std, y_train)
Requires:
* X_train → input features
* y_train → correct labels
The model uses them to learn its parameters.
For a linear model, these include:
w,\quad b
 
⸻
 
Prediction
y_pred = ppn.predict(X_test_std)
Only requires:
X_test
because the model parameters have already been learned.
Conceptually:
X_{\text{test}} \rightarrow \hat y
 
⸻
 
5. Model Evaluation
After prediction:
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
Here:
y_test = true labels
y_pred = predicted labels
Accuracy is:
\text{Accuracy} = \frac{\text{correct predictions}} {\text{total predictions}}
For example:
\frac{44}{45} \approx 0.978
so:
Accuracy = 97.8%
A classifier can also use:
ppn.score(X_test_std, y_test)
 
⸻
 
6. Complete Machine Learning Pipeline
raw data
   ↓
train_test_split
   ↓
X_train / X_test
   ↓
fit preprocessing on X_train
   ↓
transform X_train and X_test
   ↓
model.fit(X_train, y_train)
   ↓
model.predict(X_test)
   ↓
y_pred
   ↓
compare y_pred with y_test
   ↓
evaluate generalization performance
 
⸻
 
7. Perceptron Limitation
The perceptron converges only when the training data is linearly separable.
non-linearly-separable data
        ↓
misclassified samples remain
        ↓
weights keep changing
        ↓
Perceptron may not converge
This motivates moving to more powerful linear classifiers such as Logistic Regression.
 
⸻
 
Key Takeaways
Training set → learn
Test set → evaluate

StandardScaler.fit()
→ learns μ and σ

model.fit()
→ learns model parameters such as w and b

predict()
→ produces predictions

y_test
→ only used for evaluation

stratify=y
→ preserves class proportions

Never learn preprocessing or model parameters from the test set.



# Logistic Regression — First Understanding

## 1. Why Logistic Regression?

Perceptron may not converge when the classes are not perfectly linearly separable.

Logistic Regression:

- is a classification model
- can output class probabilities
- learns model parameters by minimizing a loss function

---

## 2. Net Input

The linear part is:

\[
z = w^T x + b
\]

where:

- \(x\): input features
- \(w\): weights
- \(b\): bias
- \(z\): net input

---

## 3. Logit

For probability \(p\), the odds are:

\[
\text{odds} = \frac{p}{1-p}
\]

The logit function is:

\[
\text{logit}(p)
=
\log\frac{p}{1-p}
\]

Logistic Regression assumes:

\[
\text{logit}(p)=w^Tx+b=z
\]

---

## 4. Sigmoid Function

The sigmoid function is:

\[
\sigma(z)
=
\frac{1}{1+e^{-z}}
\]

It converts the net input \(z\) into a value between 0 and 1:

\[
0 < \sigma(z) < 1
\]

Important values:

\[
z\to-\infty
\Rightarrow
\sigma(z)\to0
\]

\[
z=0
\Rightarrow
\sigma(z)=0.5
\]

\[
z\to+\infty
\Rightarrow
\sigma(z)\to1
\]

---

## 5. Probability Interpretation

In binary Logistic Regression:

\[
\sigma(z)
=
P(y=1\mid x)
\]

and:

\[
P(y=0\mid x)
=
1-\sigma(z)
\]

Example:

\[
\sigma(z)=0.8
\]

means:

\[
P(y=1\mid x)=0.8
\]

\[
P(y=0\mid x)=0.2
\]

---

## 6. From Probability to Class

The default threshold is:

\[
0.5
\]

Prediction rule:

\[
\hat y=
\begin{cases}
1 & \sigma(z)\ge0.5\\
0 & \sigma(z)<0.5
\end{cases}
\]

Because sigmoid is monotonically increasing and:

\[
\sigma(0)=0.5
\]

we have:

\[
\sigma(z)\ge0.5
\iff
z\ge0
\]

Therefore, the decision boundary is:

\[
w^Tx+b=0
\]

---

## 7. Adaline vs Logistic Regression

### Adaline

\[
x
\rightarrow
z=w^Tx+b
\rightarrow
\text{identity activation}
\rightarrow
\text{class}
\]

### Logistic Regression

\[
x
\rightarrow
z=w^Tx+b
\rightarrow
\text{sigmoid}
\rightarrow
\text{probability}
\rightarrow
\text{threshold}
\rightarrow
\text{class}
\]

The main difference is the activation function.

---

## 8. What Does Logistic Regression Learn?

Logistic Regression learns:

\[
\boxed{w,b}
\]

It does **not directly learn probabilities**.

Instead:

\[
w,b
\rightarrow
z=w^Tx+b
\rightarrow
\sigma(z)
\rightarrow
P(y=1\mid x)
\]

---

## Core Workflow

\[
\boxed{
x
\rightarrow
z=w^Tx+b
\rightarrow
\sigma(z)
\rightarrow
P(y=1\mid x)
\rightarrow
\text{threshold}
\rightarrow
\hat y
}
\]

### Key Points

- Sigmoid input: \(z=w^Tx+b\)
- Sigmoid output: \((0,1)\)
- \(\sigma(0)=0.5\)
- \(\sigma(z)\ge0.5 \iff z\ge0\)
- Decision boundary: \(w^Tx+b=0\)
- Learned parameters: \(w,b\)




# Logistic Regression — Logistic Loss

## 1. From Net Input to Probability

Logistic Regression first computes the net input:

\[
z = \mathbf{w}^T\mathbf{x} + b
\]

Then it applies the sigmoid function:

\[
p = \sigma(z)
\]

where \(p\) is the predicted probability of class 1.

---

## 2. Likelihood

For one sample:

\[
p(y|\mathbf{x})
=
p^y(1-p)^{1-y}
\]

If:

\[
y=1
\]

then:

\[
p(y|\mathbf{x}) = p
\]

If:

\[
y=0
\]

then:

\[
p(y|\mathbf{x}) = 1-p
\]

For the whole training set, assuming samples are independent:

\[
\mathcal{L}
=
\prod_{i=1}^{n}
p_i^{y_i}(1-p_i)^{1-y_i}
\]

Training tries to maximize this likelihood.

---

## 3. Log-Likelihood

We take the logarithm of the likelihood:

\[
\log \mathcal{L}
=
\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right]
\]

Reasons for using the log:

- Reduces the risk of numerical underflow.
- Converts products into sums.
- Makes derivatives easier to compute.

---

## 4. Logistic Loss

Instead of maximizing log-likelihood, we minimize its negative:

\[
L
=
-\left[
y\log p
+
(1-y)\log(1-p)
\right]
\]

### If \(y=1\)

\[
L=-\log p
\]

Therefore:

\[
p\rightarrow1
\Rightarrow
L\rightarrow0
\]

\[
p\rightarrow0
\Rightarrow
L\rightarrow\infty
\]

### If \(y=0\)

\[
L=-\log(1-p)
\]

Therefore:

\[
p\rightarrow0
\Rightarrow
L\rightarrow0
\]

\[
p\rightarrow1
\Rightarrow
L\rightarrow\infty
\]

---

## 5. Main Intuition

Logistic loss rewards correct predictions and strongly penalizes confident wrong predictions.

For example, if:

\[
y=1
\]

then:

\[
p=0.99
\]

gives a very small loss, while:

\[
p=0.01
\]

gives a very large loss.

---

## 6. Gradient Descent Update

The gradient with respect to a weight \(w_j\) is:

\[
\frac{\partial L}{\partial w_j}
=
(p-y)x_j
\]

Therefore:

\[
w_j
\leftarrow
w_j-\eta(p-y)x_j
\]

or equivalently:

\[
w_j
\leftarrow
w_j+\eta(y-p)x_j
\]

For the bias:

\[
b
\leftarrow
b+\eta(y-p)
\]

---

## 7. Logistic Regression Training Flow

\[
X
\rightarrow
z=Xw+b
\rightarrow
p=\sigma(z)
\rightarrow
L(y,p)
\rightarrow
\nabla L
\rightarrow
w,b\text{ update}
\]

The model learns:

\[
\boxed{w \text{ and } b}
\]

The probability \(p\) is computed from these parameters.

---

## 8. Code Interpretation

```python
net_input = self.net_input(X)
means:
[ z=Xw+b ]
output = self.activation(net_input)
means:
[ p=\sigma(z) ]
errors = y - output
means:
[ y-p ]
Here, output is the predicted probability, not the predicted class.
 
⸻
 
Key Takeaway
Logistic Regression learns (w) and (b) by minimizing logistic loss:
[ \boxed{ X \rightarrow z \rightarrow p \rightarrow loss \rightarrow gradient \rightarrow parameter\ update } ]







可以。下面这版把 p70–79 压缩成适合你现在复习和保存到 GitHub/VS Code 的统一英文 Markdown 格式，保留核心概念，不塞太多细节。
# Logistic Regression, Regularization, and Linear SVM

## 1. Logistic Regression with Scikit-Learn

A logistic regression model can be trained using:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=100.0,
    solver='lbfgs',
    multi_class='ovr'
)

lr.fit(X_train_std, y_train)
Basic workflow:
training data
→ fit()
→ learn model parameters
→ predict classes
A decision boundary separates different predicted classes.
 
⸻
 
2. Predicting Class Probabilities
Use:
lr.predict_proba(X_test_std)
For:
* 3 samples
* 3 classes
the output shape is:
(3, 3)
Each row represents one sample.
Example:
[0.01, 0.14, 0.85]
means:
P(class 0) = 0.01
P(class 1) = 0.14
P(class 2) = 0.85
The probabilities in each row sum to approximately:
1
 
⸻
 
3. From Probabilities to Class Labels
The predicted class is the class with the highest probability.
lr.predict_proba(X_test_std).argmax(axis=1)
Here:
axis=1
means finding the maximum value across columns for each sample.
Scikit-learn provides a simpler method:
lr.predict(X_test_std)
Conceptually:
predict_proba()
→ class probabilities
→ argmax
→ predicted class
 
⸻
 
4. Predicting a Single Sample
Scikit-learn expects input with shape:
(n_samples, n_features)
A single sample may have shape:
(n_features,)
For example:
(2,)
It should be converted to:
(1, 2)
using:
X_test_std[0, :].reshape(1, -1)
 
⸻
 
5. Underfitting and Overfitting
Underfitting
The model is too simple to capture the pattern in the data.
underfitting
→ high bias
→ poor training performance
→ poor test performance
Overfitting
The model fits the training data too closely, including noise.
overfitting
→ high variance
→ very good training performance
→ poor generalization to unseen data
Good Fit
A good model balances bias and variance.
too simple
→ underfitting

appropriate complexity
→ good generalization

too complex
→ overfitting
 
⸻
 
6. Regularization
Regularization helps reduce overfitting by penalizing large model weights.
For L2 regularization:
$$
 
L(w,b)
L_{\text{data}} + \frac{\lambda}{2n}|w|^2 $$
where:
$$
 
|w|^2
\sum_j w_j^2 $$
The regularization term penalizes large weights.
Conceptually:
large weights
→ larger penalty
→ weights shrink
→ simpler model
→ less overfitting
 
⸻
 
7. Effect of L2 Regularization on the Gradient
Without regularization:
$$ \frac{\partial L}{\partial w_j} $$
With L2 regularization:
$$
 
\frac{\partial L}{\partial w_j}
\frac{\partial L_{\text{data}}}{\partial w_j} + \frac{\lambda}{n}w_j $$
The additional term:
$$ \frac{\lambda}{n}w_j $$
pushes large weights toward zero.
 
⸻
 
8. Regularization Parameter $\lambda$
The parameter:
$$ \lambda $$
controls regularization strength.
larger λ
→ stronger regularization
→ smaller weights
→ simpler model

smaller λ
→ weaker regularization
→ larger weights allowed
→ more complex model
Too much regularization can cause underfitting.
 
⸻
 
9. The Parameter C
Scikit-learn commonly uses:
C
instead of directly using $\lambda$.
C is inversely related to regularization strength.
C ↑
→ regularization ↓
→ weights can become larger
→ stronger fit to training data

C ↓
→ regularization ↑
→ weights become smaller
→ simpler model
Therefore:
$$ C \uparrow \Rightarrow \text{regularization strength} \downarrow $$
$$ C \downarrow \Rightarrow \text{regularization strength} \uparrow $$
 
⸻
 
Linear Support Vector Machine
10. Maximum-Margin Classification
Many decision boundaries may correctly separate two classes.
SVM does not choose an arbitrary boundary.
Instead, it searches for the boundary with the largest margin.
possible decision boundaries
→ find the largest margin
→ choose maximum-margin boundary
Main objective:
$$ \boxed{\text{maximize the margin}} $$
 
⸻
 
11. Hyperplane and Decision Boundary
The linear decision boundary can be written as:
$$ w^Tx+b=0 $$
In two dimensions, it is a line.
In higher-dimensional spaces, it is called a:
hyperplane
For a linear SVM, the hyperplane is the decision boundary used for classification.
 
⸻
 
12. Support Vectors
Support vectors are the training samples closest to the decision boundary.
support vectors
→ closest important training samples
→ determine the position of the boundary
→ determine the margin
Samples far away from the decision boundary usually have less influence on the final SVM boundary.
 
⸻
 
13. Margin
The margin is the gap between the separating boundary and the closest samples from the two classes.
The closest samples are the support vectors.
SVM tries to find:
$$ \boxed{\text{maximum margin}} $$
A larger margin often improves generalization to unseen data.
 
⸻
 
14. Hard Margin vs Soft Margin
Hard Margin
A hard-margin SVM requires all training samples to be correctly separated.
This works only when the data is perfectly linearly separable.
Soft Margin
Real-world datasets may contain:
* noise
* outliers
* overlapping classes
Soft-margin SVM allows some margin violations or classification errors.
allow some errors
→ avoid fitting every training sample perfectly
→ improve generalization
 
⸻
 
15. C in SVM
In SVM, C controls the trade-off between:
large margin
vs.
classification errors
Large C
large C
→ strong penalty for errors
→ less tolerance for violations
→ usually narrower margin
→ higher overfitting risk
Small C
small C
→ weaker penalty for errors
→ more tolerance for violations
→ usually wider margin
→ stronger regularization
Important:
C does not directly specify the margin width.
It controls the trade-off between margin size and classification errors.
 
⸻
 
16. Linear SVM in Scikit-Learn
A linear SVM can be created using:
from sklearn.svm import SVC

svm = SVC(
    kernel='linear',
    C=1.0,
    random_state=1
)

svm.fit(X_train_std, y_train)
kernel='linear'
Uses a linear decision boundary.
C
Controls the trade-off between:
margin size
and
classification errors
 
⸻
 
17. Logistic Regression vs Linear SVM
Logistic Regression
Focuses on modeling class probabilities:
$$ P(y \mid x) $$
It can directly produce probability estimates using:
predict_proba()
Linear SVM
Focuses on finding a maximum-margin decision boundary.
Logistic Regression
→ probability

Linear SVM
→ maximum-margin decision boundary
Another important difference:
Logistic Regression
→ all training samples contribute to the loss

SVM
→ decision boundary is mainly determined by support vectors
Both methods can produce similar linear decision boundaries, but their optimization objectives are different.
 
⸻
 
Key Relationships
Underfitting
→ high bias

Overfitting
→ high variance
Regularization
→ penalize large weights
→ simpler model
→ reduce overfitting
C ↑
→ weaker regularization

C ↓
→ stronger regularization
Linear SVM
→ decision boundary
→ support vectors
→ margin
→ maximum margin
→ soft margin
→ C
Large C
→ punish classification errors more strongly
→ narrower margin

Small C
→ tolerate more errors
→ wider margin
Core Takeaway
Logistic Regression:
learn class probabilities

Regularization:
control model complexity by penalizing large weights

Linear SVM:
find a decision boundary with the largest possible margin

Soft-margin SVM:
balance margin width and classification errors using C


# Kernel SVM

## 1. Why Kernel SVM?

### Core Idea

A linear SVM can only create a linear decision boundary.

\[
w^T x + b = 0
\]

Some datasets, such as XOR data, are not linearly separable in the original feature space.

### Key Points

- Linear SVM learns a linear decision boundary.
- XOR data cannot be separated well by a straight line.
- Kernel SVM is used for nonlinear classification problems.

---

## 2. Feature Mapping

### Core Idea

Kernel methods map the original data into a higher-dimensional feature space.

\[
x \rightarrow \phi(x)
\]

Example:

\[
(x_1, x_2)
\rightarrow
(x_1, x_2, x_1^2 + x_2^2)
\]

### Key Points

- The original data may not be linearly separable.
- The transformed data may become linearly separable.
- SVM still learns a linear hyperplane in the higher-dimensional space.
- The hyperplane becomes a nonlinear decision boundary in the original space.

---

## 3. Kernel Trick

### Core Idea

Explicitly computing the transformed features \(\phi(x)\) can be expensive.

The kernel trick directly computes:

\[
k(x^{(i)}, x^{(j)})
=
\phi(x^{(i)})^T \phi(x^{(j)})
\]

### Key Points

- Explicit feature mapping may be computationally expensive.
- The kernel trick avoids explicitly computing \(\phi(x)\).
- It directly computes the inner product in the higher-dimensional feature space.
- This makes nonlinear SVM more efficient.

---

## 4. RBF Kernel

### Core Idea

The RBF kernel measures the similarity between two samples.

\[
k(x^{(i)}, x^{(j)})
=
\exp
\left(
-\gamma
\|x^{(i)} - x^{(j)}\|^2
\right)
\]

### Key Points

- Small distance means high similarity.
- Large distance means low similarity.
- Similar samples have kernel values close to \(1\).
- Dissimilar samples have kernel values close to \(0\).

---

## 5. Gamma

### Core Idea

The hyperparameter `gamma` controls the influence range of each training sample.

### Key Points

- Small `gamma` gives each sample a wider influence range.
- Small `gamma` produces a smoother decision boundary.
- Small `gamma` usually means lower model complexity.
- Large `gamma` gives each sample a narrower influence range.
- Large `gamma` produces a more complex decision boundary.
- Large `gamma` increases the risk of overfitting.

\[
\gamma \uparrow
\Rightarrow
\text{model complexity} \uparrow
\Rightarrow
\text{overfitting risk} \uparrow
\]

---

## 6. C

### Core Idea

The hyperparameter `C` controls the penalty for classification errors.

### Key Points

- Small `C` allows more classification errors.
- Small `C` usually gives a simpler decision boundary.
- Large `C` strongly penalizes classification errors.
- Large `C` allows less tolerance for misclassification.
- Large `C` may increase model complexity.

---

## 7. C vs. Gamma

### Core Idea

`C` and `gamma` both affect model complexity, but they control different things.

### Key Points

- `C` controls the penalty for classification errors.
- `gamma` controls the influence range of each training sample.
- Large `C` means stronger punishment for classification errors.
- Large `gamma` means more local sample influence.
- Large `gamma` can create a more complex decision boundary.

---

## 8. Scikit-Learn Example

### Core Idea

RBF Kernel SVM can be implemented using `SVC`.

```python
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    gamma=0.2,
    C=1.0
)

svm.fit(X_train_std, y_train)

Key Points
	•	kernel="rbf" selects the RBF kernel.
	•	gamma controls the influence range of training samples.
	•	C controls the penalty for classification errors.
	•	fit() learns the decision boundary from the training data.
 
⸻
 
9. Kernel SVM Workflow
Core Idea
Kernel SVM solves nonlinear classification by working in a higher-dimensional feature space.
Key Points
	1.	Start with nonlinear data.
	2.	Map the data into a higher-dimensional feature space.
	3.	Learn a linear hyperplane in that space.
	4.	Use the kernel trick to avoid explicit feature mapping.
	5.	Obtain a nonlinear decision boundary in the original feature space.
 
⸻
 
10. Key Takeaways
Core Idea
Kernel SVM extends linear SVM to nonlinear classification problems.
Key Points
	•	Linear SVM creates a linear decision boundary.
	•	Kernel SVM can create nonlinear decision boundaries.
	•	Feature mapping moves data into a higher-dimensional space.
	•	SVM still learns a linear hyperplane in the transformed space.
	•	The kernel trick avoids explicit feature mapping.
	•	RBF kernel measures similarity between samples.
	•	Small gamma produces smoother boundaries.
	•	Large gamma produces more complex boundaries.
	•	C controls classification error penalty.
	•	gamma controls sample influence range.



# Chapter 3 — Decision Trees, Random Forests, and KNN

## 1. Decision Tree

### Core Idea

A Decision Tree makes predictions by repeatedly splitting the feature space.

### Key Points

- A typical split has the form: `feature <= threshold`.
- The model learns:
  - which feature to use;
  - which threshold to use;
  - how the tree is structured.
- The model does not mainly learn `w` and `b`.
- A sample moves from the root node through internal nodes until it reaches a leaf node.
- The leaf node gives the final prediction.

### Summary

`feature + threshold -> split -> branch -> leaf -> prediction`

---

## 2. Tree Structure

### Core Idea

A Decision Tree consists of nodes and branches.

### Key Points

- Root node: the first split.
- Internal node: an intermediate split.
- Branch: the result of a split.
- Leaf node: the final prediction.

### Summary

`root -> internal nodes -> leaf -> prediction`

---

## 3. Impurity

### Core Idea

Impurity measures how mixed the classes are inside a node.

### Key Points

- Low impurity means the node is relatively pure.
- High impurity means the classes are mixed.
- A completely pure node contains samples from only one class.
- For a pure node:
  - `Gini = 0`;
  - `Entropy = 0`.

### Summary

`low impurity -> pure`

`high impurity -> mixed`

---

## 4. Information Gain

### Core Idea

Information Gain measures how much a split reduces impurity.

### Key Points

- Information Gain compares impurity before and after a split.
- The child-node impurity is weighted by the number of samples.
- A larger Information Gain means the split produces purer child nodes.
- A Decision Tree tries to choose the split with the largest Information Gain.

### Formula

`Information Gain = impurity before split - weighted impurity after split`

### Summary

`large Information Gain -> better split -> purer child nodes`

---

## 5. Entropy

### Core Idea

Entropy is one measure of node impurity.

### Key Points

- Entropy is low when a node is pure.
- Entropy is high when classes are strongly mixed.
- In binary classification:
  - `100% / 0% -> entropy = 0`;
  - `50% / 50% -> entropy is maximum`.

### Formula

`H = -Σ p_i log2(p_i)`

### Summary

`pure node -> low entropy`

`mixed node -> high entropy`

---

## 6. Gini Impurity

### Core Idea

Gini impurity is another measure of node impurity.

### Key Points

- Gini impurity is `0` for a completely pure node.
- In binary classification:
  - `100% / 0% -> Gini = 0`;
  - `50% / 50% -> Gini = 0.5`.
- Gini and Entropy usually produce similar results in practice.

### Formula

`Gini = 1 - Σ p_i²`

### Summary

`Gini ≈ Entropy`

`both measure node impurity`

---

## 7. Tree Depth

### Core Idea

Tree depth controls the complexity of a Decision Tree.

### Key Points

- A deeper tree creates more splits.
- More splits create more detailed decision regions.
- A deeper tree can fit the training data more closely.
- A tree that is too deep may fit noise in the training data.
- This increases the risk of overfitting.

### Summary

`depth ↑ -> complexity ↑ -> overfitting risk ↑`

---

## 8. max_depth

### Core Idea

`max_depth` limits how deep a Decision Tree can grow.

### Key Points

- `max_depth` is a hyperparameter.
- A smaller value creates a simpler tree.
- A larger value allows a more complex tree.
- Limiting depth is one way to control overfitting.

### Example

`DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)`

### Summary

`max_depth -> controls model complexity`

---

## 9. Nonlinear Decision Boundary

### Core Idea

A Decision Tree can create a nonlinear decision boundary using many simple splits.

### Key Points

- Each individual split is simple.
- A split usually creates an axis-aligned boundary.
- Multiple splits divide the feature space into rectangular regions.
- Combining many rectangular regions creates a piecewise nonlinear boundary.

### Summary

`many simple splits -> rectangular regions -> nonlinear decision boundary`

---

## 10. Feature Scaling in Decision Trees

### Core Idea

Decision Trees usually do not require feature standardization.

### Key Points

- Decision Trees mainly compare values using thresholds.
- The relative ordering of samples is more important than the absolute scale.
- Scaling changes the numerical threshold but usually not the ordering of samples.
- Therefore, standardization usually does not change the basic split structure.

### Summary

`Decision Tree -> ordering matters -> scaling usually unnecessary`

---

## 11. Random Forest

### Core Idea

A Random Forest combines many Decision Trees.

### Key Points

- A Random Forest is an ensemble model.
- Each tree is intentionally made different.
- Different trees make different errors.
- Their predictions are combined.
- The final model is usually more stable than a single deep Decision Tree.

### Summary

`many different Decision Trees -> combine predictions -> more robust model`

---

## 12. Why Random Forest Works

### Core Idea

Random Forest reduces the instability of individual Decision Trees.

### Key Points

- A deep Decision Tree can have high variance.
- Small changes in the training data may produce very different trees.
- Random Forest creates many different trees.
- Voting or averaging reduces the effect of individual fluctuations.
- The final ensemble usually has lower variance.

### Summary

`different trees -> different errors -> combine predictions -> variance ↓`

---

## 13. Bootstrap Sampling

### Core Idea

Each tree in a Random Forest is trained on a bootstrap sample.

### Key Points

- Bootstrap sampling means random sampling with replacement.
- A sample can appear more than once.
- Some original samples may not appear at all.

### Example

Original dataset:

`A B C D`

Possible bootstrap sample:

`A A C D`

### Summary

`bootstrap sampling = random sampling with replacement`

---

## 14. Random Feature Subsets

### Core Idea

Random Forest also introduces randomness through feature selection.

### Key Points

- At each node, only a random subset of features is considered.
- The best split is selected only from that subset.
- A new random subset can be selected at the next node.
- This makes different trees less similar to each other.

### Summary

`random feature subset -> best split within subset -> more diverse trees`

---

## 15. Random Forest Algorithm

### Core Idea

Random Forest repeatedly builds randomized Decision Trees and combines their predictions.

### Key Points

1. Draw a bootstrap sample.
2. Grow a Decision Tree.
3. At each node, randomly select a subset of features.
4. Choose the best split among those features.
5. Repeat the process many times.
6. Combine the predictions of all trees.

### Summary

`bootstrap sample -> random features -> Decision Tree -> repeat -> combine predictions`

---

## 16. Majority Vote

### Core Idea

For classification, Random Forest combines tree predictions using majority vote.

### Key Points

- Each tree predicts a class.
- The class with the most votes becomes the final prediction.

### Example

`Tree 1 -> 0`

`Tree 2 -> 1`

`Tree 3 -> 1`

`Tree 4 -> 1`

`Tree 5 -> 0`

Final prediction:

`1`

### Summary

`majority vote -> final class prediction`

---

## 17. Random Forest Hyperparameters

### Core Idea

Random Forest behavior can be controlled using hyperparameters.

### Key Points

- `n_estimators`: number of Decision Trees.
- `random_state`: controls reproducible randomness.
- `n_jobs`: controls parallel computation.

### Example

`RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)`

### Summary

`n_estimators=25 -> 25 Decision Trees`

---

## 18. Parametric Models

### Core Idea

Parametric models learn a fixed-size set of parameters.

### Key Points

- The number of learned parameters does not directly grow with the number of training samples.
- Typical learned parameters include `w` and `b`.

### Examples

- Logistic Regression
- Linear SVM
- Perceptron

### Summary

`parametric model -> fixed-size learned parameter set`

---

## 19. Non-Parametric Models

### Core Idea

Non-parametric models are not described by a fixed-size parameter vector.

### Key Points

- Model complexity can grow with the training data.
- Non-parametric does not mean that the model has no hyperparameters.

### Examples

- Decision Tree
- Random Forest
- KNN

### Summary

`non-parametric model -> no fixed-size parameter vector`

---

## 20. K-Nearest Neighbors

### Core Idea

KNN predicts a new sample using nearby training samples.

### Key Points

- KNN mainly stores the training data.
- It does not learn a fixed set of parameters such as `w` and `b`.
- Most computation happens during prediction.
- Prediction depends on distance between samples.

### Summary

`new sample -> distance -> nearest neighbors -> vote -> prediction`

---

## 21. Lazy Learning

### Core Idea

KNN is called a lazy learner because little computation happens during training.

### Key Points

- Training mainly stores the training data.
- There is no iterative parameter optimization.
- Prediction requires distance calculations.
- Prediction can therefore be computationally expensive.

### Summary

`training -> little work`

`prediction -> most work`

---

## 22. KNN Prediction

### Core Idea

KNN classification uses the labels of the nearest training samples.

### Key Points

1. Choose `k`.
2. Choose a distance metric.
3. Calculate distances from the new sample to training samples.
4. Find the `k` nearest training samples.
5. Use majority vote to predict the class.

### Summary

`distance -> k nearest neighbors -> majority vote`

---

## 23. Choosing k

### Core Idea

The value of `k` controls the complexity of the KNN decision boundary.

### Key Points

- Small `k`:
  - more flexible;
  - more sensitive to noise;
  - more complex decision boundary;
  - higher overfitting risk.
- Large `k`:
  - smoother decision boundary;
  - less sensitive to individual samples;
  - higher underfitting risk if too large.

### Summary

`k ↓ -> complexity ↑ -> overfitting risk ↑`

`k ↑ -> smoother boundary -> underfitting risk ↑`

---

## 24. Distance Metric

### Core Idea

KNN uses a distance metric to measure similarity between samples.

### Key Points

- Minkowski distance is a general distance metric.
- `p=2` gives Euclidean distance.
- `p=1` gives Manhattan distance.

### Example

`KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")`

### Summary

`p=2 -> Euclidean distance`

`p=1 -> Manhattan distance`

---

## 25. Feature Scaling in KNN

### Core Idea

KNN usually requires feature scaling because prediction depends directly on distance.

### Key Points

- Features with larger numerical scales can dominate the distance.
- Features with smaller numerical scales may contribute very little.
- Standardization makes feature scales more comparable.

### Example

`height: 150–190`

`income: 0–10,000,000`

Without scaling:

`income dominates the distance`

### Summary

`KNN -> distance matters -> scaling is very important`

---

## 26. Decision Tree vs. KNN Scaling

### Core Idea

Decision Trees and KNN react differently to feature scale.

### Key Points

- Decision Tree:
  - mainly uses ordering and thresholds;
  - scaling is usually unnecessary.
- KNN:
  - directly uses distances;
  - scaling is usually important.

### Summary

`Decision Tree -> ordering matters`

`KNN -> distance matters`

---

## 27. Curse of Dimensionality

### Core Idea

KNN becomes more difficult in very high-dimensional spaces.

### Key Points

- More dimensions create a much larger feature space.
- Training samples become increasingly sparse.
- Even the nearest neighbors may be relatively far away.
- Distance becomes less informative.
- KNN performance may decrease.

### Summary

`dimensions ↑ -> samples become sparse -> distance becomes less useful`

---

## 28. Model Comparison

| Model | Main Learned Information | Prediction Method | Feature Scaling |
|---|---|---|---|
| Logistic Regression | `w, b` | Sigmoid probability | Usually important |
| Linear SVM | `w, b` | Maximum-margin boundary | Usually important |
| Decision Tree | Features, thresholds, tree structure | Follow tree splits | Usually unnecessary |
| Random Forest | Many Decision Trees | Majority vote | Usually unnecessary |
| KNN | Mainly stores training samples | Distance + neighbors + vote | Very important |

---

## 29. Final Summary

### Decision Tree

`feature + threshold -> split -> Information Gain -> repeat -> leaf -> prediction`

### Random Forest

`bootstrap samples -> random feature subsets -> many trees -> majority vote -> lower variance`

### KNN

`store training data -> calculate distance -> find k nearest neighbors -> majority vote`

### Complexity

`Decision Tree: depth ↑ -> complexity ↑`

`KNN: k ↓ -> complexity ↑`

### Feature Scaling

`Decision Tree -> usually unnecessary`

`KNN -> very important` # Kernel SVM

## 1. Why Kernel SVM?

### Core Idea

A linear SVM can only create a linear decision boundary.

\[
w^T x + b = 0
\]

Some datasets, such as XOR data, are not linearly separable in the original feature space.

### Key Points

- Linear SVM learns a linear decision boundary.
- XOR data cannot be separated well by a straight line.
- Kernel SVM is used for nonlinear classification problems.

---

## 2. Feature Mapping

### Core Idea

Kernel methods map the original data into a higher-dimensional feature space.

\[
x \rightarrow \phi(x)
\]

Example:

\[
(x_1, x_2)
\rightarrow
(x_1, x_2, x_1^2 + x_2^2)
\]

### Key Points

- The original data may not be linearly separable.
- The transformed data may become linearly separable.
- SVM still learns a linear hyperplane in the higher-dimensional space.
- The hyperplane becomes a nonlinear decision boundary in the original space.

---

## 3. Kernel Trick

### Core Idea

Explicitly computing the transformed features \(\phi(x)\) can be expensive.

The kernel trick directly computes:

\[
k(x^{(i)}, x^{(j)})
=
\phi(x^{(i)})^T \phi(x^{(j)})
\]

### Key Points

- Explicit feature mapping may be computationally expensive.
- The kernel trick avoids explicitly computing \(\phi(x)\).
- It directly computes the inner product in the higher-dimensional feature space.
- This makes nonlinear SVM more efficient.

---

## 4. RBF Kernel

### Core Idea

The RBF kernel measures the similarity between two samples.

\[
k(x^{(i)}, x^{(j)})
=
\exp
\left(
-\gamma
\|x^{(i)} - x^{(j)}\|^2
\right)
\]

### Key Points

- Small distance means high similarity.
- Large distance means low similarity.
- Similar samples have kernel values close to \(1\).
- Dissimilar samples have kernel values close to \(0\).

---

## 5. Gamma

### Core Idea

The hyperparameter `gamma` controls the influence range of each training sample.

### Key Points

- Small `gamma` gives each sample a wider influence range.
- Small `gamma` produces a smoother decision boundary.
- Small `gamma` usually means lower model complexity.
- Large `gamma` gives each sample a narrower influence range.
- Large `gamma` produces a more complex decision boundary.
- Large `gamma` increases the risk of overfitting.

\[
\gamma \uparrow
\Rightarrow
\text{model complexity} \uparrow
\Rightarrow
\text{overfitting risk} \uparrow
\]

---

## 6. C

### Core Idea

The hyperparameter `C` controls the penalty for classification errors.

### Key Points

- Small `C` allows more classification errors.
- Small `C` usually gives a simpler decision boundary.
- Large `C` strongly penalizes classification errors.
- Large `C` allows less tolerance for misclassification.
- Large `C` may increase model complexity.

---

## 7. C vs. Gamma

### Core Idea

`C` and `gamma` both affect model complexity, but they control different things.

### Key Points

- `C` controls the penalty for classification errors.
- `gamma` controls the influence range of each training sample.
- Large `C` means stronger punishment for classification errors.
- Large `gamma` means more local sample influence.
- Large `gamma` can create a more complex decision boundary.

---

## 8. Scikit-Learn Example

### Core Idea

RBF Kernel SVM can be implemented using `SVC`.

```python
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    gamma=0.2,
    C=1.0
)

svm.fit(X_train_std, y_train)

Key Points
* kernel="rbf" selects the RBF kernel.
* gamma controls the influence range of training samples.
* C controls the penalty for classification errors.
* fit() learns the decision boundary from the training data.
 
⸻
 
9. Kernel SVM Workflow
Core Idea
Kernel SVM solves nonlinear classification by working in a higher-dimensional feature space.
Key Points
1. Start with nonlinear data.
2. Map the data into a higher-dimensional feature space.
3. Learn a linear hyperplane in that space.
4. Use the kernel trick to avoid explicit feature mapping.
5. Obtain a nonlinear decision boundary in the original feature space.
 
⸻
 
10. Key Takeaways
Core Idea
Kernel SVM extends linear SVM to nonlinear classification problems.
Key Points
* Linear SVM creates a linear decision boundary.
* Kernel SVM can create nonlinear decision boundaries.
* Feature mapping moves data into a higher-dimensional space.
* SVM still learns a linear hyperplane in the transformed space.
* The kernel trick avoids explicit feature mapping.
* RBF kernel measures similarity between samples.
* Small gamma produces smoother boundaries.
* Large gamma produces more complex boundaries.
* C controls classification error penalty.
* gamma controls sample influence range.



# Chapter 3 — Decision Trees, Random Forests, and KNN

## 1. Decision Tree

### Core Idea

A Decision Tree makes predictions by repeatedly splitting the feature space.

### Key Points

- A typical split has the form: `feature <= threshold`.
- The model learns:
  - which feature to use;
  - which threshold to use;
  - how the tree is structured.
- The model does not mainly learn `w` and `b`.
- A sample moves from the root node through internal nodes until it reaches a leaf node.
- The leaf node gives the final prediction.

### Summary

`feature + threshold -> split -> branch -> leaf -> prediction`

---

## 2. Tree Structure

### Core Idea

A Decision Tree consists of nodes and branches.

### Key Points

- Root node: the first split.
- Internal node: an intermediate split.
- Branch: the result of a split.
- Leaf node: the final prediction.

### Summary

`root -> internal nodes -> leaf -> prediction`

---

## 3. Impurity

### Core Idea

Impurity measures how mixed the classes are inside a node.

### Key Points

- Low impurity means the node is relatively pure.
- High impurity means the classes are mixed.
- A completely pure node contains samples from only one class.
- For a pure node:
  - `Gini = 0`;
  - `Entropy = 0`.

### Summary

`low impurity -> pure`

`high impurity -> mixed`

---

## 4. Information Gain

### Core Idea

Information Gain measures how much a split reduces impurity.

### Key Points

- Information Gain compares impurity before and after a split.
- The child-node impurity is weighted by the number of samples.
- A larger Information Gain means the split produces purer child nodes.
- A Decision Tree tries to choose the split with the largest Information Gain.

### Formula

`Information Gain = impurity before split - weighted impurity after split`

### Summary

`large Information Gain -> better split -> purer child nodes`

---

## 5. Entropy

### Core Idea

Entropy is one measure of node impurity.

### Key Points

- Entropy is low when a node is pure.
- Entropy is high when classes are strongly mixed.
- In binary classification:
  - `100% / 0% -> entropy = 0`;
  - `50% / 50% -> entropy is maximum`.

### Formula

`H = -Σ p_i log2(p_i)`

### Summary

`pure node -> low entropy`

`mixed node -> high entropy`

---

## 6. Gini Impurity

### Core Idea

Gini impurity is another measure of node impurity.

### Key Points

- Gini impurity is `0` for a completely pure node.
- In binary classification:
  - `100% / 0% -> Gini = 0`;
  - `50% / 50% -> Gini = 0.5`.
- Gini and Entropy usually produce similar results in practice.

### Formula

`Gini = 1 - Σ p_i²`

### Summary

`Gini ≈ Entropy`

`both measure node impurity`

---

## 7. Tree Depth

### Core Idea

Tree depth controls the complexity of a Decision Tree.

### Key Points

- A deeper tree creates more splits.
- More splits create more detailed decision regions.
- A deeper tree can fit the training data more closely.
- A tree that is too deep may fit noise in the training data.
- This increases the risk of overfitting.

### Summary

`depth ↑ -> complexity ↑ -> overfitting risk ↑`

---

## 8. max_depth

### Core Idea

`max_depth` limits how deep a Decision Tree can grow.

### Key Points

- `max_depth` is a hyperparameter.
- A smaller value creates a simpler tree.
- A larger value allows a more complex tree.
- Limiting depth is one way to control overfitting.

### Example

`DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)`

### Summary

`max_depth -> controls model complexity`

---

## 9. Nonlinear Decision Boundary

### Core Idea

A Decision Tree can create a nonlinear decision boundary using many simple splits.

### Key Points

- Each individual split is simple.
- A split usually creates an axis-aligned boundary.
- Multiple splits divide the feature space into rectangular regions.
- Combining many rectangular regions creates a piecewise nonlinear boundary.

### Summary

`many simple splits -> rectangular regions -> nonlinear decision boundary`

---

## 10. Feature Scaling in Decision Trees

### Core Idea

Decision Trees usually do not require feature standardization.

### Key Points

- Decision Trees mainly compare values using thresholds.
- The relative ordering of samples is more important than the absolute scale.
- Scaling changes the numerical threshold but usually not the ordering of samples.
- Therefore, standardization usually does not change the basic split structure.

### Summary

`Decision Tree -> ordering matters -> scaling usually unnecessary`

---

## 11. Random Forest

### Core Idea

A Random Forest combines many Decision Trees.

### Key Points

- A Random Forest is an ensemble model.
- Each tree is intentionally made different.
- Different trees make different errors.
- Their predictions are combined.
- The final model is usually more stable than a single deep Decision Tree.

### Summary

`many different Decision Trees -> combine predictions -> more robust model`

---

## 12. Why Random Forest Works

### Core Idea

Random Forest reduces the instability of individual Decision Trees.

### Key Points

- A deep Decision Tree can have high variance.
- Small changes in the training data may produce very different trees.
- Random Forest creates many different trees.
- Voting or averaging reduces the effect of individual fluctuations.
- The final ensemble usually has lower variance.

### Summary

`different trees -> different errors -> combine predictions -> variance ↓`

---

## 13. Bootstrap Sampling

### Core Idea

Each tree in a Random Forest is trained on a bootstrap sample.

### Key Points

- Bootstrap sampling means random sampling with replacement.
- A sample can appear more than once.
- Some original samples may not appear at all.

### Example

Original dataset:

`A B C D`

Possible bootstrap sample:

`A A C D`

### Summary

`bootstrap sampling = random sampling with replacement`

---

## 14. Random Feature Subsets

### Core Idea

Random Forest also introduces randomness through feature selection.

### Key Points

- At each node, only a random subset of features is considered.
- The best split is selected only from that subset.
- A new random subset can be selected at the next node.
- This makes different trees less similar to each other.

### Summary

`random feature subset -> best split within subset -> more diverse trees`

---

## 15. Random Forest Algorithm

### Core Idea

Random Forest repeatedly builds randomized Decision Trees and combines their predictions.

### Key Points

1. Draw a bootstrap sample.
2. Grow a Decision Tree.
3. At each node, randomly select a subset of features.
4. Choose the best split among those features.
5. Repeat the process many times.
6. Combine the predictions of all trees.

### Summary

`bootstrap sample -> random features -> Decision Tree -> repeat -> combine predictions`

---

## 16. Majority Vote

### Core Idea

For classification, Random Forest combines tree predictions using majority vote.

### Key Points

- Each tree predicts a class.
- The class with the most votes becomes the final prediction.

### Example

`Tree 1 -> 0`

`Tree 2 -> 1`

`Tree 3 -> 1`

`Tree 4 -> 1`

`Tree 5 -> 0`

Final prediction:

`1`

### Summary

`majority vote -> final class prediction`

---

## 17. Random Forest Hyperparameters

### Core Idea

Random Forest behavior can be controlled using hyperparameters.

### Key Points

- `n_estimators`: number of Decision Trees.
- `random_state`: controls reproducible randomness.
- `n_jobs`: controls parallel computation.

### Example

`RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)`

### Summary

`n_estimators=25 -> 25 Decision Trees`

---

## 18. Parametric Models

### Core Idea

Parametric models learn a fixed-size set of parameters.

### Key Points

- The number of learned parameters does not directly grow with the number of training samples.
- Typical learned parameters include `w` and `b`.

### Examples

- Logistic Regression
- Linear SVM
- Perceptron

### Summary

`parametric model -> fixed-size learned parameter set`

---

## 19. Non-Parametric Models

### Core Idea

Non-parametric models are not described by a fixed-size parameter vector.

### Key Points

- Model complexity can grow with the training data.
- Non-parametric does not mean that the model has no hyperparameters.

### Examples

- Decision Tree
- Random Forest
- KNN

### Summary

`non-parametric model -> no fixed-size parameter vector`

---

## 20. K-Nearest Neighbors

### Core Idea

KNN predicts a new sample using nearby training samples.

### Key Points

- KNN mainly stores the training data.
- It does not learn a fixed set of parameters such as `w` and `b`.
- Most computation happens during prediction.
- Prediction depends on distance between samples.

### Summary

`new sample -> distance -> nearest neighbors -> vote -> prediction`

---

## 21. Lazy Learning

### Core Idea

KNN is called a lazy learner because little computation happens during training.

### Key Points

- Training mainly stores the training data.
- There is no iterative parameter optimization.
- Prediction requires distance calculations.
- Prediction can therefore be computationally expensive.

### Summary

`training -> little work`

`prediction -> most work`

---

## 22. KNN Prediction

### Core Idea

KNN classification uses the labels of the nearest training samples.

### Key Points

1. Choose `k`.
2. Choose a distance metric.
3. Calculate distances from the new sample to training samples.
4. Find the `k` nearest training samples.
5. Use majority vote to predict the class.

### Summary

`distance -> k nearest neighbors -> majority vote`

---

## 23. Choosing k

### Core Idea

The value of `k` controls the complexity of the KNN decision boundary.

### Key Points

- Small `k`:
  - more flexible;
  - more sensitive to noise;
  - more complex decision boundary;
  - higher overfitting risk.
- Large `k`:
  - smoother decision boundary;
  - less sensitive to individual samples;
  - higher underfitting risk if too large.

### Summary

`k ↓ -> complexity ↑ -> overfitting risk ↑`

`k ↑ -> smoother boundary -> underfitting risk ↑`

---

## 24. Distance Metric

### Core Idea

KNN uses a distance metric to measure similarity between samples.

### Key Points

- Minkowski distance is a general distance metric.
- `p=2` gives Euclidean distance.
- `p=1` gives Manhattan distance.

### Example

`KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")`

### Summary

`p=2 -> Euclidean distance`

`p=1 -> Manhattan distance`

---

## 25. Feature Scaling in KNN

### Core Idea

KNN usually requires feature scaling because prediction depends directly on distance.

### Key Points

- Features with larger numerical scales can dominate the distance.
- Features with smaller numerical scales may contribute very little.
- Standardization makes feature scales more comparable.

### Example

`height: 150–190`

`income: 0–10,000,000`

Without scaling:

`income dominates the distance`

### Summary

`KNN -> distance matters -> scaling is very important`

---

## 26. Decision Tree vs. KNN Scaling

### Core Idea

Decision Trees and KNN react differently to feature scale.

### Key Points

- Decision Tree:
  - mainly uses ordering and thresholds;
  - scaling is usually unnecessary.
- KNN:
  - directly uses distances;
  - scaling is usually important.

### Summary

`Decision Tree -> ordering matters`

`KNN -> distance matters`

---

## 27. Curse of Dimensionality

### Core Idea

KNN becomes more difficult in very high-dimensional spaces.

### Key Points

- More dimensions create a much larger feature space.
- Training samples become increasingly sparse.
- Even the nearest neighbors may be relatively far away.
- Distance becomes less informative.
- KNN performance may decrease.

### Summary

`dimensions ↑ -> samples become sparse -> distance becomes less useful`

---

## 28. Model Comparison

| Model | Main Learned Information | Prediction Method | Feature Scaling |
|---|---|---|---|
| Logistic Regression | `w, b` | Sigmoid probability | Usually important |
| Linear SVM | `w, b` | Maximum-margin boundary | Usually important |
| Decision Tree | Features, thresholds, tree structure | Follow tree splits | Usually unnecessary |
| Random Forest | Many Decision Trees | Majority vote | Usually unnecessary |
| KNN | Mainly stores training samples | Distance + neighbors + vote | Very important |

---

## 29. Final Summary

### Decision Tree

`feature + threshold -> split -> Information Gain -> repeat -> leaf -> prediction`

### Random Forest

`bootstrap samples -> random feature subsets -> many trees -> majority vote -> lower variance`

### KNN

`store training data -> calculate distance -> find k nearest neighbors -> majority vote`

### Complexity

`Decision Tree: depth ↑ -> complexity ↑`

`KNN: k ↓ -> complexity ↑`

### Feature Scaling

`Decision Tree -> usually unnecessary`

`KNN -> very important`