MLwPS Chapter 3 — Classification with Scikit-Learn
1. Scikit-Learn Workflow
Train-Test Split
The dataset is divided into:
* Training set: used to learn model parameters.
* Test set: used to evaluate performance on unseen data.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)
Important parameters:
* test_size=0.3: 30% test data and 70% training data.
* random_state=1: makes the split reproducible.
* stratify=y: preserves class proportions.
Standardization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
fit() learns the mean and standard deviation from the training set:
\mu_j = \text{mean of feature } j
\sigma_j = \text{standard deviation of feature } j
Standardization:
x'_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j}
Important rule:
X_train
→ fit scaler
→ learn μ and σ
→ transform X_train
→ transform X_test using the same μ and σ
Never fit the scaler on the test set because this causes data leakage.
Model Workflow
create model
→ fit
→ learn parameters
→ predict
→ evaluate
Example:
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
fit() uses:
* X_train
* y_train
to learn model parameters.
predict() only needs input features because the parameters have already been learned.
X_test
→ model
→ predicted labels
Evaluation
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
Accuracy:
\text{Accuracy} = \frac{\text{correct predictions}} {\text{total predictions}}
Alternatively:
ppn.score(X_test_std, y_test)
Complete Pipeline
raw data
→ train_test_split
→ X_train / X_test
→ fit preprocessing on X_train
→ transform X_train and X_test
→ model.fit(X_train, y_train)
→ model.predict(X_test)
→ compare predictions with y_test
→ evaluate generalization
 
⸻
 
2. Perceptron Limitation
The Perceptron converges only if the training data is linearly separable.
non-linearly separable data
→ misclassified samples remain
→ weights keep changing
→ Perceptron may not converge
This motivates more powerful classifiers such as Logistic Regression.
 
⸻
 
3. Logistic Regression
Net Input
Logistic Regression first computes:
z = w^Tx+b
where:
* x: input features
* w: weights
* b: bias
* z: net input
Sigmoid Function
The sigmoid function converts z into a probability:
\sigma(z) = \frac{1}{1+e^{-z}}
Its output satisfies:
0 < \sigma(z) < 1
Important values:
z\rightarrow-\infty \Rightarrow \sigma(z)\rightarrow0
z=0 \Rightarrow \sigma(z)=0.5
z\rightarrow+\infty \Rightarrow \sigma(z)\rightarrow1
Probability Interpretation
For binary classification:
P(y=1\mid x)=\sigma(z)
P(y=0\mid x)=1-\sigma(z)
Example:
\sigma(z)=0.8
means:
P(y=1\mid x)=0.8
Classification Rule
The default threshold is:
0.5
Therefore:
\hat y= \begin{cases} 1 & \sigma(z)\ge0.5\\ 0 & \sigma(z)<0.5 \end{cases}
Because:
\sigma(0)=0.5
we have:
\sigma(z)\ge0.5 \iff z\ge0
Therefore, the decision boundary is:
w^Tx+b=0
What Logistic Regression Learns
Logistic Regression learns:
w,\ b
It does not directly learn probabilities.
x
→ z = wᵀx + b
→ sigmoid
→ probability
→ threshold
→ class
 
⸻
 
4. Logistic Loss
For one sample:
p=\sigma(z)
The logistic loss is:
L = -\left[ y\log p + (1-y)\log(1-p) \right]
If y=1
L=-\log p
p → 1
→ loss → 0

p → 0
→ loss → ∞
If y=0
L=-\log(1-p)
p → 0
→ loss → 0

p → 1
→ loss → ∞
Logistic loss strongly penalizes confident wrong predictions.
Likelihood
For the whole training set:
\mathcal{L} = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}
Logistic Regression maximizes likelihood, or equivalently minimizes negative log-likelihood.
The log-likelihood is:
\log\mathcal{L} = \sum_{i=1}^{n} \left[ y_i\log p_i + (1-y_i)\log(1-p_i) \right]
Using the logarithm:
* converts products into sums;
* reduces numerical underflow;
* simplifies differentiation.
Gradient Update
For one sample:
\frac{\partial L}{\partial w_j} = (p-y)x_j
Therefore:
w_j \leftarrow w_j-\eta(p-y)x_j
or equivalently:
w_j \leftarrow w_j+\eta(y-p)x_j
For the bias:
b \leftarrow b+\eta(y-p)
Training workflow:
X
→ z = Xw + b
→ sigmoid
→ probability
→ logistic loss
→ gradient
→ update w and b
 
⸻
 
5. Logistic Regression in Scikit-Learn
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=100.0,
    solver="lbfgs"
)

lr.fit(X_train_std, y_train)
Class Probabilities
lr.predict_proba(X_test_std)
For 3 samples and 3 classes, the output shape is:
(3, 3)
Example:
[0.01, 0.14, 0.85]
means:
P(class 0) = 0.01
P(class 1) = 0.14
P(class 2) = 0.85
Each row sums to approximately:
1
The predicted class is the class with the highest probability:
lr.predict_proba(X_test_std).argmax(axis=1)
Usually, simply use:
lr.predict(X_test_std)
Single-Sample Prediction
Scikit-learn expects:
(n_samples, n_features)
A sample with shape:
(n_features,)
must be reshaped:
X_test_std[0, :].reshape(1, -1)
 
⸻
 
6. Underfitting and Overfitting
Underfitting
The model is too simple.
underfitting
→ high bias
→ poor training performance
→ poor test performance
Overfitting
The model fits the training data too closely.
overfitting
→ high variance
→ excellent training performance
→ poor test performance
Good Generalization
too simple
→ underfitting

appropriate complexity
→ good generalization

too complex
→ overfitting
 
⸻
 
7. Regularization
Regularization reduces overfitting by penalizing large weights.
For L2 regularization:
L = L_{\text{data}} + \frac{\lambda}{2n} \|w\|^2
where:
\|w\|^2 = \sum_j w_j^2
Effect:
large weights
→ larger penalty
→ weights shrink
→ simpler model
→ lower overfitting risk
Regularization Strength
Large \lambda:
λ ↑
→ stronger regularization
→ smaller weights
→ simpler model
Small \lambda:
λ ↓
→ weaker regularization
→ larger weights allowed
→ more complex model
Too much regularization can cause underfitting.
Parameter C
Scikit-learn commonly uses C, which is inversely related to regularization strength.
C ↑
→ weaker regularization
→ larger weights allowed
→ more complex model

C ↓
→ stronger regularization
→ smaller weights
→ simpler model
 
⸻
 
8. Linear Support Vector Machine
Maximum-Margin Classification
A linear SVM searches for the decision boundary with the largest margin.
possible separating boundaries
→ compare margins
→ choose maximum-margin boundary
Decision boundary:
w^Tx+b=0
In two dimensions, this is a line.
In higher dimensions, it is called a hyperplane.
Support Vectors
Support vectors are the training samples closest to the decision boundary.
They determine:
* the position of the boundary;
* the margin width.
Samples far from the boundary usually have much less influence.
Margin
The margin is the distance between the decision boundary and the closest training samples.
support vectors
→ determine margin
→ determine decision boundary
A larger margin often improves generalization.
 
⸻
 
9. Hard Margin and Soft Margin
Hard Margin
Hard-margin SVM requires all training samples to be correctly separated.
It only works well when the data is perfectly linearly separable.
Soft Margin
Real-world data may contain:
* noise;
* outliers;
* overlapping classes.
Soft-margin SVM allows some violations.
allow some errors
→ avoid fitting every sample perfectly
→ improve generalization
Parameter C
C controls the penalty for classification errors.
Large C:
C ↑
→ stronger penalty for errors
→ fewer tolerated violations
→ usually narrower margin
→ higher overfitting risk
Small C:
C ↓
→ weaker penalty for errors
→ more tolerated violations
→ usually wider margin
→ stronger regularization
C does not directly specify margin width.
It controls the trade-off between:
large margin
vs.
classification errors
 
⸻
 
10. Linear SVM in Scikit-Learn
from sklearn.svm import SVC

svm = SVC(
    kernel="linear",
    C=1.0,
    random_state=1
)

svm.fit(X_train_std, y_train)
Important parameters:
* kernel="linear": use a linear decision boundary.
* C: control the penalty for classification errors.
 
⸻
 
11. Logistic Regression vs. Linear SVM
Property	Logistic Regression	Linear SVM
Main objective	Model class probability	Maximize margin
Learned parameters	w,b	w,b
Decision boundary	Linear	Linear
Probability output	Yes	Not naturally
Main focus	Probability	Margin
Important samples	All samples contribute	Mainly support vectors
Summary:
Logistic Regression
→ probability

Linear SVM
→ maximum-margin decision boundary
 
⸻
 
12. Kernel SVM
A linear SVM cannot solve every classification problem.
For nonlinear data such as XOR:
original feature space
→ cannot separate with one straight line
Kernel methods solve this by working in a higher-dimensional feature space.
Feature Mapping
x \rightarrow \phi(x)
Example:
(x_1,x_2) \rightarrow (x_1,x_2,x_1^2+x_2^2)
The data may become linearly separable after transformation.
original space
→ nonlinear separation

higher-dimensional space
→ linear hyperplane
A linear hyperplane in the transformed space corresponds to a nonlinear boundary in the original space.
 
⸻
 
13. Kernel Trick
Explicitly computing \phi(x) may be expensive.
Instead, a kernel directly computes:
k(x^{(i)},x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)})
Therefore:
kernel trick
→ compute similarity in transformed space
→ avoid explicit feature mapping
 
⸻
 
14. RBF Kernel
The RBF kernel is:
k(x^{(i)},x^{(j)}) = \exp \left( -\gamma \|x^{(i)}-x^{(j)}\|^2 \right)
Interpretation:
small distance
→ high similarity
→ kernel value close to 1

large distance
→ low similarity
→ kernel value close to 0
 
⸻
 
15. Gamma
gamma controls how far the influence of each training sample extends.
Small gamma:
gamma ↓
→ wider influence
→ smoother boundary
→ lower complexity
Large gamma:
gamma ↑
→ narrower influence
→ more local behavior
→ more complex boundary
→ higher overfitting risk
Therefore:
\gamma\uparrow \Rightarrow \text{model complexity}\uparrow
 
⸻
 
16. C vs. Gamma in RBF SVM
C and gamma control different aspects of the model.
C
Controls the penalty for classification errors.
large C
→ punish errors strongly
→ lower tolerance for misclassification
Gamma
Controls the influence range of each training sample.
large gamma
→ narrow influence
→ more complex local boundary
 
⸻
 
17. RBF SVM in Scikit-Learn
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    gamma=0.2,
    C=1.0
)

svm.fit(X_train_std, y_train)
Important parameters:
* kernel="rbf": use the RBF kernel.
* gamma: control sample influence range.
* C: control classification error penalty.
Workflow:
nonlinear data
→ kernel similarity
→ higher-dimensional feature space
→ linear hyperplane
→ nonlinear boundary in original space
 
⸻
 
18. Decision Tree
A Decision Tree repeatedly splits the feature space.
A typical split is:
feature <= threshold
The model learns:
* which feature to split;
* which threshold to use;
* how the tree is structured.
Tree structure:
root
→ internal nodes
→ branches
→ leaf
→ prediction
Unlike Logistic Regression or linear SVM, a Decision Tree does not mainly learn w and b.
 
⸻
 
19. Impurity
Impurity measures how mixed the classes are inside a node.
low impurity
→ pure node

high impurity
→ mixed classes
For a completely pure node:
\text{Gini}=0
\text{Entropy}=0
 
⸻
 
20. Entropy
Entropy is:
H = -\sum_i p_i\log_2 p_i
For binary classification:
100% / 0%
→ entropy = 0

50% / 50%
→ maximum entropy
Entropy measures class uncertainty.
 
⸻
 
21. Gini Impurity
Gini impurity is:
G = 1-\sum_i p_i^2
For binary classification:
100% / 0%
→ Gini = 0

50% / 50%
→ Gini = 0.5
Gini and Entropy usually produce similar Decision Trees.
 
⸻
 
22. Information Gain
Information Gain measures how much a split reduces impurity.
\text{Information Gain} = \text{parent impurity} - \text{weighted child impurity}
Decision Trees prefer splits with larger Information Gain.
better split
→ larger impurity reduction
→ purer child nodes
 
⸻
 
23. Tree Depth
Tree depth controls model complexity.
depth ↑
→ more splits
→ more complex decision regions
→ closer fit to training data
→ higher overfitting risk
max_depth limits tree depth.
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=1
)
Small max_depth:
simpler model
→ lower variance
Large max_depth:
more complex model
→ higher overfitting risk
 
⸻
 
24. Decision Tree and Feature Scaling
Decision Trees usually do not require feature scaling.
They mainly depend on:
ordering
+
threshold comparisons
Example:
x <= 5
After scaling, the threshold value changes, but the ordering of samples usually remains the same.
Therefore:
Decision Tree
→ ordering matters
→ scaling usually unnecessary
 
⸻
 
25. Random Forest
A Random Forest combines many Decision Trees.
many different trees
→ combine predictions
→ more stable model
A single deep Decision Tree can have high variance.
Random Forest reduces variance by making the trees different and combining them.
 
⸻
 
26. Bootstrap Sampling
Each tree is trained on a bootstrap sample.
Bootstrap sampling means:
random sampling with replacement
Example:
Original dataset:
A B C D
Possible bootstrap sample:
A A C D
Therefore:
* some samples may appear multiple times;
* some samples may not appear at all.
 
⸻
 
27. Random Feature Subsets
At each node, Random Forest considers only a random subset of features.
all features
→ randomly choose subset
→ find best split inside subset
This makes different trees less correlated.
 
⸻
 
28. Random Forest Algorithm
1. Draw bootstrap sample
2. Grow Decision Tree
3. At each node, choose random feature subset
4. Find best split within subset
5. Repeat for many trees
6. Combine tree predictions
For classification, predictions are usually combined using majority vote.
Example:
Tree 1 → 0
Tree 2 → 1
Tree 3 → 1
Tree 4 → 1
Tree 5 → 0

Final prediction → 1
 
⸻
 
29. Random Forest in Scikit-Learn
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=25,
    random_state=1,
    n_jobs=2
)
Important parameters:
* n_estimators: number of Decision Trees.
* random_state: reproducible randomness.
* n_jobs: number of CPU cores used.
 
⸻
 
30. Parametric vs. Non-Parametric Models
Parametric Models
Parametric models learn a fixed-size set of parameters.
Examples:
* Perceptron
* Logistic Regression
* Linear SVM
Typical parameters:
w,\ b
parametric model
→ fixed-size learned parameter set
Non-Parametric Models
Non-parametric models are not represented by a fixed-size parameter vector.
Examples:
* Decision Tree
* Random Forest
* KNN
non-parametric model
→ model complexity can grow with data
Non-parametric does not mean “no hyperparameters.”
 
⸻
 
31. K-Nearest Neighbors
KNN predicts a new sample using nearby training samples.
It does not learn a fixed parameter vector such as w and b.
Prediction workflow:
new sample
→ calculate distances
→ find k nearest training samples
→ majority vote
→ predicted class
 
⸻
 
32. Lazy Learning
KNN is called a lazy learner.
Training:
little computation
→ mainly store training data
Prediction:
calculate distances
→ find neighbors
→ vote
Therefore, most computation happens during prediction.
 
⸻
 
33. Choosing k
Small k:
k ↓
→ more flexible boundary
→ more sensitive to noise
→ higher complexity
→ higher overfitting risk
Large k:
k ↑
→ smoother boundary
→ less sensitive to individual samples
→ lower complexity
→ possible underfitting
 
⸻
 
34. Distance Metrics
KNN commonly uses Minkowski distance.
p=2:
Euclidean distance
p=1:
Manhattan distance
Example:
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    p=2,
    metric="minkowski"
)
 
⸻
 
35. Feature Scaling in KNN
Feature scaling is very important for KNN because KNN directly uses distance.
Example:
height: 150–190
income: 0–10,000,000
Without scaling:
income dominates the distance
Standardization makes feature scales more comparable.
Therefore:
KNN
→ distance matters
→ scaling is very important
 
⸻
 
36. Curse of Dimensionality
KNN becomes less effective in very high-dimensional spaces.
dimensions ↑
→ feature space becomes larger
→ training samples become sparse
→ nearest neighbors become farther away
→ distance becomes less informative
This is called the curse of dimensionality.
 
⸻
 
37. Model Comparison
Model	Main Learned Information	Main Idea	Scaling
Perceptron	w,b	Linear classification	Usually important
Logistic Regression	w,b	Probability + logistic loss	Usually important
Linear SVM	w,b	Maximum-margin boundary	Usually important
RBF SVM	Support-vector relationships	Nonlinear kernel boundary	Very important
Decision Tree	Features, thresholds, tree structure	Recursive splitting	Usually unnecessary
Random Forest	Many Decision Trees	Ensemble voting	Usually unnecessary
KNN	Training samples	Distance + neighbor voting	Very important
 
⸻
 
38. Model Complexity
Important relationships:
Regularization:
C ↓
→ stronger regularization
→ simpler model
RBF SVM:
gamma ↑
→ more complex boundary
Decision Tree:
max_depth ↑
→ model complexity ↑
KNN:
k ↓
→ model complexity ↑
 
⸻
 
39. Feature Scaling Summary
Logistic Regression
→ scaling usually important

SVM
→ scaling important

RBF SVM
→ scaling especially important because distance matters

Decision Tree
→ scaling usually unnecessary

Random Forest
→ scaling usually unnecessary

KNN
→ scaling very important because distance matters
 
⸻
 
40. Chapter 3 Core Takeaways
Scikit-Learn Workflow
split
→ preprocess training data
→ apply same transformation to test data
→ fit model
→ predict
→ evaluate
Logistic Regression
X
→ z = Xw + b
→ sigmoid
→ probability
→ logistic loss
→ learn w and b
Linear SVM
linear decision boundary
→ support vectors
→ maximize margin
→ C controls error penalty
Kernel SVM
nonlinear data
→ kernel trick
→ higher-dimensional feature space
→ linear hyperplane there
→ nonlinear boundary in original space
Decision Tree
feature + threshold
→ reduce impurity
→ recursive splits
→ leaf
→ prediction
Random Forest
bootstrap samples
→ random feature subsets
→ many Decision Trees
→ majority vote
→ lower variance
KNN
store training data
→ calculate distance
→ find k nearest neighbors
→ majority vote
Generalization
too simple
→ underfitting
→ high bias

too complex
→ overfitting
→ high variance

appropriate complexity
→ good generalization
