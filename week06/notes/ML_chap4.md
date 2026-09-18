# MLwPS Chapter 4 — Missing Data

## 1. Why Missing Data Matters

Real-world datasets often contain missing values because of:

- Data collection errors
- Missing measurements
- Empty input fields
- Unavailable information

Missing values are commonly represented as:

```python
NaN
```

Many machine learning algorithms cannot handle missing values directly.

Therefore, missing values should usually be handled before model training.

---

## 2. Identifying Missing Values

Use:

```python
df.isnull()
```

This checks whether each value is missing.

- `True` → missing value
- `False` → existing value

To count missing values in each feature:

```python
df.isnull().sum()
```

Example:

```text
A    0
B    0
C    1
D    1
```

This means:

- Feature `C` has 1 missing value.
- Feature `D` has 1 missing value.

---

## 3. Three Ways to Handle Missing Values

The main options are:

```text
Missing Value
    ↓
1. Drop samples
2. Drop features
3. Impute missing values
```

---

## 4. Dropping Samples

To remove rows containing missing values:

```python
df.dropna(axis=0)
```

Meaning:

```text
axis=0
→ remove rows
→ remove samples
```

This may be reasonable when:

- Only a small number of samples contain missing values.
- Removing them does not significantly reduce the dataset.

---

## 5. Dropping Features

To remove columns containing missing values:

```python
df.dropna(axis=1)
```

Meaning:

```text
axis=1
→ remove columns
→ remove features
```

This may be reasonable when:

- A feature contains many missing values.
- The feature is not important.

---

## 6. Why Not Always Use `dropna()`?

Dropping data may cause information loss.

Example:

```text
10,000 samples
3 samples contain missing values
```

Dropping 3 samples may be acceptable.

However:

```text
10,000 samples
4,000 samples contain missing values
```

Dropping 4,000 samples may remove too much useful information.

Therefore:

```text
Few missing samples
→ dropping may be reasonable

Many missing samples
→ imputation may be better
```

---

## 7. Imputing Missing Values

Imputation means replacing missing values with estimated values.

One common method is mean imputation.

Example:

```text
Feature C:

3
NaN
12
```

Calculate the mean:

```text
(3 + 12) / 2 = 7.5
```

Replace the missing value:

```text
3
7.5
12
```

Therefore:

```text
Mean Imputation
→ replace each missing value
→ with the mean of its feature
```

---

## 8. `SimpleImputer`

Scikit-learn provides the `SimpleImputer` class:

```python
from sklearn.impute import SimpleImputer
```

Create a mean imputer:

```python
imputer = SimpleImputer(strategy="mean")
```

Common strategies include:

```text
mean
median
most_frequent
```

---

## 9. Understanding `fit()`

Use:

```python
imputer.fit(X_train)
```

`fit()` learns information from the training data.

For:

```python
SimpleImputer(strategy="mean")
```

it learns the mean of each feature in `X_train`.

Example:

```text
Feature 1 → training mean
Feature 2 → training mean
Feature 3 → training mean
```

Important:

```text
fit()
→ learns replacement values
→ does not replace missing values
```

---

## 10. Understanding `transform()`

Use:

```python
X_train_imp = imputer.transform(X_train)
```

`transform()` applies the information learned by `fit()`.

Therefore:

```text
fit()
→ learn

transform()
→ apply
```

For mean imputation:

```text
fit(X_train)
→ learn each feature's mean

transform(X_train)
→ replace missing values using those means
```

---

## 11. Correct Train-Test Workflow

The imputer should only learn from the training data.

Correct workflow:

```python
imputer.fit(X_train)

X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

Conceptually:

```text
X_train
    ↓
fit()
    ↓
learn feature means
    ↓
    ├── transform(X_train)
    │
    └── transform(X_test)
```

The test set uses the means learned from `X_train`.

```text
X_train means
    ↓
used for X_train
used for X_test
```

The test set does not calculate its own means.

---

## 12. Why Only Fit on `X_train`?

The test set should represent unseen data.

Therefore:

```python
imputer.fit(X_test)
```

should not be used during model development.

The correct rule is:

```text
fit on X_train
transform X_train
transform X_test
```

---

## 13. Data Leakage

Suppose we impute the entire dataset before splitting:

```text
Entire Dataset
    ↓
calculate feature means
    ↓
impute missing values
    ↓
train-test split
```

The calculated means contain information from the future test set.

This causes:

```text
Data Leakage
```

Correct workflow:

```text
Dataset
    ↓
train-test split
    ↓
X_train        X_test
    ↓
fit preprocessing on X_train
    ↓
transform X_train
transform X_test
```

---

## 14. General Scikit-Learn Transformer API

Many preprocessing tools follow the same pattern:

```text
fit()
→ learn parameters from training data

transform()
→ apply learned parameters
```

### StandardScaler

```python
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

`StandardScaler.fit()` learns:

```text
mean
standard deviation
```

### SimpleImputer

```python
imputer.fit(X_train)

X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

`SimpleImputer.fit()` learns:

```text
replacement values
such as feature means
```

The general rule is:

```text
fit on training data
transform training data
transform test data
```

---

## 15. Transformer vs Predictor

A preprocessing transformer commonly uses:

```text
fit()
transform()
```

Example:

```python
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
```

A machine learning model commonly uses:

```text
fit()
predict()
```

Example:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Therefore:

```text
Transformer
→ fit()
→ transform()

Predictive Model
→ fit()
→ predict()
```

---

## 16. Key Takeaways

### Missing Data

```text
Missing Value
    ↓
1. Drop samples
2. Drop features
3. Impute missing values
```

### Drop vs Impute

```text
Removing little data
→ dropping may be acceptable

Removing too much useful data
→ consider imputation
```

### `fit()` vs `transform()`

```text
fit()
→ learn from data

transform()
→ apply what was learned
```

### Correct Preprocessing Workflow

```text
X_train
    ↓
fit()

X_train
    ↓
transform()

X_test
    ↓
transform()
```

### Most Important Rule

```text
fit on X_train only
transform X_train
transform X_test
```

This prevents data leakage and keeps the test set truly unseen.



---

# Categorical Data Encoding

## 1. Categorical Data

Categorical features contain values that represent categories rather than continuous numerical quantities.

There are two main types of categorical features:

- **Ordinal features**: categories have a natural order.
- **Nominal features**: categories do not have a natural order.

---

## 2. Ordinal Features

Ordinal features have a natural ordering.

Example:

    M < L < XL

Because the categories have an order, they can be mapped to numerical values.

    size_mapping = {
        'M': 1,
        'L': 2,
        'XL': 3
    }

    df['size'] = df['size'].map(size_mapping)

Result:

    M  -> 1
    L  -> 2
    XL -> 3

The numerical encoding should preserve the natural ordering of the categories.

**Core rule:**

    Ordinal feature
    -> Has natural order
    -> Ordinal mapping

---

## 3. Mapping Ordinal Features

The `map()` method can convert categorical values according to a manually defined dictionary.

    df['size'] = df['size'].map(size_mapping)

The mapping dictionary tells pandas how each category should be converted.

The ordering must usually be defined manually because the program does not automatically know that:

    M < L < XL

### Reverse Mapping

The numerical values can also be converted back to their original categories.

    inv_size_mapping = {
        v: k for k, v in size_mapping.items()
    }

    df['size'].map(inv_size_mapping)

Result:

    1 -> M
    2 -> L
    3 -> XL

---

## 4. Nominal Features

Nominal features do not have a natural ordering.

Example:

    red
    green
    blue

It is usually inappropriate to encode them directly as:

    blue  = 0
    green = 1
    red   = 2

This encoding introduces an artificial numerical relationship:

    red > green > blue

However, no such ordering exists between colors.

A machine learning model may incorrectly use this artificial numerical relationship.

Therefore, nominal features are commonly encoded using **one-hot encoding**.

**Core rule:**

    Nominal feature
    -> No natural order
    -> One-hot encoding

---

## 5. Encoding Class Labels

Class labels are target values (`y`) rather than ordinary input features (`X`).

Example:

    class1
    class2

They can be encoded as integers:

    class1 -> 0
    class2 -> 1

These integers are category identifiers. They do not imply that one class is larger or smaller than another.

### Manual Mapping

    class_mapping = {
        'class1': 0,
        'class2': 1
    }

    df['classlabel'] = df['classlabel'].map(class_mapping)

### LabelEncoder

Scikit-learn provides `LabelEncoder` for encoding class labels.

    from sklearn.preprocessing import LabelEncoder

    class_le = LabelEncoder()

    y = class_le.fit_transform(df['classlabel'].values)

The main methods are:

- `fit()` learns the available class labels.
- `transform()` converts class labels into integers.
- `fit_transform()` performs both operations.
- `inverse_transform()` converts integers back to the original class labels.

Example:

    class_le.inverse_transform(y)

**Core rule:**

    Class label (y)
    -> Integer encoding
    -> LabelEncoder

---

## 6. One-Hot Encoding

One-hot encoding is commonly used for nominal features.

Suppose a feature contains three categories:

    red
    green
    blue

One-hot encoding creates one binary feature for each category.

    color   red   green   blue

    red      1      0      0
    green    0      1      0
    blue     0      0      1

This prevents the model from interpreting an artificial ordering between categories.

For `n` categories, full one-hot encoding creates `n` binary features.

Example:

    3 categories
    -> 3 one-hot features

**Core rule:**

    Nominal feature
    -> One category becomes one binary feature
    -> No artificial ordering

---

## 7. OneHotEncoder

Scikit-learn provides `OneHotEncoder` for one-hot encoding.

    from sklearn.preprocessing import OneHotEncoder

    color_ohe = OneHotEncoder()

Conceptually:

    color
      |
      v
    OneHotEncoder
      |
      v
    color_blue
    color_green
    color_red

Each category becomes a separate binary feature.

---

## 8. ColumnTransformer

A dataset may contain different types of features.

Example:

    color    size    price
    green      1     10.1
    red        2     13.5
    blue       3     15.3

Suppose only `color` needs one-hot encoding.

`ColumnTransformer` can transform selected columns while leaving other columns unchanged.

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    c_transf = ColumnTransformer([
        ('onehot', OneHotEncoder(), [0]),
        ('nothing', 'passthrough', [1, 2])
    ])

Conceptually:

    color -> OneHotEncoder
    size  -> passthrough
    price -> passthrough

`passthrough` means that the selected columns are left unchanged.

**Core rule:**

    ColumnTransformer
    -> Transform selected columns
    -> Leave other columns unchanged with passthrough

---

## 9. pandas get_dummies()

Pandas provides `get_dummies()` as another convenient way to perform one-hot encoding.

    pd.get_dummies(df[['price', 'color', 'size']])

For the nominal feature `color`, it can create:

    color_blue
    color_green
    color_red

Both `OneHotEncoder` and `pd.get_dummies()` can be used to create dummy features from nominal categorical data.

---

## 10. Dropping a Redundant Dummy Feature

Suppose a nominal feature contains three categories:

    blue
    green
    red

Full one-hot encoding produces:

    blue   green   red

      1      0      0
      0      1      0
      0      0      1

However, one feature can be removed without losing category information.

For example, keep only:

    green   red

      0      0    -> blue
      1      0    -> green
      0      1    -> red

If both `green` and `red` are `0`, the category must be `blue`.

Therefore, the third dummy feature is redundant.

With pandas:

    pd.get_dummies(
        df[['price', 'color', 'size']],
        drop_first=True
    )

With scikit-learn:

    OneHotEncoder(drop='first')

**Core rule:**

    n categories
    -> n one-hot features

    n categories with drop_first
    -> n - 1 dummy features

The dropped category can be inferred from the remaining features.

---

## 11. Optional Encoding for Ordinal Features

Sometimes an ordinal feature has a known ordering, but the numerical distances between categories are unknown.

Example:

    M < L < XL

A simple mapping could be:

    M  = 1
    L  = 2
    XL = 3

This preserves the ordering, but it also suggests equal numerical distances:

    L - M = XL - L

This assumption may not always be appropriate.

An alternative is threshold encoding.

For example, create two features:

    x > M
    x > L

Then:

    M  -> 0, 0
    L  -> 1, 0
    XL -> 1, 1

This preserves the ordering:

    M < L < XL

without requiring equal numerical distances between the categories.

---

## 12. Other Encoding Methods

One-hot encoding is one of the most common methods for nominal categorical features.

Other encoding methods include:

- Binary encoding
- Count encoding
- Frequency encoding

These methods can be useful when a categorical feature has a large number of unique categories.

---

## 13. Summary

### Ordinal Features

- Have a natural ordering.
- Can be mapped to numerical values.
- The numerical encoding should preserve the original ordering.

Example:

    M < L < XL

    M  -> 1
    L  -> 2
    XL -> 3

### Nominal Features

- Do not have a natural ordering.
- Should not normally be directly encoded as ordered integers.
- One-hot encoding is commonly used.

Example:

    red
    green
    blue

    -> one-hot encoding

### Class Labels

- Represent the target variable `y`.
- Can be encoded as integers.
- `LabelEncoder` can perform the conversion.

Example:

    class1 -> 0
    class2 -> 1

### One-Hot Encoding

- Creates separate binary features for categories.
- Avoids introducing artificial ordering.
- `n` categories normally produce `n` one-hot features.

### Dropping the First Dummy Feature

- One dummy feature can be redundant.
- `n` categories can be represented using `n - 1` dummy features.
- The dropped category can be inferred from the remaining features.

### ColumnTransformer

- Applies transformations to selected columns.
- `passthrough` leaves selected columns unchanged.

---

## 14. Core Rules to Remember

    Ordinal feature
    -> Has natural order
    -> Ordinal mapping

    Nominal feature
    -> Has no natural order
    -> One-hot encoding

    Class label (y)
    -> Integer encoding
    -> LabelEncoder

    n categories
    -> n one-hot features

    drop_first=True
    -> n - 1 dummy features

    passthrough
    -> Leave columns unchanged


---

# Data Preprocessing and Regularization

## 1. Train-Test Split

A dataset should be divided into separate training and test sets.

- **Training set**: used to learn model parameters.
- **Test set**: used to evaluate performance on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y
)
```

### Important Parameters

- `test_size=0.3`
  - 70% training data
  - 30% test data

- `random_state=0`
  - Makes the random split reproducible.

- `stratify=y`
  - Preserves approximately the same class proportions in the training and test sets.

### Train-Test Ratio

There is no single best train-test ratio.

Common choices include:

- 60/40
- 70/30
- 80/20

For very large datasets, a larger proportion can often be used for training while still keeping enough test samples for reliable evaluation.

---

## 2. Feature Scaling

Features can have very different numerical scales.

For example:

```text
Feature 1: 1–10
Feature 2: 1–100,000
```

Without scaling, features with larger numerical values may dominate distance calculations or optimization.

Scaling is especially important for algorithms such as:

- Logistic Regression
- SVM
- KNN
- Gradient Descent-based models

Decision Trees and Random Forests usually do not require feature scaling because their decisions are based mainly on feature ordering and split thresholds.

---

## 3. Min-Max Normalization

Min-max scaling usually transforms a feature into the range `[0, 1]`.

### Formula

$begin:math:display$
x\_\{norm\} \= \\frac\{x\-x\_\{min\}\}\{x\_\{max\}\-x\_\{min\}\}
$end:math:display$

### Example

```text
Original:
0  1  2  3  4  5

Min-Max:
0  0.2  0.4  0.6  0.8  1
```

### Scikit-Learn

```python
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
```

Min-max scaling depends on the minimum and maximum values, so it can be sensitive to outliers.

---

## 4. Standardization

Standardization centers and scales each feature.

### Formula

$begin:math:display$
x\_\{std\} \= \\frac\{x\-\\mu\}\{\\sigma\}
$end:math:display$

where:

- $begin:math:text$\\mu$end:math:text$ = mean of the feature
- $begin:math:text$\\sigma$end:math:text$ = standard deviation of the feature

After standardization:

```text
mean ≈ 0
standard deviation ≈ 1
```

### Important

Standardization does **not** make the data normally distributed.

```text
Skewed distribution
        ↓
StandardScaler
        ↓
Mean ≈ 0, Std ≈ 1
        ↓
Still skewed
```

It changes the center and scale, not the basic shape of the distribution.

---

## 5. StandardScaler in Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```

### What Does `fit()` Learn?

```python
sc.fit(X_train)
```

learns the training-set statistics for each feature:

```text
training mean
training standard deviation
```

These same statistics must be used to transform both the training and test sets.

```text
X_train
→ use μ_train and σ_train

X_test
→ use μ_train and σ_train

future unseen data
→ use μ_train and σ_train
```

---

## 6. Avoiding Data Leakage

The scaler should be fitted only on the training data.

Correct:

```python
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

Or:

```python
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```

Incorrect:

```python
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
```

The test set must not be used to learn preprocessing parameters.

### General Rule

```text
Anything that learns information from data
should normally learn it from the training data only.
```

Examples:

```text
SimpleImputer.fit()
→ learns replacement statistics

StandardScaler.fit()
→ learns mean and standard deviation

Model.fit()
→ learns model parameters
```

---

## 7. RobustScaler

`RobustScaler` can be useful when a dataset contains significant outliers.

StandardScaler mainly uses:

```text
mean + standard deviation
```

MinMaxScaler uses:

```text
minimum + maximum
```

RobustScaler mainly uses:

```text
median + quartiles
```

Therefore:

```text
many outliers
→ mean/min/max may be strongly affected
→ RobustScaler can be more robust
```

---

## 8. Selecting Meaningful Features

Overfitting often occurs when a model becomes too complex and fits the training data too closely.

```text
High training performance
+
Poor test performance
→ possible overfitting
```

Possible ways to reduce overfitting include:

- Collect more training data.
- Apply regularization.
- Choose a simpler model.
- Reduce the number of features.

Regularization reduces model complexity by penalizing large weights.

---

## 9. L2 Regularization

L2 regularization penalizes the squared magnitude of the weights.

### L2 Penalty

$begin:math:display$
\\lambda \\sum\_j w\_j\^2
$end:math:display$

The objective becomes:

$begin:math:display$
Loss \+ \\lambda \\sum\_j w\_j\^2
$end:math:display$

where:

- `Loss` measures prediction error.
- $begin:math:text$\\lambda$end:math:text$ controls regularization strength.

### Effect

```text
Larger λ
→ stronger penalty
→ smaller weights
→ lower model complexity
→ potentially less overfitting
```

L2 usually shrinks weights toward zero but does not usually make many weights exactly zero.

---

## 10. Geometric Interpretation of L2

Without regularization:

```text
Goal:
minimize training loss
```

With L2 regularization:

```text
Goal:
minimize loss + weight penalty
```

The L2 constraint has a circular shape in two dimensions.

The optimal solution balances:

```text
small prediction error
+
small weights
```

Increasing regularization strength makes large weights increasingly expensive.

---

## 11. L1 Regularization

L1 regularization penalizes the absolute magnitude of the weights.

### L1 Penalty

$begin:math:display$
\\lambda \\sum\_j \|w\_j\|
$end:math:display$

The objective becomes:

$begin:math:display$
Loss \+ \\lambda \\sum\_j \|w\_j\|
$end:math:display$

### Effect

L1 can force some weights to become exactly zero.

Example:

```text
w1 = 0
w2 ≠ 0
w3 = 0
w4 ≠ 0
```

This is called a **sparse solution**.

---

## 12. Why L1 Produces Sparse Solutions

In two dimensions:

```text
L2 constraint → circular shape

L1 constraint → diamond shape
```

The L1 diamond has sharp corners on the coordinate axes.

The optimal solution is therefore more likely to occur at a point where:

```text
w1 = 0
```

or:

```text
w2 = 0
```

This explains geometrically why L1 regularization tends to produce sparse solutions.

---

## 13. L1 as Feature Selection

For a linear model:

$begin:math:display$
z \= w\_1x\_1 \+ w\_2x\_2 \+ \\cdots \+ w\_mx\_m \+ b
$end:math:display$

If:

$begin:math:display$
w\_j \= 0
$end:math:display$

then:

$begin:math:display$
w\_jx\_j \= 0
$end:math:display$

Therefore, feature $begin:math:text$x\_j$end:math:text$ has no effect on the model prediction.

```text
L1 regularization
→ some weights become exactly 0
→ corresponding features have no effect
→ automatic feature selection
```

This makes L1 useful when a dataset contains many potentially irrelevant features.

---

## 14. L1 vs L2

| Property | L1 | L2 |
|---|---|---|
| Penalty | $begin:math:text$\\sum \|w\_j\|$end:math:text$ | $begin:math:text$\\sum w\_j\^2$end:math:text$ |
| Constraint shape | Diamond | Circle |
| Shrinks weights | Yes | Yes |
| Weights can become exactly 0 | Common | Usually not |
| Sparse solution | Yes | Usually no |
| Feature selection | Yes | Not directly |

### Core Difference

```text
L2
→ shrink weights toward 0
→ usually not exactly 0

L1
→ shrink weights
→ some can become exactly 0
→ sparse solution
→ feature selection
```

---

## 15. Regularization in Logistic Regression

Example using L1 regularization:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    penalty="l1",
    C=1.0,
    solver="liblinear"
)

lr.fit(X_train_std, y_train)
```

### Model Parameters

```python
lr.coef_
```

corresponds to the learned weights:

$begin:math:display$
w
$end:math:display$

```python
lr.intercept_
```

corresponds to the learned bias:

$begin:math:display$
b
$end:math:display$

Therefore:

```text
coef_      → weights w
intercept_ → bias b
```

---

## 16. The Regularization Parameter C

In scikit-learn Logistic Regression, `C` is inversely related to regularization strength.

```text
C ↓
→ stronger regularization

C ↑
→ weaker regularization
```

Conceptually:

$begin:math:display$
C \\propto \\frac\{1\}\{\\lambda\}
$end:math:display$

Therefore:

```text
λ ↑ → regularization ↑
C ↓ → regularization ↑
```

With L1 regularization:

```text
very small C
→ strong regularization
→ more weights may become 0
→ sparser model
```

Example:

```text
C = 0.0001
→ very strong regularization

C = 100
→ much weaker regularization
```

---

## 17. Complete Machine Learning Pipeline

The concepts from pp.117–127 can be combined into one workflow:

```text
Raw dataset
    ↓
Train-test split
    ↓
Fit preprocessing on training data only
    ↓
Transform training and test data
    ↓
Fit model on training data
    ↓
Regularization controls model complexity
    ↓
Evaluate on unseen test data
```

Example:

```python
# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y
)

# 2. Learn scaling parameters from training data
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 3. Train the model
lr = LogisticRegression(
    penalty="l1",
    C=1.0,
    solver="liblinear"
)

lr.fit(X_train_std, y_train)

# 4. Evaluate on unseen data
test_accuracy = lr.score(X_test_std, y_test)
```

---

## 18. Key Takeaways

```text
Train set
→ learn preprocessing parameters and model parameters

Test set
→ evaluate generalization
→ never use it to fit preprocessing
```

```text
StandardScaler
→ mean ≈ 0
→ standard deviation ≈ 1
→ does not make data normally distributed
```

```text
MinMaxScaler
→ usually maps features to [0, 1]
```

```text
RobustScaler
→ uses robust statistics such as median and quartiles
→ useful when outliers are important
```

```text
L2
→ squared-weight penalty
→ smaller weights
→ reduced model complexity
```

```text
L1
→ absolute-weight penalty
→ some weights exactly 0
→ sparse solution
→ feature selection
```

```text
C ↓
→ stronger regularization

C ↑
→ weaker regularization
```

### Final Mental Model

```text
Good preprocessing
        +
No data leakage
        +
Appropriate feature scaling
        +
Regularization
        ↓
Better control of model complexity
        ↓
Better generalization to unseen data
```