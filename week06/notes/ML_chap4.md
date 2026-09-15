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