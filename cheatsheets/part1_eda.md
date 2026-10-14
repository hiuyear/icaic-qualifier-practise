# ICAIC 2026 — Part 1: Exploratory Data Analysis (EDA)

> **Goal:** In the first 10–20 minutes of an unfamiliar tabular competition, understand the prediction task, data quality, feature types, target distribution, and appropriate validation strategy. This is a general-purpose reference, not just a Titanic guide.

## 1. Quick-start workflow

1. Read the competition's **target definition, metric, prediction format, and allowed data**.
2. Load train and competition test data; note their shapes and columns.
3. Separate target (`y`) from features (`X`). Do **not** confuse competition test data with a local validation set.
4. Classify features by **meaning**, not just Pandas dtype.
5. Inspect missingness, distributions, outliers, high-cardinality columns, duplicate records, and label balance.
6. Select an appropriate validation strategy before fitting transformations.
7. Build a simple baseline quickly; return to detailed EDA only where it can improve decisions.

## 2. Core Pandas commands

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train.head()                       # First five rows; use .head(10) for ten
train.tail()                       # Last five rows
train.shape                        # (rows, columns) — attribute, no ()
train.columns                      # Column names — attribute
train.dtypes                       # Storage types — attribute
train.info()                       # Non-null counts, types, memory summary
train.describe()                   # Numerical summary by default
train.describe(include="all")     # Also summarize non-numerical columns
train.isna().sum()                 # Missing counts per column
train.isna().mean()                # Missing fractions per column
train["target"].value_counts()            # Counts by label/category
train["target"].value_counts(normalize=True) # Proportions
train["feature"].nunique()         # Number of unique non-null values
train["feature"].unique()          # Distinct values (including missing)
```

**Method vs attribute:** `df.shape` and `df.dtypes` are attributes; `df.head()`, `df.info()`, `df.isna().sum()` call methods. `df.info()` prints its result and returns `None`: avoid `print(df.info())`.

**Selection:** `df["Age"]` is a Series (1D); `df[["Age"]]` is a DataFrame (2D). Many sklearn transformers prefer 2D input.

**Pandas dtype ≠ semantic feature type:** In newer Pandas versions strings may show as `str`, rather than `object`.

## 3. Identify the task and target

| Task | Target | Typical example | Common metrics |
|---|---|---|---|
| Binary classification | Two classes | Fraud vs not fraud | Accuracy, F1, ROC-AUC, log loss, PR-AUC |
| Multiclass classification | More than two classes | Product category | Accuracy, macro F1, multiclass log loss |
| Regression | Continuous number | Sale price | MAE, RMSE, R² (only if requested) |
| Time series / forecasting | Future value(s) | Next week's demand | Forecasting metric, often MAE/RMSE/SMAPE |
| Ranking | Ordered relevance | Search results | NDCG, MAP, other ranking metrics |

**Optimize the metric specified by the competition, not whichever metric is easiest.** Check whether submissions require labels (`predict`) or probabilities (`predict_proba`).

```python
target = "Survived"   # Example: replace with actual target
X = train.drop(columns=[target])
y = train[target]
print(X.shape, y.shape)
```

Look for train/test schema differences: target in labeled train but missing from competition test is normal. Extra/missing feature columns or incompatible types need investigation.

## 4. Classify features by meaning

| Semantic type | Example | Initial thought |
|---|---|---|
| Continuous numerical | Price, age, temperature | Imputation; scaling for scale-sensitive models |
| Discrete count | Number of visits | Numeric or engineered categorical, depending on meaning |
| Nominal categorical | City, merchant, colour | One-hot, native categorical, hashing, etc. |
| Ordinal categorical | Low/medium/high, service tier | Ordinal or one-hot; order may be meaningful but distances arbitrary |
| Binary categorical | Yes/no | Binary or one-hot encoding |
| Identifier | PassengerId, transaction ID | Usually omit in baseline; may contain useful structure |
| High-cardinality categorical | UserID, postal code | Avoid naively creating huge one-hot matrices |
| Date/time | Timestamp, signup date | Time parts, durations, lags; inspect temporal leakage |
| Free text | Reviews, descriptions | TF-IDF, tokenization, embeddings |
| Image / audio | Photos, waveforms | Data-specific loaders and transforms |

**A numeric dtype is not proof of a continuous quantity:** `Pclass=1/2/3`, ZIP code, and customer IDs are numeric-looking categories/codes. The label may be stored as an integer but still be categorical.

## 5. Missingness analysis

```python
missing_count = train.isna().sum()
missing_fraction = train.isna().mean()
missing_report = pd.DataFrame({
    "count": missing_count,
    "fraction": missing_fraction
}).sort_values("fraction", ascending=False)
missing_report.head(20)
```

Ask whether missingness is rare, widespread, concentrated in one subset, or itself predictive. A feature with >90% missing values may still carry signal; consider validation results rather than a rigid deletion threshold. Missingness can differ between train and competition test.

**Do not fill missing values based on the full training+validation pool before splitting.** Learn imputation statistics inside each training fold.

## 6. Numerical distributions and outliers

```python
train.describe()
train.select_dtypes(include="number").hist(figsize=(12, 8))

# Optional targeted checks
train["Fare"].quantile([0, .01, .25, .5, .75, .99, 1])
train["Fare"].value_counts().head()
```

Look for implausible values, heavy skew, extreme outliers, constant/near-constant columns, and different scales. `mean` versus `median` indicates potential skew; scaling is relevant for SVM/KNN/logistic regression, generally not for trees. Decide transformations empirically, not mechanically.

## 7. Categories, cardinality, and class balance

```python
train["Sex"].value_counts(dropna=False)
train["Embarked"].value_counts(normalize=True, dropna=False)
train["Ticket"].nunique()

# Classification target distribution
train[target].value_counts(dropna=False)
train[target].value_counts(normalize=True, dropna=False)
```

Class imbalance matters: if 99.5% are negative, always predicting negative achieves 99.5% accuracy but misses every positive. Consider precision, recall, F1, PR-AUC, balanced accuracy, or class weights as dictated by the competition and application. For regression, inspect target skew and extreme values instead.

## 8. Detect suspicious information and leakage

Ask:
- Is this feature actually available at **prediction time**?
- Was a feature computed using the target or information from the future?
- Does the same customer, patient, device, image source, or event appear on both sides of a random split?
- Does a timestamp create a chronological prediction task?
- Was the dataset pre-sorted or grouped in a way that makes splitting tricky?
- Does the competition specify a particular holdout period or group split?

Duplicated observations and nearly identical records can inflate a random-split score. Don't automatically drop them without considering what a real-world prediction is meant to generalize to.

## 9. Choose validation before model fitting

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # Ordinarily useful for classification, not continuous regression
)
```

| Situation | Better starting validation |
|---|---|
| Independent binary/multiclass rows | Stratified random split or StratifiedKFold |
| Ordinary regression | Random split / KFold if observations independent |
| Chronologically dependent forecasting | Chronological holdout / TimeSeriesSplit |
| Repeated subjects/entities | GroupShuffleSplit / GroupKFold |
| Rare classes | Stratify when possible; ensure class presence in folds |

**The competition's unlabeled test.csv is not your local validation set.** Reserve validation labels for evaluating and choosing approaches. Frequent tuning against the same validation set can itself overfit model selection.

## 10. Short reusable EDA template

```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
target = "TARGET"  # change this

print("Train/test:", train.shape, test.shape)
print("Train columns:", list(train.columns))
train.info()
print("Missing fraction:\n", train.isna().mean().sort_values(ascending=False))
print("Numerical summary:\n", train.describe())
print("Target distribution:\n", train[target].value_counts(normalize=True, dropna=False))

X = train.drop(columns=[target])
y = train[target]
# Then choose a validation method BEFORE fitting imputers, encoders or scalers.
```

For regression, replace `value_counts()` with `y.describe()` and suitable distribution plots. For classification, check class frequency and the competition's metric.

## 11. Competition time-management checklist

- [ ] Identify target, prediction task, metric, output requirements.
- [ ] Check shapes, columns, data types, missingness.
- [ ] Inspect class/target distribution and obvious leakage risks.
- [ ] Decide the validation approach, including groups/time if relevant.
- [ ] Choose an initial subset of sensible features.
- [ ] Move to a **working baseline** quickly; investigate deeper only where needed.

**Next:** [Part 2 — Data Preprocessing](part2_preprocessing.md).
