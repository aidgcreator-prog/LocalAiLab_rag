---
name: cuml-machine-learning
description: Use for GPU-accelerated machine learning on tabular data using NVIDIA cuML with scikit-learn fallback. Triggers when tasks involve classification, regression, clustering, dimensionality reduction, or model training.
---

# cuML Machine Learning Skill

GPU-accelerated machine learning using NVIDIA RAPIDS cuML, with automatic scikit-learn fallback.

## When to Use This Skill

Use this skill when:
- Training classification models (predict categories, detect fraud)
- Training regression models (forecast values, predict prices)
- Clustering data (segment customers, group documents)
- Dimensionality reduction (visualize high-dimensional data)
- Any ML task on datasets where GPU acceleration helps

## Initialization (REQUIRED)

```python
import pandas as pd
import numpy as np

try:
    import cudf
    import cuml
    _test_data = cudf.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]})
    _km = cuml.cluster.KMeans(n_clusters=2, n_init=1, random_state=42)
    _km.fit(_test_data)
    assert len(_km.labels_) == 4
    GPU = True
except Exception as e:
    print(f"[GPU] cuml unavailable, falling back to scikit-learn: {e}")
    GPU = False

def read_csv(path):
    return cudf.read_csv(path) if GPU else pd.read_csv(path)

def to_pd(df):
    if not GPU:
        return df
    try:
        return df.to_pandas()
    except Exception as e:
        print(f"[GPU] .to_pandas() failed, using Arrow fallback: {e}")
        return df.to_arrow().to_pandas()
```

## Import Patterns

```python
if GPU:
    from cuml.cluster import KMeans, DBSCAN
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
    from cuml.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from cuml.decomposition import PCA
    from cuml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from cuml.model_selection import train_test_split
    from cuml.metrics import accuracy_score, r2_score, mean_squared_error
else:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
```

## Quick Reference

### Train/Test Split
```python
X = df[["feature1", "feature2", "feature3"]].astype("float32")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Classification
```python
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = float(accuracy_score(to_pd(y_test), to_pd(predictions)))
```

### Regression
```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = float(r2_score(to_pd(y_test), to_pd(predictions)))
```

### Clustering
```python
model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X)
labels = to_pd(model.labels_)
```

### Dimensionality Reduction
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype("float32"))
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled)
```

## Data Type Requirements

- cuML requires **float32 or float64** for features: `X.astype("float32")`
- Integer targets (classification labels) work directly
- Categorical columns must be encoded first (LabelEncoder)

## Output Guidelines

When reporting ML results:
- Include dataset shape (rows x features) and target distribution
- Show train/test split sizes
- Report key metrics (accuracy, R², MSE, etc.)
- List feature importances ranked by magnitude
- Note any data quality issues
