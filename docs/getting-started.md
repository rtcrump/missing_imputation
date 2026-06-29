# Getting Started

## Installation

```bash
pip install -e .            # core: classical methods + synthetic data + CLI
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[plot]"    # + matplotlib / seaborn (for plotting helpers)
```

Requires Python 3.9 or later. The core install has **no GPU or deep-learning dependencies**.

## Your first imputation

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

# 1. Generate synthetic data (or load your own DataFrame)
df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)

# 2. Impute missing FACT-E items with KNN
imputed, _ = mi.apply_knn_imputation(df, FACTE_COLUMNS)

# 3. Verify no missing values remain
assert imputed[FACTE_COLUMNS].isna().sum().sum() == 0
```

## Using your own data

Your DataFrame needs at minimum:

- An **id** column identifying each patient
- A **visit/event** column (e.g. `redcap_event_name`) for longitudinal structure
- The **item columns** you want to impute (numeric, ordinal values)

```python
import pandas as pd
import missing_imputation as mi

df = pd.read_csv("my_data.csv")
columns_to_impute = ["item1", "item2", "item3"]

imputed, _ = mi.apply_knn_imputation(df, columns_to_impute)
```

You are not limited to FACT-E columns — any list of numeric column names works.

## Benchmarking methods

To compare methods on your data, hold out some observed values and score reconstruction:

```python
import numpy as np
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=120, n_visits=6, missing_rate=0.25)

# Cross-validated benchmark across all classical methods
results = mi.evaluate_with_sparse_validation(df, FACTE_COLUMNS, n_folds=5)

for method, r in results.items():
    print(f"{method:10s}  MAE={r['avg_mae']:.3f}  QWK={r['avg_qwk']:.3f}  "
          f"within1={r['avg_within1_accuracy']:.3f}")
```

See the [Evaluation & Benchmarking](evaluation.md) guide for the full evaluator suite.

## From the command line

```bash
# Generate synthetic demo data
missing-impute demo -o demo.csv --patients 60 --missing 0.2

# Impute a file with KNN
missing-impute impute demo.csv -o filled.csv --method knn

# List available methods
missing-impute methods
```

See the [CLI Reference](cli.md) for all options.
