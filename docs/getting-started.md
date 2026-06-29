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
import pandas as pd
import missing_imputation as mi

df = pd.read_csv("my_pro_data.csv")
columns_to_impute = ["q1", "q2", "q3", "q4", "q5"]

# Impute missing values with KNN
imputed, _ = mi.apply_knn_imputation(df, columns_to_impute)

# Verify no missing values remain
assert imputed[columns_to_impute].isna().sum().sum() == 0
```

Your DataFrame needs at minimum:

- An **id** column identifying each patient
- A **visit/event** column (e.g. `redcap_event_name`) for longitudinal structure
- The **item columns** you want to impute (numeric or ordinal values)

You can impute any list of numeric column names — the package is not tied to any specific PRO instrument.

## Using the included demo data

The package ships with a synthetic FACT-E (Functional Assessment of Cancer Therapy — Esophageal) dataset generator for testing and examples:

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)
imputed, _ = mi.apply_knn_imputation(df, FACTE_COLUMNS)
```

This generates PHI-free data with the same structure as a real PRO dataset (patient IDs, visit events, 44 ordinal 0–4 items).

## Benchmarking methods

To compare methods on your data, use the evaluation suite:

```python
import missing_imputation as mi

# Cross-validated benchmark across all classical methods
results = mi.evaluate_with_sparse_validation(df, columns_to_impute, n_folds=5)

for method, r in results.items():
    print(f"{method:10s}  MAE={r['avg_mae']:.3f}  QWK={r['avg_qwk']:.3f}  "
          f"within1={r['avg_within1_accuracy']:.3f}")
```

See the [Evaluation & Benchmarking](evaluation.md) guide for the full evaluator suite.

## From the command line

```bash
# Generate synthetic demo data
missing-impute demo -o demo.csv --patients 60 --missing 0.2

# Impute a file with KNN (specify your columns)
missing-impute impute data.csv -o filled.csv --method knn --columns q1,q2,q3

# List available methods
missing-impute methods
```

See the [CLI Reference](cli.md) for all options.
