# missing_imputation

Imputation of missing values in **longitudinal clinical patient-reported outcome (PRO) data** — any questionnaire, any ordinal scale.

This package provides a uniform interface to impute missing values in longitudinal clinical datasets and to **benchmark** imputation methods against held-out values, so you can pick the method that best preserves your data. It works with any numeric or ordinal PRO instrument (FACT-E, EORTC QLQ, PROMIS, SF-36, etc.).

!!! info "v1.0 — Classical methods"
    The current release includes five classical imputation methods (mean, median, KNN, MICE, SoftImpute) and a full evaluation suite. Deep-learning and LLM-based methods are planned for v1.1.

## Key features

- **Five imputation methods** with a uniform API — swap methods with a single argument change
- **Works with any PRO instrument** — pass your own DataFrame and column list
- **Evaluation suite** — cross-validated benchmarking with ordinal-aware metrics (QWK, within-1-category accuracy)
- **Longitudinal evaluators** — trajectory fidelity, temporal smoothness, and missingness-pattern robustness
- **Synthetic data generator** — PHI-free demo data for testing (uses FACT-E structure as the included example)
- **CLI** — impute CSV files and generate demo data from the command line
- **No GPU required** — the core install has no deep-learning dependencies

## Quick example

```python
import pandas as pd
import missing_imputation as mi

df = pd.read_csv("my_pro_data.csv")
columns_to_impute = ["q1", "q2", "q3", "q4", "q5"]

# Impute with KNN
imputed, _ = mi.apply_knn_imputation(df, columns_to_impute)
```

The package also includes a synthetic FACT-E dataset for testing and demos:

```python
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)
imputed, _ = mi.apply_knn_imputation(df, FACTE_COLUMNS)
```

## Data safety

This repository contains **no patient data**. All examples and tests run on synthetic data, which reproduces only the *structure* of a PRO dataset (column layout, visit schedule, ordinal range), never any real values.
