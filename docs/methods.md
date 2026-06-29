# Methods Guide

All imputation methods share a uniform signature:

```python
imputed_df, validation_results = apply_<method>_imputation(
    df,                      # DataFrame with missing values
    columns_to_impute,       # list of column names to fill
    validation_df=None,      # optional: frame with held-out cells blanked
    validation_masks=None,   # optional: {col: boolean Series} of held-out cells
    original_values=None,    # optional: {col: Series} of true held-out values
)
```

If the optional validation arguments are supplied, `validation_results` is a `{column: {mae, rmse, accuracy, qwk, within1_accuracy, ...}}` dict scoring the imputation against the true held-out values. Otherwise it is `None`.

## Available methods

| Method | Function | How it works |
|---|---|---|
| **Mean** | `apply_mean_imputation` | Fills each column's missing values with the column mean. Fast baseline. |
| **Median** | `apply_median_imputation` | Fills with the column median. More robust to outliers than mean. |
| **KNN** | `apply_knn_imputation` | scikit-learn's `KNNImputer` — fills each missing value using the weighted average of its k nearest observed neighbors. |
| **MICE** | `apply_mice_imputation` | Multiple Imputation by Chained Equations via `miceforest` (random forests). Iteratively models each column conditional on all others. |
| **SoftImpute** | `apply_softimpute_imputation` | Iterative soft-thresholded SVD matrix completion (Mazumder, Hastie & Tibshirani 2010). Captures low-rank structure in the data. |

## Programmatic dispatch

Methods are registered in `missing_imputation.METHODS` for iteration and dispatch:

```python
import missing_imputation as mi

for name, fn in mi.METHODS.items():
    imputed, _ = fn(df, columns_to_impute)
    print(f"{name}: {imputed.isna().sum().sum()} remaining NaNs")
```

## Choosing a method

**Start with KNN** for most clinical PRO datasets — it balances accuracy and speed well for ordinal data with moderate missingness.

- **Mean/Median**: Use as baselines for comparison. Median is preferred when item distributions are skewed.
- **KNN**: Good default. Works well when patients with similar response patterns exist in the data. Adjusts `n_neighbors` automatically based on available data.
- **MICE**: Often the most accurate, but slower. Best when missingness has complex conditional structure. Uses random forests internally.
- **SoftImpute**: Good for data with underlying low-rank structure (e.g., items loading on a few latent factors). Scales well to wide matrices.

## Error handling

All methods raise `ValueError` if a target column is entirely NaN — there must be at least some observed values to impute from.

KNN and SoftImpute accept a `fallback` parameter:

```python
# Default: exceptions propagate
imputed, _ = mi.apply_knn_imputation(df, cols)

# Opt-in: fall back to mean imputation on failure
imputed, results = mi.apply_knn_imputation(df, cols, fallback="mean")
if results and results.get("_fallback"):
    print("Warning: KNN failed, used mean imputation instead")
```

## FACT-E column structure

The 44 FACT-E items are organised into subscales, defined in `missing_imputation.columns`:

| Subscale | Items | Count |
|---|---|---|
| PWB (Physical Well-Being) | gp1 — gp7 | 7 |
| SWB (Social/Family Well-Being) | gs1 — gs7 | 7 |
| EWB (Emotional Well-Being) | ge1 — ge6 | 6 |
| FWB (Functional Well-Being) | gf1 — gf7 | 7 |
| ECS (Esophageal Cancer Subscale) | a_hn1—a_hn5, a_hn7, a_hn10, a_e1—a_e7, a_c6, a_c2, a_act11 | 17 |

All items are scored on a 0–4 Likert scale (`LIKERT_MIN=0`, `LIKERT_MAX=4`).

```python
from missing_imputation.columns import FACTE_COLUMNS, SUBSCALES

print(f"Total items: {len(FACTE_COLUMNS)}")
for subscale, items in SUBSCALES.items():
    print(f"  {subscale}: {len(items)} items")
```
