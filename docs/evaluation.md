# Evaluation & Benchmarking

The `missing_imputation.evaluation` module provides four evaluators for comparing imputation methods. Each defaults to scoring the classical `METHODS` registry but accepts any `{name: apply_*_imputation}` mapping via `methods=`.

## Evaluators at a glance

| Evaluator | What it measures | When to use |
|---|---|---|
| `evaluate_with_sparse_validation` | Cross-validated reconstruction accuracy | Primary benchmark — "how accurately does each method recover held-out values?" |
| `evaluate_trajectory_fidelity` | Per-patient trajectory preservation | Longitudinal data — "does the imputed trajectory match the real one?" |
| `evaluate_temporal_smoothness` | Visit-to-visit jump magnitude | Longitudinal data — "are imputed trajectories realistic or jumpy?" |
| `evaluate_missing_pattern_robustness` | Accuracy by missingness pattern | Longitudinal data — "does accuracy differ for dropout vs. sporadic missingness?" |

## Sparse cross-validation

The primary benchmark. Repeatedly masks observed cells, imputes, and scores recovery.

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=120, n_visits=6, missing_rate=0.25)

results = mi.evaluate_with_sparse_validation(df, FACTE_COLUMNS, n_folds=5)

for method, r in sorted(results.items(), key=lambda kv: kv[1]["avg_mae"]):
    print(f"{method:12s} MAE={r['avg_mae']:.3f}  QWK={r['avg_qwk']:.3f}  "
          f"within1={r['avg_within1_accuracy']:.3f}")
```

**Returned metrics** (each with `avg_` and `std_` variants):

- `mae`, `rmse` — continuous error on the raw imputed values (no rounding)
- `accuracy` — exact match after rounding to 0–4
- `qwk` — quadratic weighted kappa (ordinal agreement, rewards near-misses)
- `within1_accuracy` — fraction of imputations off by at most one Likert category
- `auc_multiclass` — macro-averaged ROC-AUC
- `avg_sensitivity`, `avg_specificity`, `avg_ppv`, `avg_npv` — per-class clinical metrics
- `precision_macro`, `recall_macro`
- `time` — per-fold execution time

## Trajectory fidelity

Masks each patient's middle visit and measures how well the imputed values match the true per-patient trajectory.

```python
traj = mi.evaluate_trajectory_fidelity(df, FACTE_COLUMNS)

for method, r in sorted(traj.items(), key=lambda kv: kv[1]["trajectory_mae"]):
    print(f"{method:12s} trajectory MAE={r['trajectory_mae']:.3f}  "
          f"corr={r['trajectory_correlation']:.3f}")
```

**Returned metrics:** `trajectory_mae`, `trajectory_mae_std`, `trajectory_correlation`, `n_patients`.

## Temporal smoothness

Penalises unrealistic visit-to-visit jumps in already-imputed frames. Lower is smoother.

```python
# First, produce imputed frames for each method
imputed_dfs = mi.run_all_methods(df, FACTE_COLUMNS)

# Then score smoothness
smooth = mi.evaluate_temporal_smoothness(df, imputed_dfs, FACTE_COLUMNS)

for method, r in sorted(smooth.items(), key=lambda kv: kv[1]["mean_smoothness"]):
    print(f"{method:12s} smoothness={r['mean_smoothness']:.3f}")
```

**Returned metrics:** `mean_smoothness`, `std_smoothness`, `n_patients`.

## Missing-pattern robustness

Compares accuracy on **monotone** (contiguous visits then permanent dropout) vs **intermittent** (sporadic, gaps between visits) missingness patterns.

```python
robust = mi.evaluate_missing_pattern_robustness(df, FACTE_COLUMNS, n_folds=5)

for pattern, group in robust.items():
    if not group:
        continue
    ranked = ", ".join(f"{m}={v['mae']:.3f}" for m, v in sorted(group.items()))
    print(f"  {pattern}: {ranked}")
```

**Returned structure:** `{'monotone': {method: {'mae', 'rmse'}}, 'intermittent': {...}}`.

## Scoring metrics

The metrics are designed for **ordinal clinical data** (0–4 Likert scale):

- **Quadratic weighted kappa (QWK)** rewards near-misses on the ordinal scale. A prediction of 2 when the truth is 3 is penalised less than a prediction of 0.
- **Within-1-category accuracy** reports the fraction of imputations that are off by at most one category — clinically, a "close enough" threshold.

Both are computed by `missing_imputation.metrics.calculate_classification_metrics` after rounding imputed values to the nearest integer and clipping to [0, 4].

## Using custom methods

Pass any `{name: callable}` mapping to benchmark your own imputation functions alongside the built-in ones:

```python
def my_custom_imputer(df, columns, **kwargs):
    imputed = df.copy()
    for col in columns:
        imputed[col] = imputed[col].fillna(0)
    return imputed, None

custom_methods = {**mi.METHODS, "zero_fill": my_custom_imputer}

results = mi.evaluate_with_sparse_validation(
    df, FACTE_COLUMNS, methods=custom_methods
)
```
