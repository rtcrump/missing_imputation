# missing_imputation

Imputation of missing values in **longitudinal clinical patient-reported
outcome (PRO) data** — any questionnaire, any ordinal scale.

This package provides a uniform interface to impute missing values in
longitudinal clinical datasets and to **benchmark** imputation methods against
held-out values, so you can pick the method that best preserves your data.
It works with any numeric or ordinal PRO instrument (FACT-E, EORTC QLQ,
PROMIS, SF-36, etc.).

> **Status:** v1.0 — Classical imputation methods (mean, median, KNN, MICE,
> SoftImpute) and a full evaluation suite are packaged, tested, and ready to
> use. Deep-learning and LLM-based methods are planned for v1.1 — see
> [Roadmap](#roadmap).

**[Documentation](https://rtcrump.github.io/missing_imputation/)** — full
guides on methods, evaluation, CLI, and auto-generated API reference.

## What it does

Clinical PRO instruments ask patients to answer ordinal questions at each
study visit. In practice many answers are missing. This package provides:

- **Five imputation methods** with a uniform API — swap methods with a single
  argument change
- **Evaluation suite** — cross-validated benchmarking with ordinal-aware
  metrics (QWK, within-1-category accuracy)
- **Longitudinal evaluators** — trajectory fidelity, temporal smoothness,
  and missingness-pattern robustness
- **Synthetic data generator** — PHI-free demo data for testing
- **CLI** — impute CSV files from the command line

## Installation

```bash
pip install -e .            # core: classical methods + synthetic data + CLI
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[plot]"    # + matplotlib / seaborn (for plotting helpers)
```

Requires Python ≥ 3.9. The core install has **no GPU or deep-learning
dependencies**.

## Quick start

### With your own data

```python
import pandas as pd
import missing_imputation as mi

df = pd.read_csv("my_pro_data.csv")
columns_to_impute = ["q1", "q2", "q3", "q4", "q5"]

# Impute with KNN
imputed, _ = mi.apply_knn_imputation(df, columns_to_impute)
```

### With the included demo data

The package ships with a synthetic FACT-E dataset generator for testing:

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)
imputed, _ = mi.apply_knn_imputation(df, FACTE_COLUMNS)

assert imputed[FACTE_COLUMNS].isna().sum().sum() == 0
```

Every method shares the same signature and return shape:

```python
imputed_df, validation_results = apply_<method>_imputation(
    df,                      # DataFrame with missing values
    columns_to_impute,       # list of column names to fill
    validation_df=None,      # optional: frame with held-out cells blanked
    validation_masks=None,   # optional: {col: boolean Series} of held-out cells
    original_values=None,    # optional: {col: Series} of true held-out values
)
```

If the optional validation arguments are supplied, `validation_results` is a
`{column: {mae, rmse, accuracy, qwk, within1_accuracy, ...}}` dict scoring the
imputation against the true held-out values. Otherwise it is `None`.

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end
benchmark across all classical methods.

## Available methods

| Name (`mi.METHODS`) | Function | Notes |
|---|---|---|
| `mean` | `apply_mean_imputation` | Column-mean baseline |
| `median` | `apply_median_imputation` | Column-median baseline |
| `knn` | `apply_knn_imputation` | scikit-learn `KNNImputer` |
| `mice` | `apply_mice_imputation` | MICE via `miceforest` (random forests) |
| `softimpute` | `apply_softimpute_imputation` | Soft-thresholded SVD matrix completion |

Methods are also registered in `missing_imputation.METHODS` for programmatic
dispatch:

```python
for name, fn in mi.METHODS.items():
    imputed, _ = fn(df, columns_to_impute)
```

## Benchmarking and evaluation

The `missing_imputation.evaluation` module compares methods on your own data.
Every evaluator defaults to scoring the classical `METHODS` registry, but
accepts any `{name: apply_*_imputation}` mapping via `methods=`.

```python
import missing_imputation as mi

# Use your own DataFrame and column list
results = mi.evaluate_with_sparse_validation(df, columns_to_impute, n_folds=5)
for method, r in results.items():
    print(f"{method:10s} MAE={r['avg_mae']:.3f}  QWK={r['avg_qwk']:.3f}  "
          f"within1={r['avg_within1_accuracy']:.3f}")
```

| Evaluator | What it measures |
|---|---|
| `evaluate_with_sparse_validation` | Fold-wise masking cross-validation: MAE/RMSE plus the full classification suite (accuracy, AUC, **QWK**, **within-1-category accuracy**, sensitivity/specificity/PPV/NPV). |
| `evaluate_trajectory_fidelity` | Masks each patient's middle visit; scores how well the imputed values match the true per-patient trajectory (MAE + correlation). |
| `evaluate_temporal_smoothness` | Penalises unrealistic visit-to-visit jumps in already-imputed frames. Pair with `run_all_methods`. |
| `evaluate_missing_pattern_robustness` | Compares accuracy on monotone (permanent dropout) vs intermittent (sporadic) missingness. |

The scoring metrics are ordinal-aware: **quadratic weighted kappa (QWK)** rewards
near-misses on ordinal scales, and **within-1-category accuracy** reports the
fraction of imputations off by at most one category. The ordinal range defaults
to 0–4 but is configurable via `process_for_classification(values, min_val, max_val)`.

See [`examples/benchmark.py`](examples/benchmark.py) for a full run across all
four evaluators.

## Command line

```bash
missing-impute demo -o demo.csv --patients 60 --missing 0.2   # synthetic data
missing-impute impute data.csv -o filled.csv --method knn --columns q1,q2,q3
missing-impute methods                                         # list methods
```

Use `--columns` to specify which columns to impute. If omitted, the CLI
looks for FACT-E item columns in the file (the included demo instrument).

## Data safety

This repository contains **no patient data**. All examples and tests run on
synthetic data generated by `make_synthetic_facte`, which reproduces only the
*structure* of a PRO dataset (column layout, visit schedule, ordinal range),
never any real values.

## Roadmap

### v1.0 (current)

- [x] Package skeleton, `pip install -e .`, CLI
- [x] Classical methods extracted, tested, benchmarkable on synthetic data
- [x] Synthetic demo data generator
- [x] Evaluation suite (sparse-validation, trajectory fidelity,
      temporal smoothness, missing-pattern robustness) + QWK / within-1 metrics
- [x] Type hints on the public imputation API
- [x] CI (GitHub Actions): lint + test on Python 3.9 / 3.11 / 3.12
- [x] Documentation site (MkDocs-Material + mkdocstrings, deployed to GitHub Pages)

### v1.1 (planned)

- [ ] Deep-learning methods (VAE, deep autoencoder, BayesianPCA, LSTM)
- [ ] LLM-based methods (Gemma + LoRA, TimesFM)
- [ ] Plotting helpers for evaluation results

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
