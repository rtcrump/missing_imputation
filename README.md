# missing_imputation

Imputation of missing values in **longitudinal clinical patient-reported
outcome (PRO) data**, built around the FACT-E (Functional Assessment of Cancer
Therapy — Esophageal) questionnaire.

This package began as a research codebase comparing LLM-based imputation against
traditional statistical and deep-learning methods for clinical questionnaire
data collected at multiple timepoints. It is being packaged into an installable,
documented open-source tool for clinical researchers who need to handle missing
data in longitudinal PRO datasets.

> **Status:** Alpha. The classical imputation methods (mean, median, KNN, MICE,
> SoftImpute) are packaged, tested, and usable today. The deep-learning
> (VAE, deep autoencoder, BayesianPCA, LSTM) and LLM-based methods
> (Gemma + LoRA, TimesFM) currently live in the research notebooks under
> [`notebook/`](notebook/) and are being extracted into the package — see
> [Roadmap](#roadmap).

## What it does

Clinical PRO instruments like FACT-E ask patients to answer dozens of ordinal
(0–4 Likert) questions at each study visit. In practice many answers are
missing. This package provides a uniform interface to impute those missing
values and to **benchmark** imputation methods against held-out values, so you
can pick the method that best preserves your data.

The 44 FACT-E items are organised into subscales (Physical, Social/Family,
Emotional, Functional well-being, and an Esophageal Cancer Subscale); see
[`missing_imputation/columns.py`](src/missing_imputation/columns.py).

## Installation

```bash
pip install -e .            # core: classical methods + synthetic data + CLI
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[deep]"    # + torch / tensorflow (for deep-learning methods)
pip install -e ".[llm]"     # + transformers / peft (for LLM methods)
pip install -e ".[plot]"    # + matplotlib / seaborn (for plotting helpers)
```

Requires Python ≥ 3.9. The core install has **no GPU or deep-learning
dependencies**.

## Quick start

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

# Generate PHI-free synthetic FACT-E data (or load your own DataFrame).
df = mi.make_synthetic_facte(n_patients=80, n_visits=5, missing_rate=0.2)

# Impute the FACT-E item columns with KNN.
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
`{column: {mae, rmse, accuracy, ...}}` dict scoring the imputation against the
true held-out values. Otherwise it is `None`.

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end
benchmark across all classical methods.

## Available methods

| Name (`mi.METHODS`) | Function | Dependencies | Notes |
|---|---|---|---|
| `mean` | `apply_mean_imputation` | core | Column-mean baseline |
| `median` | `apply_median_imputation` | core | Column-median baseline |
| `knn` | `apply_knn_imputation` | core | scikit-learn `KNNImputer` |
| `mice` | `apply_mice_imputation` | core | MICE via `miceforest` (random forests) |
| `softimpute` | `apply_softimpute_imputation` | core | Soft-thresholded SVD matrix completion |

Methods are also registered in `missing_imputation.METHODS` for programmatic
dispatch:

```python
for name, fn in mi.METHODS.items():
    imputed, _ = fn(df, FACTE_COLUMNS)
```

## Benchmarking and evaluation

The `missing_imputation.evaluation` module compares methods on your own data.
Every evaluator defaults to scoring the classical `METHODS` registry, but
accepts any `{name: apply_*_imputation}` mapping via `methods=`.

```python
import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

df = mi.make_synthetic_facte(n_patients=120, n_visits=6, missing_rate=0.25)

# Cross-validated benchmark: repeatedly mask observed cells, impute, and score.
results = mi.evaluate_with_sparse_validation(df, FACTE_COLUMNS, n_folds=5)
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
near-misses on the 0–4 Likert scale, and **within-1-category accuracy** reports
the fraction of imputations off by at most one category. Both are returned by
`missing_imputation.metrics.calculate_classification_metrics`.

See [`examples/benchmark.py`](examples/benchmark.py) for a full run across all
four evaluators.

## Command line

```bash
missing-impute demo -o demo.csv --patients 60 --missing 0.2   # synthetic data
missing-impute impute demo.csv -o filled.csv --method knn      # impute a file
missing-impute methods                                         # list methods
```

## Data safety

This repository contains **no patient data**. The real clinical data used in
the study is PHI and is excluded via [`.gitignore`](.gitignore); all examples
and tests run on synthetic data generated by `make_synthetic_facte`, which
reproduces only the *structure* of the data (column names, visit schedule,
ordinal range), never any real values.

## Roadmap

- [x] Package skeleton, `pip install -e .`, CLI
- [x] Classical methods extracted, tested, benchmarkable on synthetic data
- [x] Synthetic FACT-E data generator
- [x] Evaluation suite extracted (sparse-validation, trajectory fidelity,
      temporal smoothness, missing-pattern robustness) + QWK / within-1 metrics
- [x] Type hints on the public imputation API
- [ ] Extract deep-learning methods (VAE, deep autoencoder, BayesianPCA, LSTM)
- [ ] Extract longitudinal MICE and LLM methods (Gemma + LoRA, TimesFM)
- [ ] Plotting helpers for the evaluation results
- [ ] API reference docs site

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
