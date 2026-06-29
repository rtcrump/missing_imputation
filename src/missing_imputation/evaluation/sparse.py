"""Sparse cross-validation benchmarking harness.

``evaluate_with_sparse_validation`` is the core entry point for comparing
imputation methods on a dataset that is itself sparse (the FACT-E data has many
missing cells). It repeatedly masks a fold of each column's *observed* values,
imputes, and scores the recovered values against the held-out truth — yielding
MAE/RMSE plus the full classification metric suite (accuracy, AUC, QWK,
within-1-category accuracy, sensitivity/specificity, ...).

The fold-construction and scoring logic is preserved verbatim from the research
pipeline (``notebook/Timeseries/imputation21``). The only packaging change is
that the set of methods to benchmark is now a parameter (defaulting to the
library's classical :data:`missing_imputation.METHODS` registry) rather than a
hard-coded list of methods that live in the research notebooks.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..metrics import calculate_classification_metrics, process_for_classification

__all__ = ["evaluate_with_sparse_validation"]

# Metrics aggregated (mean + std) across folds/columns at the end of a run.
_METRIC_NAMES = [
    "mae", "rmse", "accuracy", "auc_multiclass", "qwk", "within1_accuracy",
    "avg_sensitivity", "avg_specificity", "avg_ppv", "avg_npv",
    "precision_macro", "recall_macro", "time",
]


def _resolve_methods(
    methods: Optional[Dict[str, Callable]],
) -> Dict[str, Callable]:
    """Return a name->callable mapping, defaulting to the package registry."""
    if methods is None:
        from .. import METHODS  # lazy import to avoid a circular import

        return dict(METHODS)
    return dict(methods)


def evaluate_with_sparse_validation(
    df,
    columns_to_impute: List[str],
    methods: Optional[Dict[str, Callable]] = None,
    n_folds: int = 5,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Benchmark imputation methods via fold-wise masking of observed values.

    For each fold, a disjoint subset of every eligible column's observed values
    is masked simultaneously, each method imputes the whole frame once, and the
    recovered values are scored against the truth. Results are accumulated across
    folds and columns, then summarised as ``avg_<metric>`` / ``std_<metric>``.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format data with missing values. Must contain ``columns_to_impute``.
    columns_to_impute : list of str
        Columns to evaluate (and impute).
    methods : dict, optional
        Mapping ``{display_name: apply_*_imputation}``. Defaults to the classical
        :data:`missing_imputation.METHODS` registry. Each callable must accept
        ``(df, columns_to_impute)`` and return ``(imputed_df, validation_results)``.
    n_folds : int
        Target number of folds. Columns with too few observed values use fewer.
    verbose : bool
        If True, print per-fold progress (matches the original research output).

    Returns
    -------
    dict
        ``{method_name: {<metric>: [per-fold values], 'avg_<metric>': float,
        'std_<metric>': float, ...}}``.
    """
    methods = _resolve_methods(methods)

    results = {
        method: {name: [] for name in _METRIC_NAMES} for method in methods
    }

    # Filter valid columns
    valid_columns = [
        col for col in columns_to_impute
        if col in df.columns and not col.startswith("qol_date_")
    ]
    if verbose and len(valid_columns) != len(columns_to_impute):
        excluded_cols = set(columns_to_impute) - set(valid_columns)
        print(f"Warning: Excluding {len(excluded_cols)} columns: {sorted(excluded_cols)}")

    # Pre-compute per-column observed indices once
    col_observed_indices = {}
    for col in valid_columns:
        observed = df.index[df[col].notna()].tolist()
        if len(observed) >= max(5, n_folds * 2):
            col_observed_indices[col] = observed
        elif verbose:
            print(f"Skipping {col}: too few observed values ({len(observed)})")

    eligible_columns = list(col_observed_indices.keys())
    if verbose:
        print(f"\nEligible columns for validation: {len(eligible_columns)}/{len(valid_columns)}")

    # Pre-compute fold test indices per column once
    col_fold_indices = {}
    for col in eligible_columns:
        observed = col_observed_indices[col]
        n_observed = len(observed)
        fold_size = max(5, n_observed // n_folds)
        np.random.seed(42)
        shuffled = np.random.permutation(observed)
        n_actual_folds = min(n_folds, n_observed // fold_size)
        folds = []
        for fold in range(n_actual_folds):
            start = fold * fold_size
            end = min(start + fold_size, n_observed)
            folds.append(shuffled[start:end])
        col_fold_indices[col] = folds

    max_folds = max((len(f) for f in col_fold_indices.values()), default=0)

    for fold in range(max_folds):
        if verbose:
            print(f"\n{'='*80}\nFOLD {fold + 1}/{max_folds}\n{'='*80}")

        # --- Mask ALL columns at once for this fold ------------------------
        df_fold = df.copy()
        fold_originals = {}  # {col: {'test_indices', 'original_values'}}

        for col in eligible_columns:
            folds = col_fold_indices[col]
            if fold >= len(folds):
                continue
            test_indices = folds[fold]
            fold_originals[col] = {
                "test_indices": test_indices,
                "original_values": df_fold.loc[test_indices, col].copy(),
            }
            df_fold.loc[test_indices, col] = np.nan

        participating_cols = list(fold_originals.keys())
        if verbose:
            print(f"Fold {fold+1}: masking {len(participating_cols)} columns simultaneously")

        # --- Run each method ONCE on the multi-column masked dataframe -----
        for method_name, method_func in methods.items():
            if verbose:
                print(f"\nRunning {method_name}...")
            try:
                method_df = df_fold.copy()
                if "qol_date" in method_df.columns:
                    method_df = method_df.drop(columns=["qol_date"])

                start_time = time.time()
                imputed_df, _ = method_func(method_df, valid_columns)
                execution_time = time.time() - start_time

                # --- Evaluate all columns from this single imputed result --
                for col in participating_cols:
                    if col not in imputed_df.columns:
                        continue

                    test_indices = fold_originals[col]["test_indices"]
                    original_values = fold_originals[col]["original_values"]
                    imputed_values = imputed_df.loc[test_indices, col]

                    still_missing = imputed_values.isna().sum()
                    if still_missing > 0:
                        valid_idx = [i for i in test_indices
                                     if not _is_nan(imputed_df.loc[i, col])]
                        if not valid_idx:
                            continue
                        original_values_filtered = df.loc[valid_idx, col]
                        imputed_values_filtered = imputed_df.loc[valid_idx, col]
                    else:
                        original_values_filtered = original_values
                        imputed_values_filtered = imputed_values

                    # Continuous metrics (no rounding)
                    mae = mean_absolute_error(original_values_filtered, imputed_values_filtered)
                    rmse = np.sqrt(mean_squared_error(original_values_filtered, imputed_values_filtered))

                    # Classification metrics (with rounding to the 0-4 scale)
                    real_cls = process_for_classification(original_values_filtered)
                    imputed_cls = process_for_classification(imputed_values_filtered)
                    cls = calculate_classification_metrics(real_cls, imputed_cls)

                    results[method_name]["mae"].append(mae)
                    results[method_name]["rmse"].append(rmse)
                    results[method_name]["accuracy"].append(cls["accuracy"])
                    results[method_name]["auc_multiclass"].append(
                        cls["auc_multiclass"] if not np.isnan(cls["auc_multiclass"]) else 0)
                    results[method_name]["qwk"].append(cls["qwk"])
                    results[method_name]["within1_accuracy"].append(cls["within1_accuracy"])
                    results[method_name]["avg_sensitivity"].append(cls["avg_sensitivity"])
                    results[method_name]["avg_specificity"].append(cls["avg_specificity"])
                    results[method_name]["avg_ppv"].append(cls["avg_ppv"])
                    results[method_name]["avg_npv"].append(cls["avg_npv"])
                    results[method_name]["precision_macro"].append(cls["precision_macro"])
                    results[method_name]["recall_macro"].append(cls["recall_macro"])

                results[method_name]["time"].append(execution_time)

                if verbose and results[method_name]["mae"]:
                    recent = results[method_name]["mae"][-len(participating_cols):]
                    print(f"  {method_name} fold {fold+1} avg MAE across "
                          f"{len(participating_cols)} cols: {np.mean(recent):.4f} "
                          f"| Time: {execution_time:.2f}s")

            except Exception as e:  # noqa: BLE001 (mirror research robustness)
                if verbose:
                    print(f"  Error with {method_name}: {e}")

    # Aggregate results
    for method in methods:
        if results[method]["mae"]:
            for metric in _METRIC_NAMES:
                results[method][f"avg_{metric}"] = np.mean(results[method][metric])
                results[method][f"std_{metric}"] = np.std(results[method][metric])
        else:
            if verbose:
                print(f"No valid results for {method}")
            for metric in _METRIC_NAMES:
                results[method][f"avg_{metric}"] = np.nan
                results[method][f"std_{metric}"] = np.nan

    return results


def _is_nan(value) -> bool:
    """True if ``value`` is a float NaN (scalar-safe)."""
    try:
        return bool(value != value)
    except Exception:  # noqa: BLE001
        return False
