"""Simple column-statistic imputation baselines (mean / median).

These are the lightweight, dependency-free baselines the research pipeline
compares against (the published comparison set is "mean, median, KNN, MICE,
MICE, deep autoencoders"). In the original notebooks mean imputation only
appeared as a fallback path; here it is exposed as a first-class method with
the same ``apply_*_imputation`` signature and validation reporting as the other
methods, so it can be benchmarked uniformly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ..metrics import calculate_classification_metrics, process_for_classification

__all__ = ["apply_mean_imputation", "apply_median_imputation"]

# Common shape of an ``apply_*_imputation`` return value.
ImputationResult = Tuple[pd.DataFrame, Optional[Dict[str, dict]]]


def _validate(imputed_df, columns_to_impute, validation_df, validation_masks,
              original_values, fill_stats):
    """Build the standard validation_results dict for a simple imputer.

    ``fill_stats`` maps column -> scalar fill value computed on the validation
    frame, so held-out cells are scored against the same statistic that would
    have been used to fill them.
    """
    validation_results = {}
    with tqdm(columns_to_impute, desc="Validating results") as pbar:
        for col in pbar:
            pbar.set_description(f"Validating {col}")
            mask = validation_masks[col] & validation_df[col].isna()

            if mask.sum() == 0:
                validation_results[col] = {'error': "No artificially missing values"}
                continue

            real_vals = original_values[col][mask]
            imputed_vals = pd.Series(fill_stats[col], index=real_vals.index)

            mae = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))

            real_vals_class = process_for_classification(real_vals)
            imputed_vals_class = process_for_classification(imputed_vals)
            classification_metrics = calculate_classification_metrics(real_vals_class, imputed_vals_class)

            validation_results[col] = {
                'mae': mae,
                'rmse': rmse,
                'accuracy': classification_metrics['accuracy'],
                'auc_multiclass': classification_metrics['auc_multiclass'],
                'avg_sensitivity': classification_metrics['avg_sensitivity'],
                'avg_specificity': classification_metrics['avg_specificity'],
                'avg_ppv': classification_metrics['avg_ppv'],
                'avg_npv': classification_metrics['avg_npv'],
                'precision_macro': classification_metrics['precision_macro'],
                'recall_macro': classification_metrics['recall_macro'],
                'real_distribution': real_vals.describe(),
                'imputed_distribution': pd.Series(imputed_vals).describe(),
            }
    return validation_results


def _apply_simple(df, columns_to_impute, statistic, validation_df,
                  validation_masks, original_values):
    """Shared implementation for mean / median imputation.

    ``statistic`` is either 'mean' or 'median'.
    """
    imputed_df = df.copy()
    for col in columns_to_impute:
        if col in imputed_df.columns:
            imputed_df[col] = pd.to_numeric(imputed_df[col], errors='coerce')
            fill = getattr(imputed_df[col], statistic)()
            imputed_df[col] = imputed_df[col].fillna(fill)

    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        fill_stats = {}
        for col in columns_to_impute:
            col_vals = pd.to_numeric(validation_df[col], errors='coerce')
            fill_stats[col] = getattr(col_vals, statistic)()
        validation_results = _validate(
            imputed_df, columns_to_impute, validation_df,
            validation_masks, original_values, fill_stats,
        )

    return imputed_df, validation_results


def apply_mean_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    validation_df: Optional[pd.DataFrame] = None,
    validation_masks: Optional[Dict[str, pd.Series]] = None,
    original_values: Optional[Dict[str, pd.Series]] = None,
) -> ImputationResult:
    """Impute missing values with each column's mean. See module docstring."""
    return _apply_simple(df, columns_to_impute, 'mean',
                         validation_df, validation_masks, original_values)


def apply_median_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    validation_df: Optional[pd.DataFrame] = None,
    validation_masks: Optional[Dict[str, pd.Series]] = None,
    original_values: Optional[Dict[str, pd.Series]] = None,
) -> ImputationResult:
    """Impute missing values with each column's median. See module docstring."""
    return _apply_simple(df, columns_to_impute, 'median',
                         validation_df, validation_masks, original_values)
