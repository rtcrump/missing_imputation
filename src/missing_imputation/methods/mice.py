"""MICE (Multiple Imputation by Chained Equations) via miceforest.

Extracted from the research pipeline. The public entry point is
``apply_mice_imputation``; the imputation logic is preserved exactly so results
match the published experiments.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ..metrics import calculate_classification_metrics, process_for_classification
from .simple import ImputationResult

__all__ = ["apply_mice_imputation"]


def apply_mice_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    validation_df: Optional[pd.DataFrame] = None,
    validation_masks: Optional[Dict[str, pd.Series]] = None,
    original_values: Optional[Dict[str, pd.Series]] = None,
) -> ImputationResult:
    """
    Apply MICE imputation using miceforest package

    Parameters:
    -----------
    df : pandas.DataFrame
        Data with missing values
    columns_to_impute : list
        List of column names to impute
    validation_df : pandas.DataFrame, optional
        Validation dataset with artificially missing values
    validation_masks : dict, optional
        Dictionary of masks for validation data
    original_values : dict, optional
        Dictionary of original values for validation

    Returns:
    --------
    imputed_df : pandas.DataFrame
        Data with imputed values
    validation_results : dict, optional
        Validation results if validation data provided
    """

    for col in columns_to_impute:
        if col in df.columns and df[col].isna().all():
            raise ValueError(
                f"Column '{col}' is entirely NaN — cannot impute from no observations"
            )

    # Set threads for LightGBM
    os.environ['OMP_NUM_THREADS'] = '10'

    # Drop non-numeric columns that miceforest cannot handle
    df_mice = df.select_dtypes(exclude=['object', 'datetime64[ns]', 'datetime64']).copy()

    # miceforest renamed `datasets` -> `num_datasets` in 6.x; the original
    # research code targeted 5.x. We target the 6.x API (required for numpy 2.x
    # compatibility) while preserving the same configuration: a single dataset,
    # 5 MICE iterations, and LightGBM with 80 boosting rounds / max_depth 10.
    # Running 5 single-iteration steps is equivalent to ``iterations=5`` because
    # the chained equations are applied sequentially.
    kernel = mf.ImputationKernel(
        df_mice,
        num_datasets=1,
        variable_schema={
            col: [c for c in df_mice.columns if c != col] for col in columns_to_impute
        },
        random_state=42
    )

    # Run imputation
    for _ in tqdm(range(5), desc="MICE Imputation"):
        kernel.mice(
            iterations=1,
            verbose=False,
            num_boost_round=80,
            max_depth=10,
            num_threads=10
        )

    # Get imputed data
    imputed_mice = kernel.complete_data(0)
    # Put results back into a copy of the original df (which still has redcap_event_name etc.)
    imputed_df = df.copy()
    for col in columns_to_impute:
        if col in imputed_mice.columns:
            imputed_df[col] = imputed_mice[col].values

    # Validate if validation data provided
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        validation_results = {}

        # Fit a separate MICE kernel on the validation frame so we score
        # values imputed from the held-out data, not from the training frame.
        val_mice = validation_df.select_dtypes(exclude=['object', 'datetime64[ns]', 'datetime64']).copy()
        val_kernel = mf.ImputationKernel(
            val_mice,
            num_datasets=1,
            variable_schema={
                col: [c for c in val_mice.columns if c != col] for col in columns_to_impute
            },
            random_state=42,
        )
        for _ in range(5):
            val_kernel.mice(iterations=1, verbose=False, num_boost_round=80,
                            max_depth=10, num_threads=10)
        val_imputed_mice = val_kernel.complete_data(0)

        for col in columns_to_impute:
            mask = validation_masks[col] & validation_df[col].isna()

            if mask.sum() == 0:
                validation_results[col] = {
                    'error': "No artificially missing values"
                }
                continue

            real_vals = original_values[col][mask]
            imputed_vals = val_imputed_mice[col][mask]

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
                'qwk': classification_metrics['qwk'],
                'within1_accuracy': classification_metrics['within1_accuracy'],
                'avg_sensitivity': classification_metrics['avg_sensitivity'],
                'avg_specificity': classification_metrics['avg_specificity'],
                'avg_ppv': classification_metrics['avg_ppv'],
                'avg_npv': classification_metrics['avg_npv'],
                'precision_macro': classification_metrics['precision_macro'],
                'recall_macro': classification_metrics['recall_macro'],
                'real_distribution': real_vals.describe(),
                'imputed_distribution': imputed_vals.describe()
            }

    return imputed_df, validation_results
