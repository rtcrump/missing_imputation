"""MICE (Multiple Imputation by Chained Equations) via miceforest.

Extracted from the research pipeline. The public entry point is
``apply_mice_imputation``; the imputation logic is preserved exactly so results
match the published experiments.
"""

import os

import miceforest as mf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ..metrics import calculate_classification_metrics, process_for_classification

__all__ = ["apply_mice_imputation"]


def apply_mice_imputation(df, columns_to_impute, validation_df=None, validation_masks=None, original_values=None):
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

        # Compare imputed values to real values
        for col in columns_to_impute:
            # Get indices where values were artificially set to NaN
            mask = validation_masks[col] & validation_df[col].isna()

            if mask.sum() == 0:
                validation_results[col] = {
                    'error': "No artificially missing values"
                }
                continue

            real_vals = original_values[col][mask]
            imputed_vals = imputed_df[col][mask]

            # Calculate continuous metrics (MAE and RMSE) - NO ROUNDING
            mae = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))

            # Calculate classification metrics - WITH ROUNDING
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
                'imputed_distribution': imputed_vals.describe()
            }

    return imputed_df, validation_results
