"""K-Nearest-Neighbours imputation via scikit-learn's KNNImputer.

Extracted from the research pipeline. The public entry point is
``apply_knn_imputation``; the imputation logic is preserved exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ..metrics import calculate_classification_metrics, process_for_classification
from .simple import ImputationResult

__all__ = ["apply_knn_imputation"]


def apply_knn_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    validation_df: Optional[pd.DataFrame] = None,
    validation_masks: Optional[Dict[str, pd.Series]] = None,
    original_values: Optional[Dict[str, pd.Series]] = None,
    n_neighbors: int = 5,
    weights: str = 'uniform',
    metric: str = 'nan_euclidean',
) -> ImputationResult:
    """
    Apply KNN imputation using scikit-learn's KNNImputer

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
    n_neighbors : int
        Number of neighboring samples to use for imputation
    weights : str or callable
        Weight function used in prediction. Possible values:
        - 'uniform': uniform weights (default)
        - 'distance': weight points by the inverse of their distance
    metric : str
        Distance metric for searching neighbors. Possible values:
        - 'nan_euclidean': euclidean distance ignoring NaNs (default)
        - 'euclidean': standard euclidean distance

    Returns:
    --------
    imputed_df : pandas.DataFrame
        Data with imputed values
    validation_results : dict, optional
        Validation results if validation data provided
    """
    try:
        # Extract relevant columns including potential predictors
        # Use all columns except those with excessive missing values
        threshold = 0.5  # Columns with more than 50% missing values are excluded
        columns_to_use = [col for col in df.columns
                            if df[col].isna().mean() < threshold]

        # Ensure all columns_to_impute are included
        for col in columns_to_impute:
            if col not in columns_to_use:
                columns_to_use.append(col)

        # Extract subset of data
        X = df[columns_to_use].copy()

        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        print(f"KNN Imputation: Using {len(columns_to_use)} columns, imputing {len(columns_to_impute)} columns")

        # Adjust n_neighbors based on available data
        # For each column to impute, check how many complete cases we have
        min_complete_cases = float('inf')
        for col in columns_to_impute:
            if col in X.columns:
                # Count rows where this column has data and at least one other column has data
                col_observed = X[col].notna()
                other_cols = [c for c in X.columns if c != col]
                if other_cols:
                    other_observed = X[other_cols].notna().any(axis=1)
                    complete_cases = (col_observed & other_observed).sum()
                else:
                    complete_cases = col_observed.sum()
                min_complete_cases = min(min_complete_cases, complete_cases)

        # Adjust n_neighbors to be at most the number of complete cases - 1
        if min_complete_cases != float('inf') and min_complete_cases > 0:
            n_neighbors = min(n_neighbors, max(1, min_complete_cases - 1))

        print(f"KNN parameters: n_neighbors={n_neighbors}, weights='{weights}', metric='{metric}'")

        # Initialize KNNImputer
        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            keep_empty_features=True  # Keep features that are all NaN
        )

        # Fit and transform
        print("Training KNN imputation model...")
        X_imputed_array = imputer.fit_transform(X.values)

        # Convert back to DataFrame
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)

        # Create imputed dataframe - only replace missing values
        imputed_df = df.copy()
        for col in columns_to_impute:
            if col in X_imputed.columns:
                missing_mask = df[col].isna()
                imputed_df.loc[missing_mask, col] = X_imputed.loc[missing_mask, col]

        # Print imputation statistics
        print("KNN Imputation completed successfully!")
        for col in columns_to_impute:
            if col in df.columns:
                original_missing = df[col].isna().sum()
                final_missing = imputed_df[col].isna().sum()
                imputed_count = original_missing - final_missing
                print(f"  {col}: {imputed_count} values imputed")

        # Validate if validation data provided
        validation_results = None
        if validation_df is not None and validation_masks is not None and original_values is not None:
            validation_results = {}

            # Extract validation data
            X_val = validation_df[columns_to_use].copy()

            # Ensure all validation columns are numeric
            for col in X_val.columns:
                X_val[col] = pd.to_numeric(X_val[col], errors='coerce')

            # Check validation data for n_neighbors adjustment
            val_min_complete_cases = float('inf')
            for col in columns_to_impute:
                if col in X_val.columns:
                    col_observed = X_val[col].notna()
                    other_cols = [c for c in X_val.columns if c != col]
                    if other_cols:
                        other_observed = X_val[other_cols].notna().any(axis=1)
                        complete_cases = (col_observed & other_observed).sum()
                    else:
                        complete_cases = col_observed.sum()
                    val_min_complete_cases = min(val_min_complete_cases, complete_cases)

            # Adjust n_neighbors for validation
            val_n_neighbors = n_neighbors
            if val_min_complete_cases != float('inf') and val_min_complete_cases > 0:
                val_n_neighbors = min(n_neighbors, max(1, val_min_complete_cases - 1))

            # Create a new imputer for validation data
            val_imputer = KNNImputer(
                n_neighbors=val_n_neighbors,
                weights=weights,
                metric=metric,
                keep_empty_features=True
            )

            # Impute validation data
            print("Imputing validation data...")
            X_val_imputed_array = val_imputer.fit_transform(X_val.values)
            X_val_imputed = pd.DataFrame(X_val_imputed_array, columns=X_val.columns, index=X_val.index)

            # Compare imputed values to real values
            with tqdm(columns_to_impute, desc="Validating results") as pbar:
                for col in pbar:
                    pbar.set_description(f"Validating {col}")
                    # Get indices where values were artificially set to NaN
                    mask = validation_masks[col] & validation_df[col].isna()

                    if mask.sum() == 0:
                        validation_results[col] = {
                            'error': "No artificially missing values"
                        }
                        continue

                    real_vals = original_values[col][mask]
                    imputed_vals = X_val_imputed.loc[mask, col]

                    # Calculate MAE and RMSE
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
                        'imputed_distribution': pd.Series(imputed_vals).describe()
                    }

                    # Update progress
                    pbar.set_postfix({"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "Acc": f"{classification_metrics['accuracy']:.4f}"})

        return imputed_df, validation_results

    except Exception as e:
        print(f"Error in KNN imputation: {str(e)}")
        import traceback
        traceback.print_exc()

        # Fallback to simple mean imputation
        print("Falling back to simple mean imputation")
        result_df = df.copy()
        for col in columns_to_impute:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                result_df[col] = result_df[col].fillna(result_df[col].mean())
        return result_df, None
