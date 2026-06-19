"""Evaluation metrics shared across imputation methods.

These helpers convert continuous imputed values into the ordinal 0-4 Likert
scale used by FACT-E items and compute classification-style metrics against the
held-out true values. Extracted verbatim from the research pipeline so results
remain identical to the published experiments.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

__all__ = ["process_for_classification", "calculate_classification_metrics"]


def process_for_classification(values, min_val: int = 0, max_val: int = 4) -> np.ndarray:
    """
    Process imputed values for classification evaluation

    Parameters:
    -----------
    values : array-like
        Raw imputed values
    min_val : int
        Minimum allowed value (default: 0)
    max_val : int
        Maximum allowed value (default: 4)

    Returns:
    --------
    numpy.ndarray
        Processed values as integers in range [min_val, max_val]
    """
    # Round to nearest integer
    rounded = np.round(values)
    # Clip to valid range
    clipped = np.clip(rounded, min_val, max_val)
    return clipped.astype(int)


def calculate_classification_metrics(
    y_true, y_pred, classes: Optional[Sequence[int]] = None
) -> dict:
    """
    Calculate classification metrics for multi-class problem

    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    classes : array-like, optional
        Class labels (default: [0, 1, 2, 3, 4])

    Returns:
    --------
    dict
        Dictionary containing classification metrics
    """
    if classes is None:
        classes = np.array([0, 1, 2, 3, 4])

    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Ordinal-aware metrics (FACT-E items are a 0-4 ordinal scale).
        # Quadratic weighted kappa rewards near-misses; "within-1" accuracy is the
        # fraction of predictions off by at most one Likert category. Formulas are
        # preserved verbatim from the research pipeline.
        qwk = (
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
            if len(np.unique(y_true)) > 1
            else 0.0
        )
        within1_accuracy = np.mean(np.abs(y_true - y_pred) <= 1)

        # Macro-averaged metrics (average across classes)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Weighted-averaged metrics (weighted by class frequency)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # For multi-class AUC, we need to binarize the labels
        try:
            # Only calculate AUC if we have more than one class present
            if len(np.unique(y_true)) > 1:
                y_true_bin = label_binarize(y_true, classes=classes)
                y_pred_bin = label_binarize(y_pred, classes=classes)

                # If only 2 classes present, reshape
                if y_true_bin.shape[1] == 1:
                    auc_score = roc_auc_score(y_true_bin, y_pred_bin)
                else:
                    # Multi-class AUC (macro average)
                    auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
            else:
                auc_score = np.nan
        except Exception:
            auc_score = np.nan

        # Per-class metrics
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Calculate sensitivity (recall) and specificity for each class
        per_class_metrics = {}
        for i, class_label in enumerate(classes):
            if i < cm.shape[0] and i < cm.shape[1]:
                tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
                fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                tn = cm.sum() - tp - fp - fn

                # Sensitivity (recall/true positive rate)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

                # Specificity (true negative rate)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # PPV (precision)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                # NPV
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                per_class_metrics[f'class_{class_label}'] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv
                }

        # Average across classes for summary
        avg_sensitivity = np.mean([metrics['sensitivity'] for metrics in per_class_metrics.values()])
        avg_specificity = np.mean([metrics['specificity'] for metrics in per_class_metrics.values()])
        avg_ppv = np.mean([metrics['ppv'] for metrics in per_class_metrics.values()])
        avg_npv = np.mean([metrics['npv'] for metrics in per_class_metrics.values()])

        return {
            'accuracy': accuracy,
            'auc_multiclass': auc_score,
            'qwk': qwk,
            'within1_accuracy': within1_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'avg_sensitivity': avg_sensitivity,
            'avg_specificity': avg_specificity,
            'avg_ppv': avg_ppv,
            'avg_npv': avg_npv,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm
        }

    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return {
            'accuracy': np.nan,
            'auc_multiclass': np.nan,
            'qwk': np.nan,
            'within1_accuracy': np.nan,
            'precision_macro': np.nan,
            'recall_macro': np.nan,
            'precision_weighted': np.nan,
            'recall_weighted': np.nan,
            'avg_sensitivity': np.nan,
            'avg_specificity': np.nan,
            'avg_ppv': np.nan,
            'avg_npv': np.nan,
            'per_class_metrics': {},
            'confusion_matrix': None
        }
