"""Benchmarking and evaluation harness for imputation methods.

Two complementary families of evaluators, both of which default to scoring the
classical :data:`missing_imputation.METHODS` registry but accept any
``{name: apply_*_imputation}`` mapping via the ``methods`` argument:

Cross-sectional
    :func:`evaluate_with_sparse_validation` — fold-wise masking of observed
    values, scored with MAE/RMSE and the full classification suite (accuracy,
    AUC, quadratic weighted kappa, within-1-category accuracy, ...).

Longitudinal
    :func:`evaluate_trajectory_fidelity`, :func:`evaluate_temporal_smoothness`,
    and :func:`evaluate_missing_pattern_robustness` — preservation of per-patient
    temporal structure. :func:`run_all_methods` is a helper that produces the
    ``{name: imputed_df}`` mapping the smoothness metric consumes.
"""

from .longitudinal import (
    evaluate_missing_pattern_robustness,
    evaluate_temporal_smoothness,
    evaluate_trajectory_fidelity,
    run_all_methods,
)
from .sparse import evaluate_with_sparse_validation

__all__ = [
    "evaluate_with_sparse_validation",
    "evaluate_trajectory_fidelity",
    "evaluate_temporal_smoothness",
    "evaluate_missing_pattern_robustness",
    "run_all_methods",
]
