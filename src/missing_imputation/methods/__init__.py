"""Imputation methods.

Each method exposes an ``apply_<name>_imputation(df, columns_to_impute, ...)``
function returning ``(imputed_df, validation_results)``.

Classical methods (mean, median, KNN, MICE, SoftImpute) depend only on the core
dependencies and are imported here directly. Deep-learning and LLM-based methods
(VAE, deep autoencoder, BayesianPCA, LSTM, longitudinal MICE, Gemma/LoRA,
TimesFM) live in the research notebooks under ``notebook/`` and will be exposed
here as they are extracted; they require the optional ``deep`` / ``llm`` extras.
"""

from .knn import apply_knn_imputation
from .mice import apply_mice_imputation
from .simple import apply_mean_imputation, apply_median_imputation
from .softimpute import SoftImpute, apply_softimpute_imputation

__all__ = [
    "apply_mean_imputation",
    "apply_median_imputation",
    "apply_knn_imputation",
    "apply_mice_imputation",
    "apply_softimpute_imputation",
    "SoftImpute",
]
