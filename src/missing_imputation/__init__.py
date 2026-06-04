"""missing_imputation - imputation of missing longitudinal clinical PRO data.

A toolkit for imputing missing values in longitudinal patient-reported outcome
(PRO) data, built around the FACT-E (Functional Assessment of Cancer Therapy -
Esophageal) questionnaire. It compares classical statistical imputation against
deep-learning and LLM-based methods.

Quick start
-----------
>>> from missing_imputation import make_synthetic_facte, apply_knn_imputation
>>> from missing_imputation.columns import FACTE_COLUMNS
>>> df = make_synthetic_facte(n_patients=40, missing_rate=0.2)
>>> imputed, _ = apply_knn_imputation(df, FACTE_COLUMNS)
>>> imputed[FACTE_COLUMNS].isna().sum().sum()
0

Every method shares the signature
``apply_<name>_imputation(df, columns_to_impute, validation_df=None,
validation_masks=None, original_values=None, ...)`` and returns
``(imputed_df, validation_results)``.
"""

from . import columns
from .methods import (
    SoftImpute,
    apply_knn_imputation,
    apply_mean_imputation,
    apply_median_imputation,
    apply_mice_imputation,
    apply_softimpute_imputation,
)
from .metrics import calculate_classification_metrics, process_for_classification
from .synthetic import introduce_missingness, make_synthetic_facte

__version__ = "0.1.0"

# Registry mapping method name -> callable, for programmatic dispatch / CLI.
METHODS = {
    "mean": apply_mean_imputation,
    "median": apply_median_imputation,
    "knn": apply_knn_imputation,
    "mice": apply_mice_imputation,
    "softimpute": apply_softimpute_imputation,
}

__all__ = [
    "__version__",
    "columns",
    "METHODS",
    "apply_mean_imputation",
    "apply_median_imputation",
    "apply_knn_imputation",
    "apply_mice_imputation",
    "apply_softimpute_imputation",
    "SoftImpute",
    "process_for_classification",
    "calculate_classification_metrics",
    "make_synthetic_facte",
    "introduce_missingness",
]
