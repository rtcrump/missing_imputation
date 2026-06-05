"""Longitudinal-specific evaluation metrics.

These complement the cross-sectional :func:`evaluate_with_sparse_validation` by
scoring how well each method preserves *temporal* structure in patient
trajectories:

- :func:`evaluate_trajectory_fidelity` — mask each patient's middle visit and
  measure how well the imputed values match the true trajectory shape.
- :func:`evaluate_temporal_smoothness` — penalise unrealistic visit-to-visit
  jumps in already-imputed frames.
- :func:`evaluate_missing_pattern_robustness` — compare accuracy on monotone
  (permanent dropout) vs intermittent (sporadic) missingness patterns.

Logic is preserved from the research pipeline
(``notebook/Timeseries/imputation21``); the packaging changes are the same as in
``sparse.py``: methods are a parameter (defaulting to the classical registry),
and the canonical visit ordering comes from :data:`missing_imputation.columns.VISIT_ORDER`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..columns import VISIT_ORDER
from .sparse import _resolve_methods, evaluate_with_sparse_validation

__all__ = [
    "run_all_methods",
    "evaluate_trajectory_fidelity",
    "evaluate_temporal_smoothness",
    "evaluate_missing_pattern_robustness",
]


def run_all_methods(
    df,
    columns_to_impute: List[str],
    methods: Optional[Dict[str, Callable]] = None,
) -> Dict[str, "object"]:
    """Run every method once and return ``{method_name: imputed_df}``.

    Convenience helper for producing the ``imputed_dfs`` input that
    :func:`evaluate_temporal_smoothness` expects.
    """
    methods = _resolve_methods(methods)
    imputed = {}
    for name, func in methods.items():
        try:
            method_df = df.copy()
            if "qol_date" in method_df.columns:
                method_df = method_df.drop(columns=["qol_date"])
            imputed_df, _ = func(method_df, columns_to_impute)
            imputed[name] = imputed_df
        except Exception:  # noqa: BLE001 (skip a method that fails, mirror research)
            continue
    return imputed


def evaluate_trajectory_fidelity(
    df,
    columns_to_impute: List[str],
    methods: Optional[Dict[str, Callable]] = None,
    patient_col: str = "id",
    event_col: str = "redcap_event_name",
    n_patients_sample: int = 400,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Measure how well methods preserve individual patient trajectories.

    For patients with >= 3 observed timepoints, all of their *middle* visits are
    masked simultaneously, each method imputes the full frame once, and the
    recovered middle-visit values are compared (MAE + Pearson correlation) to the
    originally-observed truth.

    Returns ``{method_name: {'trajectory_mae', 'trajectory_mae_std',
    'trajectory_correlation', 'n_patients'}}``.
    """
    methods = _resolve_methods(methods)
    results: Dict[str, dict] = {}

    # Step 1: eligible patients (>= 3 distinct timepoints)
    visit_counts = df.groupby(patient_col)[event_col].nunique()
    eligible_patients = visit_counts[visit_counts >= 3].index.tolist()
    if len(eligible_patients) == 0:
        if verbose:
            print("No patients with >=3 timepoints for trajectory evaluation")
        return results

    if len(eligible_patients) > n_patients_sample:
        np.random.seed(42)
        eligible_patients = np.random.choice(
            eligible_patients, n_patients_sample, replace=False
        ).tolist()

    # Step 2: mask each eligible patient's middle visit
    masked_data = df.copy()
    middle_visit_indices = {}
    for pid in eligible_patients:
        patient_data = df[df[patient_col] == pid].copy()
        if len(patient_data) < 3:
            continue
        if event_col in patient_data.columns:
            patient_data["_sort_key"] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
            patient_data = patient_data.sort_values("_sort_key")
        mid_idx = len(patient_data) // 2
        middle_visit_idx = patient_data.index[mid_idx]
        orig_vals = df.loc[middle_visit_idx, columns_to_impute]
        if orig_vals.isna().all():
            continue
        middle_visit_indices[pid] = {
            "row_idx": middle_visit_idx,
            "original_values": orig_vals.astype(float).values.copy(),
            "obs_mask": ~orig_vals.isna().values,
        }
        masked_data.loc[middle_visit_idx, columns_to_impute] = np.nan

    if len(middle_visit_indices) == 0:
        if verbose:
            print("No valid patients to evaluate after masking")
        return results

    # Step 3: run each method once on the shared masked frame
    trajectory_errors = {name: [] for name in methods}
    trajectory_corrs = {name: [] for name in methods}

    for method_name, method_func in methods.items():
        try:
            method_df = masked_data.copy()
            if "qol_date" in method_df.columns:
                method_df = method_df.drop(columns=["qol_date"])
            imputed_df, _ = method_func(method_df, columns_to_impute)

            # Step 4: evaluate every masked patient
            for pid, info in middle_visit_indices.items():
                row_idx = info["row_idx"]
                orig_middle = info["original_values"]
                obs_mask = info["obs_mask"]
                if obs_mask.sum() == 0:
                    continue
                imp_middle = imputed_df.loc[row_idx, columns_to_impute].values.astype(float)
                orig_obs = orig_middle[obs_mask]
                imp_obs = imp_middle[obs_mask]
                if np.isnan(imp_obs).any():
                    continue
                trajectory_errors[method_name].append(np.abs(imp_obs - orig_obs).mean())
                if len(orig_obs) > 1 and orig_obs.std() > 0:
                    corr = np.corrcoef(orig_obs, imp_obs)[0, 1]
                    if not np.isnan(corr):
                        trajectory_corrs[method_name].append(corr)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  Error with {method_name}: {e}")

    # Step 5: aggregate
    for method_name in methods:
        errors = trajectory_errors[method_name]
        corrs = trajectory_corrs[method_name]
        if errors:
            results[method_name] = {
                "trajectory_mae": np.mean(errors),
                "trajectory_mae_std": np.std(errors),
                "trajectory_correlation": np.mean(corrs) if corrs else np.nan,
                "n_patients": len(errors),
            }
    return results


def evaluate_temporal_smoothness(
    df,
    imputed_dfs: Dict[str, "object"],
    columns_to_impute: List[str],
    patient_col: str = "id",
    event_col: str = "redcap_event_name",
) -> Dict[str, dict]:
    """Score visit-to-visit smoothness of already-imputed trajectories.

    Lower mean smoothness (average absolute consecutive difference across
    features and timepoints) indicates more realistic, less jumpy trajectories.

    Parameters
    ----------
    df : pandas.DataFrame
        Original frame (unused beyond signature compatibility with the research
        pipeline; smoothness is computed from ``imputed_dfs``).
    imputed_dfs : dict
        ``{method_name: imputed_df}`` — e.g. the output of :func:`run_all_methods`.

    Returns
    -------
    dict
        ``{method_name: {'mean_smoothness', 'std_smoothness', 'n_patients'}}``.
    """
    results: Dict[str, dict] = {}
    for method_name, imputed_df in imputed_dfs.items():
        smoothness_scores = []
        for _pid, patient_data in imputed_df.groupby(patient_col):
            if len(patient_data) < 2:
                continue
            if event_col in patient_data.columns:
                patient_data = patient_data.copy()
                patient_data["_sort_key"] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
                patient_data = patient_data.sort_values("_sort_key")
            trajectory = patient_data[columns_to_impute].values
            diffs = np.abs(np.diff(trajectory, axis=0))
            smoothness_scores.append(np.nanmean(diffs))
        if smoothness_scores:
            results[method_name] = {
                "mean_smoothness": np.mean(smoothness_scores),
                "std_smoothness": np.std(smoothness_scores),
                "n_patients": len(smoothness_scores),
            }
    return results


def evaluate_missing_pattern_robustness(
    df,
    columns_to_impute: List[str],
    methods: Optional[Dict[str, Callable]] = None,
    patient_col: str = "id",
    event_col: str = "redcap_event_name",
    n_folds: int = 5,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Compare method accuracy across monotone vs intermittent missingness.

    Patients are classified by their observed-visit pattern:

    - **monotone** — contiguous visits from baseline then permanent dropout.
    - **intermittent** — gaps between visits (dropout then return / sporadic).

    :func:`evaluate_with_sparse_validation` is then run separately on each group.

    Returns ``{'monotone': {method: {'mae', 'rmse'}}, 'intermittent': {...}}``.
    """
    methods = _resolve_methods(methods)

    monotone_patients = []
    intermittent_patients = []
    for pid, patient_data in df.groupby(patient_col):
        if len(patient_data) < 2:
            continue
        if event_col in patient_data.columns:
            patient_data = patient_data.copy()
            patient_data["_sort_key"] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
            patient_data = patient_data.sort_values("_sort_key")
            visit_indices = patient_data["_sort_key"].values
        else:
            visit_indices = np.arange(len(patient_data))

        gaps = np.diff(visit_indices)
        has_gaps = np.any(gaps > 1)
        if has_gaps:
            intermittent_patients.append(pid)
        else:
            min_visit = min(visit_indices)
            max_visit = max(visit_indices)
            if min_visit == 0 and max_visit < 9:
                monotone_patients.append(pid)
            else:
                intermittent_patients.append(pid)

    if verbose:
        print(f"Monotone dropout patients: {len(monotone_patients)}")
        print(f"Intermittent missing patients: {len(intermittent_patients)}")

    results: Dict[str, dict] = {"monotone": {}, "intermittent": {}}
    for pattern, patient_list in [("monotone", monotone_patients),
                                  ("intermittent", intermittent_patients)]:
        if len(patient_list) == 0:
            continue
        pattern_df = df[df[patient_col].isin(patient_list)].copy()
        pattern_results = evaluate_with_sparse_validation(
            pattern_df, columns_to_impute, methods=methods,
            n_folds=n_folds, verbose=verbose,
        )
        for method, metrics in pattern_results.items():
            if "avg_mae" in metrics:
                results[pattern].setdefault(method, {})
                results[pattern][method]["mae"] = metrics["avg_mae"]
                results[pattern][method]["rmse"] = metrics["avg_rmse"]
    return results
