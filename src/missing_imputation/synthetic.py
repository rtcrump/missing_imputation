"""Synthetic FACT-E longitudinal data generator.

Produces a PHI-free dataset that mimics the *shape* of the real clinical data
(patient ids, REDCap visit events, 44 ordinal 0-4 FACT-E items, longitudinal
structure with realistic missingness) so examples and tests can run without any
access to the real clinical files.

All values are randomly generated. No real patient data is used or reproduced.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .columns import FACTE_COLUMNS, LIKERT_MAX, LIKERT_MIN, VISITS

__all__ = ["make_synthetic_facte", "introduce_missingness"]


def make_synthetic_facte(
    n_patients: int = 60,
    n_visits: int = 5,
    missing_rate: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic longitudinal FACT-E dataset.

    Each patient has a latent well-being trajectory that drifts smoothly across
    visits; item responses are noisy ordinal samples around that latent state.
    This yields cross-item and cross-visit correlation structure similar to the
    real data, which the imputation methods rely on.

    Parameters
    ----------
    n_patients : int
        Number of synthetic patients.
    n_visits : int
        Number of visits per patient (first ``n_visits`` of the canonical
        visit order).
    missing_rate : float
        Fraction of FACT-E item cells set to NaN (completely-at-random), in
        addition to the latent structure. Set to 0 for a fully observed frame.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: ``id``, ``redcap_event_name``, and the 44 FACT-E item columns.
    """
    rng = np.random.default_rng(seed)
    visits = VISITS[:n_visits]
    n_items = len(FACTE_COLUMNS)

    # Per-item baseline difficulty (mean response level), in [1, 3].
    item_base = rng.uniform(1.0, 3.0, size=n_items)

    rows = []
    for pid in range(1, n_patients + 1):
        # Latent patient well-being offset and a per-visit drift.
        patient_offset = rng.normal(0.0, 0.7)
        drift = rng.normal(0.0, 0.15)
        for v_idx, visit in enumerate(visits):
            latent = patient_offset + drift * v_idx
            # Item-level noisy continuous score, then round to the Likert scale.
            noise = rng.normal(0.0, 0.6, size=n_items)
            cont = item_base + latent + noise
            vals = np.clip(np.round(cont), LIKERT_MIN, LIKERT_MAX).astype(float)
            row = {"id": pid, "redcap_event_name": visit}
            row.update(dict(zip(FACTE_COLUMNS, vals)))
            rows.append(row)

    df = pd.DataFrame(rows, columns=["id", "redcap_event_name"] + FACTE_COLUMNS)

    if missing_rate and missing_rate > 0:
        df = introduce_missingness(df, FACTE_COLUMNS, rate=missing_rate, seed=seed)

    return df


def introduce_missingness(
    df: pd.DataFrame,
    columns: List[str],
    rate: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Set a random fraction of cells in ``columns`` to NaN (MCAR).

    Returns a copy; the input frame is not modified.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out
