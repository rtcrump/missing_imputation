"""Tests for the synthetic data generator."""

import numpy as np

from missing_imputation.columns import FACTE_COLUMNS, LIKERT_MAX, LIKERT_MIN, VISITS
from missing_imputation.synthetic import introduce_missingness, make_synthetic_facte


def test_shape_and_columns():
    df = make_synthetic_facte(n_patients=20, n_visits=4, missing_rate=0.0)
    assert len(df) == 20 * 4
    for col in ["id", "redcap_event_name"]:
        assert col in df.columns
    for col in FACTE_COLUMNS:
        assert col in df.columns


def test_fully_observed_when_no_missing():
    df = make_synthetic_facte(n_patients=15, missing_rate=0.0)
    assert df[FACTE_COLUMNS].isna().sum().sum() == 0


def test_values_in_likert_range():
    df = make_synthetic_facte(n_patients=25, missing_rate=0.0)
    vals = df[FACTE_COLUMNS].to_numpy()
    assert np.nanmin(vals) >= LIKERT_MIN
    assert np.nanmax(vals) <= LIKERT_MAX


def test_visits_are_known():
    df = make_synthetic_facte(n_patients=10, n_visits=5)
    assert set(df["redcap_event_name"]).issubset(set(VISITS))


def test_reproducible_with_seed():
    a = make_synthetic_facte(n_patients=10, seed=7)
    b = make_synthetic_facte(n_patients=10, seed=7)
    assert a.equals(b)


def test_introduce_missingness_does_not_mutate_input():
    df = make_synthetic_facte(n_patients=10, missing_rate=0.0)
    before = df[FACTE_COLUMNS].isna().sum().sum()
    out = introduce_missingness(df, FACTE_COLUMNS, rate=0.3, seed=1)
    assert before == 0
    assert out[FACTE_COLUMNS].isna().sum().sum() > 0


def test_invalid_n_visits_raises():
    import pytest
    with pytest.raises(ValueError, match="n_visits must be between"):
        make_synthetic_facte(n_visits=99)
    with pytest.raises(ValueError, match="n_visits must be between"):
        make_synthetic_facte(n_visits=0)
