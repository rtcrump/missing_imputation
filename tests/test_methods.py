"""Smoke + validation tests for the classical imputation methods.

These confirm that each method (a) runs on synthetic data without error and
fills every missing FACT-E cell, and (b) returns a well-formed validation
results dict when given held-out values.
"""

import numpy as np
import pandas as pd
import pytest

import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

# Use a modest subset of columns to keep the (stochastic, RF-based) methods fast.
COLS = FACTE_COLUMNS[:12]

METHOD_NAMES = ["mean", "median", "knn", "mice", "softimpute"]


@pytest.fixture(scope="module")
def data():
    return mi.make_synthetic_facte(n_patients=50, n_visits=4, missing_rate=0.2, seed=3)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_fills_all_missing(name, data):
    method = mi.METHODS[name]
    imputed, validation = method(data, COLS)
    assert isinstance(imputed, pd.DataFrame)
    assert imputed[COLS].isna().sum().sum() == 0
    # Non-imputed structural columns are preserved.
    assert "id" in imputed.columns
    assert "redcap_event_name" in imputed.columns
    assert len(imputed) == len(data)
    # Without validation data, the second return value is None.
    assert validation is None


def _make_validation(df, cols, frac=0.2, seed=11):
    """Hold out a random subset of *observed* cells for validation scoring.

    Returns (validation_df, masks, originals) matching the method API:
    masks[col] marks which cells were artificially blanked, originals[col]
    holds the true values at those cells.
    """
    rng = np.random.default_rng(seed)
    validation_df = df.copy()
    masks = {}
    originals = {}
    for col in cols:
        observed = df[col].notna().to_numpy()
        draw = rng.random(len(df)) < frac
        mask = observed & draw
        masks[col] = pd.Series(mask, index=df.index)
        originals[col] = df[col].copy()
        validation_df.loc[mask, col] = np.nan
    return validation_df, masks, originals


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_validation_results(name, data):
    method = mi.METHODS[name]
    validation_df, masks, originals = _make_validation(data, COLS)
    _, validation = method(data, COLS, validation_df, masks, originals)
    assert validation is not None
    for col in COLS:
        res = validation[col]
        if "error" in res:
            continue
        # Core metrics are present and numeric.
        assert "mae" in res and "rmse" in res and "accuracy" in res
        assert np.isfinite(res["mae"])
        assert res["mae"] >= 0


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_validation_includes_ordinal_metrics(name, data):
    method = mi.METHODS[name]
    validation_df, masks, originals = _make_validation(data, COLS)
    _, validation = method(data, COLS, validation_df, masks, originals)
    assert validation is not None
    for col in COLS:
        res = validation[col]
        if "error" in res:
            continue
        assert "qwk" in res, f"qwk missing from {name} validation for {col}"
        assert "within1_accuracy" in res, f"within1_accuracy missing from {name} validation for {col}"
        assert np.isfinite(res["within1_accuracy"])


@pytest.mark.parametrize("name", ["mean", "median", "knn", "mice", "softimpute"])
def test_all_nan_column_raises(name, data):
    method = mi.METHODS[name]
    df = data.copy()
    df[COLS[0]] = np.nan
    with pytest.raises(ValueError, match="entirely NaN"):
        method(df, COLS[:1])


def test_frob_identical_iterates_returns_zero():
    from missing_imputation.methods.softimpute import frob
    U = np.eye(3, 2)
    Dsq = np.array([[3.0], [1.0]])
    V = np.eye(4, 2)
    assert frob(U, Dsq, V, U, Dsq, V) == pytest.approx(0.0, abs=1e-12)


def test_frob_different_iterates_returns_positive():
    from missing_imputation.methods.softimpute import frob
    U = np.eye(3, 2)
    Dsq_old = np.array([[3.0], [1.0]])
    Dsq_new = np.array([[5.0], [2.0]])
    V = np.eye(4, 2)
    assert frob(U, Dsq_old, V, U, Dsq_new, V) > 0


def test_registry_matches_callables():
    assert set(mi.METHODS) == set(METHOD_NAMES)
    for fn in mi.METHODS.values():
        assert callable(fn)
