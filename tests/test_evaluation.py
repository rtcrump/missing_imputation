"""Tests for the benchmarking / evaluation harness.

Each evaluator is exercised on synthetic data with a small set of fast classical
methods. The goal is to confirm the harness runs end-to-end and returns
well-formed, finite results — not to assert on specific numeric values.
"""

import numpy as np
import pytest

import missing_imputation as mi
from missing_imputation.columns import FACTE_COLUMNS

# A small column subset keeps the (stochastic) methods fast.
COLS = FACTE_COLUMNS[:12]
# mean/median/knn are deterministic-ish and fast; skip RF-based mice/softimpute.
FAST_METHODS = {k: mi.METHODS[k] for k in ("mean", "median", "knn")}


@pytest.fixture(scope="module")
def data():
    return mi.make_synthetic_facte(n_patients=60, n_visits=6, missing_rate=0.25, seed=7)


def test_sparse_validation_returns_aggregated_metrics(data):
    res = mi.evaluate_with_sparse_validation(
        data, COLS, methods=FAST_METHODS, n_folds=3
    )
    assert set(res) == set(FAST_METHODS)
    for method, r in res.items():
        # Aggregated summary metrics are present and finite.
        for key in ("avg_mae", "avg_rmse", "avg_accuracy", "avg_qwk",
                    "avg_within1_accuracy"):
            assert key in r, f"{method} missing {key}"
            assert np.isfinite(r[key]), f"{method}.{key} not finite"
        assert r["avg_mae"] >= 0
        # within-1 accuracy is a fraction.
        assert 0.0 <= r["avg_within1_accuracy"] <= 1.0


def test_sparse_validation_defaults_to_registry(data):
    # With methods=None, every registered method is benchmarked.
    res = mi.evaluate_with_sparse_validation(
        data, COLS[:6], methods=None, n_folds=2
    )
    assert set(res) == set(mi.METHODS)


def test_trajectory_fidelity(data):
    res = mi.evaluate_trajectory_fidelity(
        data, COLS, methods=FAST_METHODS, n_patients_sample=60
    )
    assert set(res) == set(FAST_METHODS)
    for r in res.values():
        assert r["n_patients"] > 0
        assert np.isfinite(r["trajectory_mae"])
        assert r["trajectory_mae"] >= 0


def test_temporal_smoothness(data):
    imputed = mi.run_all_methods(data, COLS, methods=FAST_METHODS)
    assert set(imputed) == set(FAST_METHODS)
    res = mi.evaluate_temporal_smoothness(data, imputed, COLS)
    assert set(res) == set(FAST_METHODS)
    for r in res.values():
        assert r["n_patients"] > 0
        assert np.isfinite(r["mean_smoothness"])


def test_missing_pattern_robustness(data):
    res = mi.evaluate_missing_pattern_robustness(
        data, COLS, methods=FAST_METHODS, n_folds=3
    )
    assert set(res) == {"monotone", "intermittent"}
    # At least one pattern group should be populated on this data.
    assert res["monotone"] or res["intermittent"]
    for group in res.values():
        for method_metrics in group.values():
            assert np.isfinite(method_metrics["mae"])
