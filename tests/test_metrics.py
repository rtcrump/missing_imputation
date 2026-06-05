"""Tests for the ordinal-aware classification metrics (QWK, within-1)."""

import numpy as np

from missing_imputation.metrics import (
    calculate_classification_metrics,
    process_for_classification,
)


def test_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 2, 1, 0])
    m = calculate_classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["qwk"] == 1.0
    assert m["within1_accuracy"] == 1.0


def test_within1_counts_near_misses():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([1, 1, 2, 3, 3])  # two off-by-one, three exact
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["accuracy"] < 1.0
    # every prediction is within one category of the truth
    assert m["within1_accuracy"] == 1.0


def test_qwk_zero_for_constant_prediction():
    # A constant prediction yields no agreement structure -> kappa 0.
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([2, 2, 2, 2, 2])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["qwk"] == 0.0


def test_qwk_present_keys():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 1, 3, 4])
    m = calculate_classification_metrics(y_true, y_pred)
    for key in ("qwk", "within1_accuracy", "accuracy", "auc_multiclass"):
        assert key in m


def test_process_for_classification_rounds_and_clips():
    out = process_for_classification(np.array([-0.4, 0.6, 2.5, 4.9, 3.2]))
    assert out.tolist() == [0, 1, 2, 4, 3]
    assert out.dtype.kind in "iu"
