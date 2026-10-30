"""Tests for threshold conversion, metric correctness, and confusion-matrix orientation."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    evaluate,
    roc_auc_from_probabilities,
    threshold_predictions,
)


def test_threshold_predictions_boundary_inclusive():
    probs = np.array([0.10, 0.29, 0.30, 0.31, 0.95])
    preds = threshold_predictions(probs, 0.30)
    # probability >= threshold counts as positive.
    assert preds.tolist() == [0, 0, 1, 1, 1]


def test_confusion_matrix_orientation():
    # y_true: 0,0,1,1 ; y_pred: 0,1,1,1
    # Expected: TN=1, FP=1, FN=0, TP=2
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.8, 0.9, 0.7])
    ev = evaluate(y_true, probs, "Test model", "Validation / Interactive", threshold=0.5)
    assert ev.confusion_matrix == [[1, 1], [0, 2]]


def test_known_metric_values():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.8, 0.9, 0.7])  # preds at 0.5: [0,1,1,1]
    ev = evaluate(y_true, probs, "M", "Validation / Interactive")
    assert ev.accuracy == pytest.approx(3 / 4)
    assert ev.precision == pytest.approx(2 / 3)
    assert ev.recall == pytest.approx(1.0)
    assert ev.f1_score == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))
    assert ev.roc_auc == pytest.approx(roc_auc_score(y_true, probs))


def test_roc_auc_unchanged_when_threshold_moves():
    rng = np.random.default_rng(0)
    probs = rng.random(500)
    y_true = (rng.random(500) < probs).astype(int)
    aucs = {
        evaluate(y_true, probs, "M", "V", t).roc_auc for t in (0.05, 0.3, 0.5, 0.7, 0.95)
    }
    assert len(aucs) == 1  # threshold-independent
    assert aucs.pop() == pytest.approx(roc_auc_score(y_true, probs))


def test_no_positive_predictions_does_not_crash():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.05, 0.05, 0.05, 0.05])
    ev = evaluate(y_true, probs, "M", "V", threshold=0.5)
    assert ev.precision == 0.0  # zero_division=0, reported not crashed
    assert ev.recall == 0.0
    assert ev.predicted_churn_count == 0
    assert ev.confusion_matrix == [[2, 0], [2, 0]]


def test_roc_auc_none_for_single_class_sample():
    assert roc_auc_from_probabilities([0, 0, 0], [0.1, 0.4, 0.9]) is None


def test_predicted_churn_count_and_rate():
    y_true = np.array([0, 1, 1, 1])
    probs = np.array([0.6, 0.7, 0.2, 0.9])
    ev = evaluate(y_true, probs, "M", "V", threshold=0.5)
    assert ev.predicted_churn_count == 3
    assert ev.predicted_churn_rate == pytest.approx(0.75)


def test_threshold_moves_labels_but_not_scores():
    rng = np.random.default_rng(1)
    probs = rng.random(200)
    y_true = (rng.random(200) < probs).astype(int)
    low = evaluate(y_true, probs, "M", "V", 0.2)
    high = evaluate(y_true, probs, "M", "V", 0.8)
    assert low.predicted_churn_count > high.predicted_churn_count
    assert low.roc_auc == high.roc_auc
    assert low.recall >= high.recall  # lower threshold cannot miss more churners
