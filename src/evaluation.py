"""Threshold conversion and metric calculation.

Owns: all metric computation. The critical behavioral rule — changing the
threshold only changes binary predictions and threshold-dependent metrics;
ROC-AUC stays probability-based and stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


@dataclass
class EvaluationResult:
    """Consistent shape for both models so the UI can compare side by side."""

    model_name: str
    split_label: str          # "Validation / Interactive" or "Test / Final Evaluation"
    threshold: float
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None     # None when undefined (single-class sample)
    confusion_matrix: list[list[int]]   # [[TN, FP], [FN, TP]]
    predicted_churn_count: int
    predicted_churn_rate: float

    def metrics_row(self) -> dict:
        """One row for comparison tables/charts."""
        return {
            "Model": self.model_name,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1-score": self.f1_score,
            "ROC-AUC": self.roc_auc if self.roc_auc is not None else float("nan"),
        }


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions: churn-risk when probability >= threshold."""
    return (np.asarray(probabilities) >= threshold).astype(int)


def roc_auc_from_probabilities(y_true, probabilities) -> float | None:
    """ROC-AUC from probability scores. None if undefined (one class present)."""
    labels = np.unique(np.asarray(y_true))
    if labels.size < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def roc_curve_points(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate) for the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return fpr, tpr


def evaluate(
    y_true,
    probabilities: np.ndarray,
    model_name: str,
    split_label: str,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> EvaluationResult:
    """Compute the full metric set at a given threshold.

    Uses zero_division=0 for precision/recall/F1 so a threshold that flags
    nobody yields a reported 0 rather than a crash.
    """
    y_true = np.asarray(y_true)
    y_pred = threshold_predictions(probabilities, threshold)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    predicted_positive = int(y_pred.sum())

    return EvaluationResult(
        model_name=model_name,
        split_label=split_label,
        threshold=float(threshold),
        sample_count=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc_from_probabilities(y_true, probabilities),
        confusion_matrix=cm.tolist(),
        predicted_churn_count=predicted_positive,
        predicted_churn_rate=float(predicted_positive / len(y_true)) if len(y_true) else 0.0,
    )
