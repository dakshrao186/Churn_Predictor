"""Cautious, data-derived insight generation.

All statements are computed from the current cleaned dataset and fitted model
results. Association language only — never causal claims. Missing or
ambiguous evidence produces a neutral message rather than a conclusion.
"""

from __future__ import annotations

from src import config
from src.eda import churn_by_category, highest_churn_category, tenure_band_churn
from src.modeling import ModelResult, logistic_regression_coefficients


def contract_insight(df) -> str:
    """Highest observed churn-rate contract group, cautiously phrased."""
    top = highest_churn_category(df, "Contract", min_customers=30)
    if top is None:
        return "There are not enough customers per contract group to rank churn rates reliably."
    return (
        f"Customers on **{top['category']}** contracts have the highest observed churn rate "
        f"({top['churn_rate']:.1%} of {top['customers']:,} customers in this group). "
        "This is an association in this sample, not proof that contract type causes churn."
    )


def tenure_insight(df) -> str:
    """Churn variation across tenure bands."""
    table = tenure_band_churn(df)
    if table.empty or table["churn_rate"].isna().all():
        return "Tenure-band churn rates could not be computed for this data."
    highest = table.loc[table["churn_rate"].idxmax()]
    lowest = table.loc[table["churn_rate"].idxmin()]
    return (
        f"Churn varies strongly by tenure: customers with **{highest['band']}** tenure show an "
        f"observed churn rate of {highest['churn_rate']:.1%}, while **{lowest['band']}** customers "
        f"show {lowest['churn_rate']:.1%}. Newer customers are associated with higher observed "
        "churn in this dataset."
    )


def payment_or_internet_insight(df) -> str:
    """Churn variation for internet service and payment method."""
    parts = []
    for column, label in [("InternetService", "internet service"), ("PaymentMethod", "payment method")]:
        top = highest_churn_category(df, column, min_customers=30)
        if top is not None:
            parts.append(
                f"**{top['category']}** {label} customers have the highest observed churn rate "
                f"({top['churn_rate']:.1%})"
            )
    if not parts:
        return "No reliable internet-service or payment-method churn differences were found."
    return " and ".join(parts) + ". These are observed associations in this sample."


def top_coefficient_insights(model_result: ModelResult, top_n: int = 4) -> list[dict]:
    """Largest positive and negative Logistic Regression coefficients, explained."""
    if not model_result.is_valid:
        return []
    try:
        ranked = logistic_regression_coefficients(model_result)
    except ValueError:
        return []
    positive = [c for c in ranked if c["coefficient"] > 0][:top_n]
    negative = [c for c in reversed(ranked) if c["coefficient"] < 0][:top_n]
    return [
        {
            "direction": "higher predicted churn risk",
            "items": positive,
            "explanation": "A one-unit increase in this encoded feature is associated with higher log-odds of churn, holding other features constant.",
        },
        {
            "direction": "lower predicted churn risk",
            "items": negative,
            "explanation": "This encoded feature is associated with lower log-odds of churn under the model. Categorical effects are relative to the one-hot encoding scheme.",
        },
    ]


def model_comparison_insight(
    logreg_eval, lda_eval, metric: str = "f1_score"
) -> str:
    """Which model scores higher on the selected metric, given evaluation results.

    ``logreg_eval`` and ``lda_eval`` are EvaluationResult objects computed on
    the same split and threshold.
    """
    metric_labels = {
        "f1_score": "F1-score",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
        "roc_auc": "ROC-AUC",
    }
    label = metric_labels.get(metric, metric)
    if logreg_eval is None or lda_eval is None:
        return "Both models must be trained before they can be compared."
    a = getattr(logreg_eval, metric, None)
    b = getattr(lda_eval, metric, None)
    if a is None or b is None:
        return "Model comparison requires computed evaluation results."

    if a > b:
        winner, wa, wb = "Logistic Regression", a, b
    elif b > a:
        winner, wa, wb = "LDA", b, a
    else:
        return f"Both models score equally on validation {label} ({a:.3f})."
    gap = abs(wa - wb)
    qualifier = "slightly" if gap < 0.02 else "clearly"
    return (
        f"On validation data, **{winner}** scores {qualifier} higher on {label} "
        f"({wa:.3f} vs {wb:.3f}). The difference is measured on the same split and threshold; "
        "it does not guarantee superiority on other samples."
    )


def threshold_tradeoff_note(threshold: float) -> str:
    """Explain the FP/FN trade-off at the current threshold."""
    if threshold < 0.45:
        return (
            f"At threshold {threshold:.2f}, the model flags more customers as churn-risk. "
            "This typically raises recall (fewer true churners missed) but lowers precision "
            "(more false alarms), increasing outreach cost."
        )
    if threshold > 0.55:
        return (
            f"At threshold {threshold:.2f}, the model flags fewer customers. "
            "This typically raises precision (fewer false alarms) but lowers recall — "
            "more true churners go undetected and receive no retention offer."
        )
    return (
        f"At the balanced default threshold {threshold:.2f}, neither false positives nor "
        "false negatives is strongly favored. The best operating point depends on the relative "
        "cost of a missed churner versus an unnecessary retention offer."
    )


LIMITATIONS = [
    "The dataset is a public educational sample for a fictional telecom company — not live customer data.",
    "It is a static snapshot; the models describe patterns in this sample, they do not forecast a future month.",
    "Predictions are risk estimates, not certainties about any individual customer.",
    "Observed associations (e.g., contract type and churn) do not prove causation; changing a feature may not change behavior.",
    "Demographic attributes such as gender or senior-citizen status raise fairness concerns and should not drive unfair treatment.",
    "No intervention outcomes exist in the data, so retention savings or campaign effectiveness cannot be claimed.",
]
