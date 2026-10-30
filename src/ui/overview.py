"""Overview page: project context, dataset status, and headline observed churn stats."""

from __future__ import annotations

import streamlit as st

from src import charts, config, eda
from src.ui import layout


def render(cleaned_df, quality_report, source_label: str, model_rows: list[dict] | None) -> None:
    layout.page_header(
        "Overview",
        "What this app does, where the data comes from, and how many customers churned in the sample.",
    )

    overview = eda.churn_overview(cleaned_df)

    cards = [
        {"label": "Customer records", "value": f"{overview['total_customers']:,}", "hint": "Rows in the cleaned dataset"},
        {"label": "Observed churn rate", "value": f"{overview['churn_rate']:.1%}", "hint": "Historical, in this sample", "dark": True},
        {"label": "Churned customers", "value": f"{overview['churned']:,}", "hint": "Churn = Yes (positive class)"},
        {"label": "Retained customers", "value": f"{overview['retained']:,}", "hint": "Churn = No"},
    ]
    if model_rows:
        best_f1 = max(
            (r["F1-score"] for r in model_rows if r.get("F1-score") is not None),
            default=None,
        )
        if best_f1 is not None:
            cards.append({"label": "Best validation F1", "value": f"{best_f1:.3f}", "hint": "Model performance (validation split)"})
    layout.metric_row(cards)

    layout.status_chip_row(
        [
            ("Dataset loaded", "green"),
            (source_label, "neutral"),
            (f"{quality_report.cleaned_shape[0]:,} rows × {quality_report.cleaned_shape[1]} columns", "neutral"),
            ("Sample data — not live customer records", "blue"),
        ]
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Observed churn distribution")
        st.plotly_chart(
            charts.churn_distribution_chart(cleaned_df[config.TARGET_COLUMN].value_counts()),
            width="stretch",
        )
    with right:
        st.markdown("#### About churn")
        st.markdown(
            "**Churn** means a customer ended their subscription. In this dataset the "
            "`Churn` column records whether each customer left the fictional telecom "
            "provider.\n\n"
            "- `Churn = Yes` is the **positive class** the models learn to identify.\n"
            f"- About **{overview['churn_rate']:.1%}** of customers in this sample churned, "
            "so the classes are imbalanced — accuracy alone can be misleading and F1-score, "
            "recall, and ROC-AUC matter.\n\n"
            "All figures on this page describe the **historical sample**. Predictions and "
            "model metrics appear separately in the Model Lab."
        )
        st.info(
            "This is a public educational sample dataset (IBM Telco Customer Churn). "
            "Values shown are dataset records, not live customer data.",
            icon="ℹ️",
        )
