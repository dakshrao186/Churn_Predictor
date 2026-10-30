"""Data Explorer page: interactive EDA controls and charts over the cleaned dataset."""

from __future__ import annotations

import streamlit as st

from src import charts, config, eda
from src.ui import layout

CATEGORY_OPTIONS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PaperlessBilling",
]

NUMERIC_OPTIONS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges (USD)",
    "TotalCharges": "Total charges (USD)",
}


def render(cleaned_df) -> None:
    layout.page_header(
        "Data Explorer",
        "Explore how customer characteristics relate to observed churn in this sample. "
        "All charts describe the full cleaned dataset (descriptive analysis, not model performance).",
    )

    # ---- Controls above the charts they affect -----------------------------
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        category = st.selectbox(
            "Churn-rate breakdown by category",
            CATEGORY_OPTIONS,
            index=0,
            help="Churn rate is computed within each category group: churned customers in the group ÷ all customers in the group.",
        )
    with col_b:
        numeric = st.selectbox(
            "Distribution chart",
            list(NUMERIC_OPTIONS.keys()),
            format_func=lambda c: NUMERIC_OPTIONS[c],
            help="Choose which numeric feature to display as a histogram and box plot.",
        )

    category_table = eda.churn_by_category(cleaned_df, category)

    top1, top2 = st.columns(2, gap="large")
    with top1:
        st.plotly_chart(
            charts.churn_distribution_chart(cleaned_df[config.TARGET_COLUMN].value_counts()),
            width="stretch",
        )
    with top2:
        st.plotly_chart(
            charts.churn_by_category_chart(category_table, category),
            width="stretch",
        )
        top_row = category_table.iloc[0]
        st.markdown(
            f"<p class='cc-note'>Highest observed churn rate: <b>{top_row['category']}</b> — "
            f"{top_row['churn_rate']:.1%} of {top_row['customers']:,} customers in that group "
            "churned in this sample (association, not causation).</p>",
            unsafe_allow_html=True,
        )

    mid1, mid2 = st.columns(2, gap="large")
    with mid1:
        st.plotly_chart(
            charts.numeric_histogram_chart(cleaned_df, numeric, NUMERIC_OPTIONS[numeric]),
            width="stretch",
        )
    with mid2:
        st.plotly_chart(
            charts.numeric_box_chart(cleaned_df, numeric, NUMERIC_OPTIONS[numeric]),
            width="stretch",
        )

    st.markdown("#### Numeric summary by churn class")
    stats = eda.numeric_stats_by_churn(cleaned_df, config.NUMERIC_COLUMNS)
    st.dataframe(stats, width="stretch")

    with st.expander("Correlation heatmap (numeric columns only)"):
        st.plotly_chart(
            charts.correlation_heatmap(eda.correlation_matrix(cleaned_df)),
            width="stretch",
        )
        st.caption(
            "Correlation measures linear association between numeric columns. "
            "It is not a measure of causation."
        )

    with st.expander("Data preview (cleaned dataset)"):
        st.dataframe(cleaned_df.head(50), width="stretch")
        st.caption(f"Showing the first 50 of {len(cleaned_df):,} rows.")
