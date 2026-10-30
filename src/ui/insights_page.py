"""Insights & Limitations page: data-derived findings, coefficients, responsible use."""

from __future__ import annotations

import streamlit as st

from src import charts
from src.insights import (
    LIMITATIONS,
    contract_insight,
    model_comparison_insight,
    payment_or_internet_insight,
    tenure_insight,
    top_coefficient_insights,
)
from src.modeling import ModelResult, logistic_regression_coefficients
from src.ui import layout


def render(cleaned_df, logreg: ModelResult, lda: ModelResult, logreg_eval, lda_eval) -> None:
    layout.page_header(
        "Insights & Limitations",
        "What the data and models actually show in this sample — expressed cautiously, "
        "with association language rather than causal claims.",
    )

    st.markdown("#### Observed patterns in the data")
    st.markdown(
        f"<div class='cc-panel'><p class='cc-note'>📋 {contract_insight(cleaned_df)}</p></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"<div class='cc-panel'><p class='cc-note'>⏳ {tenure_insight(cleaned_df)}</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div class='cc-panel'><p class='cc-note'>💳 {payment_or_internet_insight(cleaned_df)}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Model comparison")
    st.markdown(
        f"<div class='cc-panel'><p class='cc-note'>⚖️ {model_comparison_insight(logreg_eval, lda_eval)}</p></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "LDA models class-conditional Gaussian distributions and finds a linear discriminant "
        "boundary under its assumptions; Logistic Regression directly models the log-odds of "
        "churn. Their coefficients are on different scales and are not compared numerically."
    )

    st.markdown("#### What drives the Logistic Regression model")
    if logreg.is_valid:
        try:
            coefs = logistic_regression_coefficients(logreg)
            st.plotly_chart(charts.coefficient_chart(coefs), width="stretch")
            groups = top_coefficient_insights(logreg)
            for group in groups:
                if group["items"]:
                    names = ", ".join(
                        f"**{c['feature']}** ({c['coefficient']:+.2f})" for c in group["items"][:4]
                    )
                    st.markdown(f"- Associated with **{group['direction']}**: {names}")
            st.caption(
                "A positive coefficient means higher log-odds of churn for that encoded feature, "
                "holding others constant; negative means lower. Categorical effects are relative "
                "to the one-hot encoding reference. These are associations in this sample — "
                "they do not prove that changing a feature would change churn."
            )
        except ValueError as exc:
            st.info(f"Coefficient interpretation unavailable: {exc}")
    else:
        st.info("Train the models in the Model Lab to see coefficient interpretation.")

    st.markdown("#### Limitations & responsible use")
    for limitation in LIMITATIONS:
        st.markdown(f"- {limitation}")
    st.info(
        "Predictions indicate estimated risk, not certainty about any individual customer. "
        "This is an educational decision-support demonstration, not an operational retention tool.",
        icon="⚠️",
    )
