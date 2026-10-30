"""Customer Churn Prediction & Retention Analytics — Streamlit entry point.

Small router: configures the page, loads the shared dataset once, and renders
the selected section. Analysis lives in src/; presentation in src/ui/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when launched from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import app_state
from src.data_loader import DataLoadError, load_dataset_cached
from src.data_cleaning import clean_data
from src.ui import layout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
layout.inject_css()

# ---------------------------------------------------------------------------
# Sidebar: identity + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Customer Churn Prediction")
    st.caption("Explore customer patterns and compare churn classifiers.")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Data Explorer", "Data Quality & Cleaning", "Model Lab", "Insights & Limitations"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Public IBM Telco sample dataset · educational mini-project for Predictive Analytics."
    )

# ---------------------------------------------------------------------------
# Shared data state (cached across reruns)
# ---------------------------------------------------------------------------
try:
    loaded = load_dataset_cached()
except DataLoadError as exc:
    st.error("## Dataset could not be loaded")
    st.markdown(str(exc))
    st.stop()

if loaded.schema_status != "valid":
    st.error("## Dataset schema is invalid")
    for message in loaded.schema_messages:
        st.markdown(f"- {message}")
    st.caption(
        f"Loaded from: {loaded.source_location}. Fix the CSV (21 required columns including "
        "customerID, tenure, MonthlyCharges, TotalCharges, and Churn) and reload the app."
    )
    st.stop()

cleaning = clean_data(loaded.raw_df)  # deterministic; raw snapshot stays untouched
cleaned_df = cleaning.cleaned_df
quality_report = cleaning.quality_report


def _models_and_evaluations():
    """Lazily fit both models on the shared split; returns bundle + evaluations."""
    return app_state.models_and_evaluations(cleaned_df)


# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------
if page == "Overview":
    from src.ui import overview

    _, evals = _models_and_evaluations()
    model_rows = [r.metrics_row() for r in (evals["logreg_val"], evals["lda_val"]) if r]
    overview.render(cleaned_df, quality_report, loaded.source_label, model_rows or None)

elif page == "Data Explorer":
    from src.ui import data_explorer

    data_explorer.render(cleaned_df)

elif page == "Data Quality & Cleaning":
    from src.ui import data_quality

    data_quality.render(loaded.raw_df, cleaned_df, quality_report)

elif page == "Model Lab":
    from src.ui import model_lab

    model_lab.render(cleaned_df)

else:  # Insights & Limitations
    from src.ui import insights_page

    bundle, evals = _models_and_evaluations()
    insights_page.render(
        cleaned_df,
        bundle["logreg"],
        bundle["lda"],
        evals["logreg_val"],
        evals["lda_val"],
    )
