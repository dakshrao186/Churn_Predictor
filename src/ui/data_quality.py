"""Data Quality & Cleaning page: transparent raw-to-cleaned report."""

from __future__ import annotations

import streamlit as st

from src import charts, config
from src.ui import layout


def render(raw_df, cleaned_df, quality_report) -> None:
    layout.page_header(
        "Data Quality & Cleaning",
        "What was checked, what was corrected, and what remains for the model pipeline to handle. "
        "The raw file is never modified.",
    )

    qr = quality_report
    layout.metric_row(
        [
            {"label": "Raw shape", "value": f"{qr.raw_shape[0]:,} × {qr.raw_shape[1]}", "hint": "Rows × columns as loaded"},
            {"label": "Cleaned shape", "value": f"{qr.cleaned_shape[0]:,} × {qr.cleaned_shape[1]}", "hint": "Used for EDA and modeling"},
            {"label": "Blank TotalCharges fixed", "value": f"{qr.zero_tenure_charge_correction_count}", "hint": "Zero-tenure rows set to 0", "dark": True},
            {"label": "Duplicates (rows / IDs)", "value": f"{qr.duplicate_row_count} / {qr.duplicate_customer_id_count}", "hint": "Reported, retained"},
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Missing values — before cleaning")
        if qr.missing_before_by_column:
            import pandas as pd

            before_df = pd.DataFrame(
                {"Column": list(qr.missing_before_by_column.keys()),
                 "Missing cells": list(qr.missing_before_by_column.values())}
            )
            st.dataframe(before_df, width="stretch", hide_index=True)
        else:
            st.caption("No missing values detected before cleaning.")
        st.caption(
            f"Total missing/blank cells before cleaning: **{qr.missing_before_total}** "
            f"(including {qr.whitespace_blank_cells} whitespace-only cell(s))."
        )
    with right:
        st.markdown("#### Missing values — after cleaning")
        if qr.missing_after_by_column:
            import pandas as pd

            after_df = pd.DataFrame(
                {"Column": list(qr.missing_after_by_column.keys()),
                 "Missing cells": list(qr.missing_after_by_column.values())}
            )
            st.dataframe(after_df, width="stretch", hide_index=True)
        else:
            st.success("No missing values remain after cleaning for this dataset.")
        st.caption(
            "Any missing predictor values at model time would be imputed inside the "
            "train-fitted pipeline (median for numeric, most-frequent for categorical) — "
            "never using validation or test statistics."
        )

    st.markdown("#### Cleaning actions")
    st.dataframe(qr.to_actions_table(), width="stretch", hide_index=True)

    if qr.invalid_target_labels:
        st.warning(
            "Unexpected target labels found and removed: "
            + ", ".join(qr.invalid_target_labels)
        )

    st.markdown("#### Outlier inspection")
    st.caption(
        "Box plots flag values outside the IQR rule, but extreme values can be legitimate "
        "customers (e.g., very high TotalCharges after many years). Records are inspected, "
        "not deleted. If IQR capping is enabled in the Model Lab, it is fitted on training "
        "data only and applied unchanged to validation/test."
    )
    box1, box2, box3 = st.columns(3, gap="small")
    with box1:
        st.plotly_chart(charts.numeric_box_chart(cleaned_df, "tenure", "Tenure (months)"), width="stretch")
    with box2:
        st.plotly_chart(charts.numeric_box_chart(cleaned_df, "MonthlyCharges", "Monthly charges (USD)"), width="stretch")
    with box3:
        st.plotly_chart(charts.numeric_box_chart(cleaned_df, "TotalCharges", "Total charges (USD)"), width="stretch")

    tab1, tab2 = st.tabs(["Raw preview", "Cleaned preview"])
    with tab1:
        st.dataframe(raw_df.head(25), width="stretch")
        st.caption(f"First 25 of {len(raw_df):,} raw rows (unchanged snapshot).")
    with tab2:
        st.dataframe(cleaned_df.head(25), width="stretch")
        st.caption(f"First 25 of {len(cleaned_df):,} cleaned rows.")
