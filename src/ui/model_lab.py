"""Model Lab: model selection, interactive threshold, comparison, and fixed test evaluation.

Threshold changes reuse cached probability arrays and recompute only
threshold-dependent metrics — no retraining. Model/setting changes rebuild
via caches keyed on the true training configuration (dataset fingerprint,
seed, outlier policy, C). Test evaluation always uses the fixed default
threshold 0.50 and never responds to the validation slider.
"""

from __future__ import annotations

import streamlit as st

from src import app_state, charts, config
from src.evaluation import evaluate, roc_curve_points
from src.modeling import ModelResult, logistic_regression_coefficients
from src.insights import threshold_tradeoff_note
from src.preprocessing import DataSplit
from src.ui import layout

VALIDATION_LABEL = "Validation / Interactive"
TEST_LABEL = "Test / Final Evaluation"


# ---------------------------------------------------------------------------
# Cached heavy computation — deliberately NOT keyed on the threshold
# ---------------------------------------------------------------------------
def _dataset_fingerprint() -> int:
    return app_state.dataset_fingerprint(st.session_state["cleaned_df"])


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _metric_cards(ev) -> list[dict]:
    return [
        {"label": "Accuracy", "value": f"{ev.accuracy:.3f}", "hint": "Share of all predictions correct"},
        {"label": "Precision", "value": f"{ev.precision:.3f}", "hint": "Flagged customers who churned"},
        {"label": "Recall", "value": f"{ev.recall:.3f}", "hint": "Churners identified", "dark": True},
        {"label": "F1-score", "value": f"{ev.f1_score:.3f}", "hint": "Balance of precision & recall"},
        {"label": "ROC-AUC", "value": f"{ev.roc_auc:.3f}" if ev.roc_auc is not None else "n/a",
         "hint": "From probabilities — threshold-independent"},
    ]


def _single_model_view(result: ModelResult, split, threshold: float) -> None:
    if not result.is_valid:
        st.error(result.error_message or "Model training failed.")
        return
    y_val = split.y_validation
    ev = evaluate(y_val, result.validation_probabilities, result.model_name, VALIDATION_LABEL, threshold)
    layout.metric_row(_metric_cards(ev))
    layout.status_chip_row([
        (f"Validation · n = {ev.sample_count:,}", "blue"),
        (f"Threshold = {threshold:.2f}", "blue"),
        (f"Flagged as churn-risk: {ev.predicted_churn_count:,} ({ev.predicted_churn_rate:.1%})", "green"),
    ])
    cm_col, roc_col = st.columns(2, gap="large")
    with cm_col:
        st.plotly_chart(charts.confusion_matrix_chart(ev, VALIDATION_LABEL), width="stretch")
        st.caption(
            "Rows are actual classes, columns are predicted. Top-left: correctly retained; "
            "bottom-right: correctly flagged churners."
        )
    with roc_col:
        fpr, tpr = roc_curve_points(y_val, result.validation_probabilities)
        st.plotly_chart(
            charts.roc_curve_chart(
                [{"model_name": result.model_name, "fpr": fpr, "tpr": tpr, "auc": ev.roc_auc}],
                VALIDATION_LABEL,
            ),
            width="stretch",
        )
        st.caption("ROC-AUC is computed from probabilities, so it does not move when the threshold slider moves.")
    with st.expander("Classification report"):
        from sklearn.metrics import classification_report

        preds = (result.validation_probabilities >= threshold).astype(int)
        st.text(classification_report(y_val, preds, target_names=["No churn", "Churn"], zero_division=0))


def _compare_view(logreg: ModelResult, lda: ModelResult, split, threshold: float) -> None:
    if not (logreg.is_valid and lda.is_valid):
        for r in (logreg, lda):
            if not r.is_valid:
                st.error(r.error_message or f"{r.model_name} training failed.")
        return
    y_val = split.y_validation
    ev_lr = evaluate(y_val, logreg.validation_probabilities, "Logistic Regression", VALIDATION_LABEL, threshold)
    ev_lda = evaluate(y_val, lda.validation_probabilities, "LDA", VALIDATION_LABEL, threshold)

    layout.metric_row([
        {"label": "LogReg F1", "value": f"{ev_lr.f1_score:.3f}", "hint": f"LDA: {ev_lda.f1_score:.3f}"},
        {"label": "LDA F1", "value": f"{ev_lda.f1_score:.3f}", "hint": f"LogReg: {ev_lr.f1_score:.3f}", "dark": True},
        {"label": "LogReg ROC-AUC", "value": f"{ev_lr.roc_auc:.3f}" if ev_lr.roc_auc is not None else "n/a", "hint": "Probability-based"},
        {"label": "LDA ROC-AUC", "value": f"{ev_lda.roc_auc:.3f}" if ev_lda.roc_auc is not None else "n/a", "hint": "Probability-based"},
        {"label": "Threshold", "value": f"{threshold:.2f}", "hint": "Applied to both models"},
    ])

    st.plotly_chart(
        charts.metric_comparison_chart([ev_lr.metrics_row(), ev_lda.metrics_row()]),
        width="stretch",
    )

    cm1, cm2 = st.columns(2, gap="large")
    with cm1:
        st.plotly_chart(charts.confusion_matrix_chart(ev_lr, f"Validation · t={threshold:.2f}"), width="stretch")
    with cm2:
        st.plotly_chart(charts.confusion_matrix_chart(ev_lda, f"Validation · t={threshold:.2f}"), width="stretch")

    roc_series = []
    for res, probs, auc in [
        (logreg, logreg.validation_probabilities, ev_lr.roc_auc),
        (lda, lda.validation_probabilities, ev_lda.roc_auc),
    ]:
        fpr, tpr = roc_curve_points(y_val, probs)
        roc_series.append({"model_name": res.model_name, "fpr": fpr, "tpr": tpr, "auc": auc})
    st.plotly_chart(charts.roc_curve_chart(roc_series, VALIDATION_LABEL), width="stretch")

    st.dataframe(
        [r | {"Split": VALIDATION_LABEL, "Threshold": f"{threshold:.2f}"} for r in
         [ev_lr.metrics_row(), ev_lda.metrics_row()]],
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def render(cleaned_df) -> None:
    layout.page_header(
        "Model Lab",
        "Train and compare Logistic Regression and LDA, and explore how the classification "
        "threshold changes validation results. The final test evaluation stays fixed at 0.50.",
    )
    st.session_state["cleaned_df"] = cleaned_df
    fingerprint = _dataset_fingerprint()

    # ----- Controls ---------------------------------------------------------
    c1, c2, c3 = st.columns([1.1, 1.7, 1.2], gap="small")
    with c1:
        model_choice = st.radio(
            "Model",
            ["Logistic Regression", "LDA", "Compare both"],
            help="Which model's validation results to show, or a side-by-side comparison on the same split.",
        )
    with c2:
        threshold = st.slider(
            "Classification threshold (validation set)",
            min_value=config.THRESHOLD_MIN,
            max_value=config.THRESHOLD_MAX,
            value=config.DEFAULT_THRESHOLD,
            step=config.THRESHOLD_STEP,
            format="%.2f",
            help=(
                "A customer is flagged as churn-risk when predicted churn probability ≥ threshold. "
                "Lowering it flags more customers (usually higher recall, lower precision); "
                "raising it flags fewer."
            ),
        )
    with c3:
        C_value = st.select_slider(
            "LogReg regularization (C)",
            options=config.LOGREG_C_OPTIONS,
            value=config.LOGREG_DEFAULT_C,
            help="Changing C retrains Logistic Regression on training data only and re-evaluates on validation data. Test metrics are never used for this choice.",
        )

    cap_outliers = st.checkbox(
        "Enable IQR capping of numeric predictors (fitted on training data only)",
        value=False,
        help="Clips numeric predictors to IQR bounds learned from the training partition, applied unchanged to validation/test. Changing this refits both models.",
    )

    layout.status_chip_row([
        ("Validation / Interactive", "blue"),
        ("Test / Final Evaluation — fixed threshold 0.50", "green"),
        (f"Random seed = {config.DEFAULT_RANDOM_SEED}", "neutral"),
        ("Threshold changes never retrain the models", "neutral"),
    ])

    # ----- Models (cached by true configuration) -----------------------------
    bundle = app_state.cached_models(
        fingerprint, st.session_state["cleaned_df"], cap_outliers, C_value
    )
    split = bundle["split"]
    logreg: ModelResult = bundle["logreg"]
    lda: ModelResult = bundle["lda"]

    st.markdown("### Validation / Interactive results")
    if model_choice == "Logistic Regression":
        _single_model_view(logreg, split, threshold)
    elif model_choice == "LDA":
        _single_model_view(lda, split, threshold)
    else:
        _compare_view(logreg, lda, split, threshold)

    st.markdown(
        f"<div class='cc-panel'><div class='cc-panel-title'>Threshold trade-off</div>"
        f"<p class='cc-note'>{threshold_tradeoff_note(threshold)}</p></div>",
        unsafe_allow_html=True,
    )

    # ----- Fixed test evaluation -------------------------------------------
    st.markdown("### Test / Final Evaluation (held out, threshold = 0.50)")
    st.caption(
        "The test set was never used for training, threshold tuning, or model choices. "
        "These fixed-threshold numbers are the final, unbiased performance estimate and "
        "do not respond to the validation slider."
    )
    test_cols = st.columns(2, gap="large")
    for col, result in zip(test_cols, [logreg, lda]):
        with col:
            if not result.is_valid:
                st.error(result.error_message or "Model unavailable.")
                continue
            ev_test = evaluate(
                split.y_test, result.test_probabilities,
                result.model_name, TEST_LABEL, config.DEFAULT_THRESHOLD,
            )
            layout.metric_row([
                {"label": f"{result.model_name} — Accuracy", "value": f"{ev_test.accuracy:.3f}"},
                {"label": f"{result.model_name} — F1", "value": f"{ev_test.f1_score:.3f}", "dark": True},
                {"label": f"{result.model_name} — ROC-AUC",
                 "value": f"{ev_test.roc_auc:.3f}" if ev_test.roc_auc is not None else "n/a"},
            ])
            st.plotly_chart(
                charts.confusion_matrix_chart(ev_test, TEST_LABEL),
                width="stretch",
            )
            st.caption(f"Test sample: n = {ev_test.sample_count:,} · fixed threshold 0.50")

    # ----- Logistic Regression coefficients ---------------------------------
    if logreg.is_valid:
        with st.expander("Logistic Regression coefficients (interpretation)"):
            coefs = logistic_regression_coefficients(logreg)
            st.plotly_chart(charts.coefficient_chart(coefs), width="stretch")
            st.caption(
                "Coefficients are changes in log-odds of churn per unit of the encoded feature, "
                "holding other features constant. Positive values are associated with higher churn "
                "risk; negative with lower. One-hot categories are relative to the encoding scheme. "
                "Association is not causation."
            )
