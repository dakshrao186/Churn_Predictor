"""Reusable Plotly chart builders.

Charts receive already-computed data (summaries, metrics, probabilities) and
never load files or fit models. Consistent white backgrounds, subtle grid,
and the shared churn color encoding (Yes = red, No = blue) with text labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import config


def _apply_layout(fig: go.Figure, title: str, x_title: str, y_title: str, legend: bool = True):
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 15, "color": config.COLOR_TEXT}},
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
        showlegend=legend,
        legend=dict(orientation="h", y=-0.15) if legend else None,
    )
    fig.update_xaxes(title_text=x_title, gridcolor="#EEF2F7")
    fig.update_yaxes(title_text=y_title, gridcolor="#EEF2F7")
    return fig


def churn_distribution_chart(churn_counts: pd.Series) -> go.Figure:
    """Bar chart of Churn = Yes vs No with counts and percentages."""
    total = int(churn_counts.sum())
    labels = ["Retained (No)", "Churned (Yes)"]
    keys = ["No", "Yes"]
    values = [int(churn_counts.get(k, 0)) for k in keys]
    colors = [config.COLOR_CHURN_NO, config.COLOR_CHURN_YES]
    text = [f"{v:,} ({v / total:.1%})" if total else "0" for v in values]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="%{x}: %{y:,} customers<extra></extra>",
        )
    )
    fig.update_layout(yaxis_range=[0, max(values) * 1.18 if values else 1])
    # Single trace; the x labels already name both classes.
    return _apply_layout(fig, "Observed churn distribution", "Churn status", "Customers", legend=False)


def churn_by_category_chart(category_table: pd.DataFrame, column_label: str) -> go.Figure:
    """Churn rate by category (within-group denominator), sorted descending."""
    fig = px.bar(
        category_table,
        x="category",
        y="churn_rate",
        color="churn_rate",
        color_continuous_scale=["#93C5FD", config.COLOR_ACCENT_BLUE, config.COLOR_CHURN_YES],
        range_color=[0, max(0.6, float(category_table["churn_rate"].max() or 0))],
        text=category_table["churn_rate"].map(lambda v: f"{v:.1%}"),
        hover_data={"customers": True, "churned": True, "churn_rate": ":.1%"},
        labels={"category": column_label, "churn_rate": "Churn rate", "customers": "Customers"},
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(yaxis_tickformat=".0%")
    fig.update_layout(yaxis_range=[0, 1.05])
    return _apply_layout(fig, f"Observed churn rate by {column_label}", column_label, "Churn rate")


def numeric_histogram_chart(df: pd.DataFrame, column: str, label: str, bins: int = 30) -> go.Figure:
    """Histogram of a numeric column, colored by churn status."""
    fig = px.histogram(
        df,
        x=column,
        color=config.TARGET_COLUMN,
        barmode="overlay",
        nbins=bins,
        opacity=0.62,
        color_discrete_map=config.CHURN_COLOR_MAP,
        labels={column: label, "count": "Customers", config.TARGET_COLUMN: "Churn"},
    )
    fig.for_each_trace(lambda t: t.update(name="Churned (Yes)" if t.name == "Yes" else "Retained (No)"))
    return _apply_layout(fig, f"{label} distribution by churn", label, "Customers")


def numeric_box_chart(df: pd.DataFrame, column: str, label: str) -> go.Figure:
    """Box plot of a numeric column grouped by churn class."""
    fig = px.box(
        df,
        x=config.TARGET_COLUMN,
        y=column,
        color=config.TARGET_COLUMN,
        color_discrete_map=config.CHURN_COLOR_MAP,
        labels={column: label, config.TARGET_COLUMN: "Churn"},
        category_orders={config.TARGET_COLUMN: ["No", "Yes"]},
    )
    fig.for_each_trace(lambda t: t.update(name="Churned (Yes)" if t.name == "Yes" else "Retained (No)"))
    return _apply_layout(fig, f"{label} by churn class", "Churn", label)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Numeric correlation heatmap (descriptive association, not causation)."""
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"color": "Correlation"},
    )
    return _apply_layout(fig, "Numeric correlation (association, not causation)", "", "")


def confusion_matrix_chart(evaluation_result, title_suffix: str = "") -> go.Figure:
    """Confusion matrix heatmap, Actual vs Predicted, order [No(0), Yes(1)]."""
    cm = np.array(evaluation_result.confusion_matrix)
    class_labels = ["No churn (0)", "Churn (1)"]
    annotations = [
        dict(
            x=class_labels[j % 2],
            y=class_labels[i % 2],
            text=str(cm[i][j]),
            showarrow=False,
            font=dict(color="white" if cm[i][j] > cm.max() / 2 else config.COLOR_TEXT, size=18),
        )
        for i in range(2)
        for j in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(annotations=annotations)
    return _apply_layout(
        fig,
        f"{evaluation_result.model_name} — confusion matrix ({title_suffix})".strip(),
        "Predicted",
        "Actual",
    )


def roc_curve_chart(roc_series: list[dict], split_label: str) -> go.Figure:
    """ROC curves for one or both models on the same labelled split.

    ``roc_series`` items: {model_name, fpr, tpr, auc}.
    """
    fig = go.Figure()
    palette = [config.COLOR_ACCENT_BLUE, config.COLOR_ACCENT_GREEN]
    for i, series in enumerate(roc_series):
        auc_text = f"{series['auc']:.3f}" if series.get("auc") is not None else "n/a"
        fig.add_trace(
            go.Scatter(
                x=series["fpr"],
                y=series["tpr"],
                mode="lines",
                name=f"{series['model_name']} (AUC = {auc_text})",
                line=dict(color=palette[i % len(palette)], width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance (AUC = 0.500)",
            line=dict(color="#9CA3AF", dash="dot"),
            hoverinfo="skip",
        )
    )
    return _apply_layout(
        fig,
        f"ROC curves — {split_label}",
        "False positive rate",
        "True positive rate",
    )


def metric_comparison_chart(rows: list[dict]) -> go.Figure:
    """Grouped bar chart comparing both models across the main metrics."""
    df = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    fig = go.Figure()
    palette = [config.COLOR_ACCENT_BLUE, config.COLOR_ACCENT_GREEN]
    for i, model_name in enumerate(df["Model"].unique()):
        model_row = df[df["Model"] == model_name].iloc[0]
        values = [float(model_row[m]) if pd.notna(model_row[m]) else 0.0 for m in metric_cols]
        text = [f"{v:.3f}" for v in values]
        fig.add_trace(
            go.Bar(
                x=metric_cols,
                y=values,
                name=model_name,
                marker_color=palette[i % len(palette)],
                text=text,
                textposition="outside",
                textfont_size=11,
            )
        )
    fig.update_layout(barmode="group", yaxis_range=[0, 1.12])
    return _apply_layout(fig, "Model comparison (same split and threshold)", "Metric", "Score")


def coefficient_chart(coefs: list[dict], top_n: int = 12) -> go.Figure:
    """Ranked Logistic Regression coefficients (largest positive/negative)."""
    top = coefs[:top_n]
    names = [c["feature"] for c in top][::-1]
    values = [c["coefficient"] for c in top][::-1]
    colors = [
        config.COLOR_CHURN_YES if v > 0 else config.COLOR_ACCENT_GREEN for v in values
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Coefficient: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_range=[min(values) * 1.35, max(values) * 1.35] if values else [-1, 1])
    return _apply_layout(
        fig,
        "Logistic Regression coefficients (log-odds of churn)",
        "Coefficient (log-odds)",
        "",
        legend=False,
    )
