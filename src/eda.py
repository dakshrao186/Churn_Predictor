"""Descriptive analytics on the cleaned dataset.

Owns: chart-ready summaries (counts, rates, distributions, correlations).
Does not fit models. Grouped churn rates always use the within-group
denominator: churned customers in a group / all customers in that group.
"""

from __future__ import annotations

import pandas as pd

from src import config

TENURE_BANDS = [-1, 6, 12, 24, 48, 72]
TENURE_BAND_LABELS = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]


def churn_overview(df: pd.DataFrame) -> dict:
    """Overall churn counts and rate from the cleaned dataset."""
    counts = df[config.TARGET_COLUMN].value_counts()
    total = int(len(df))
    churned = int(counts.get("Yes", 0))
    retained = int(counts.get("No", 0))
    rate = churned / total if total else 0.0
    return {
        "total_customers": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": rate,
    }


def churn_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn counts and within-group rate for one categorical column.

    Returns columns: category, customers, churned, churn_rate.
    Empty/missing category values are grouped under "(Missing)" so that
    charts never silently drop rows.
    """
    work = df[[column, config.TARGET_COLUMN]].copy()
    work[column] = work[column].astype(object)
    work[column] = work[column].where(work[column].notna(), "(Missing)")

    grouped = work.groupby(column, observed=True)[config.TARGET_COLUMN].agg(
        customers="size", churned=lambda s: int((s == "Yes").sum())
    )
    grouped = grouped.reset_index().rename(columns={column: "category"})
    grouped["churn_rate"] = grouped.apply(
        lambda r: r["churned"] / r["customers"] if r["customers"] else 0.0, axis=1
    )
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def numeric_distribution(df: pd.DataFrame, column: str, bins: int = 30):
    """Histogram-ready values of one numeric column, split by churn class."""
    return df[[column, config.TARGET_COLUMN]].dropna()


def numeric_stats_by_churn(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean/median of numeric columns grouped by the churn target."""
    stats = (
        df.groupby(config.TARGET_COLUMN)[columns]
        .agg(["mean", "median"])
        .round(2)
    )
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only correlation matrix (descriptive; not causation)."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr(numeric_only=True).round(3)


def tenure_band_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate by tenure band (within-band denominator)."""
    work = df[["tenure", config.TARGET_COLUMN]].dropna().copy()
    work["band"] = pd.cut(work["tenure"], bins=TENURE_BANDS, labels=TENURE_BAND_LABELS)
    grouped = (
        work.groupby("band", observed=True)[config.TARGET_COLUMN]
        .agg(customers="size", churned=lambda s: int((s == "Yes").sum()))
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    return grouped


def highest_churn_category(df: pd.DataFrame, column: str, min_customers: int = 30):
    """Category with the highest observed churn rate, or None if ambiguous."""
    table = churn_by_category(df, column)
    table = table[table["customers"] >= min_customers]
    if table.empty:
        return None
    top = table.iloc[0]
    return {"category": str(top["category"]), "churn_rate": float(top["churn_rate"]), "customers": int(top["customers"])}
