"""Tests for EDA aggregations with a hand-calculated fixture."""

import pandas as pd
import pytest

from src.eda import churn_by_category, churn_overview, tenure_band_churn


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    # Contract=MTM: 2 of 4 churn -> 50%; Two year: 0 of 2 -> 0%.
    return pd.DataFrame(
        {
            "customerID": [f"c{i}" for i in range(6)],
            "Contract": ["Month-to-month"] * 4 + ["Two year"] * 2,
            "tenure": [1, 3, 10, 40, 60, 65],
            "MonthlyCharges": [50, 60, 70, 80, 90, 95],
            "TotalCharges": [50, 120, 700, 3200, 5400, 6100],
            "Churn": ["Yes", "No", "Yes", "No", "No", "No"],
        }
    )


def test_overview_counts(tiny_df):
    overview = churn_overview(tiny_df)
    assert overview["total_customers"] == 6
    assert overview["churned"] == 2
    assert overview["retained"] == 4
    assert overview["churn_rate"] == pytest.approx(2 / 6)


def test_within_group_denominator(tiny_df):
    table = churn_by_category(tiny_df, "Contract")
    mtm = table[table["category"] == "Month-to-month"].iloc[0]
    two_year = table[table["category"] == "Two year"].iloc[0]
    # Rate = churned in group / ALL customers in group (not share of all churners).
    assert mtm["customers"] == 4
    assert mtm["churned"] == 2
    assert mtm["churn_rate"] == pytest.approx(0.5)
    assert two_year["churn_rate"] == pytest.approx(0.0)
    # Sorted by churn rate, descending.
    assert table.iloc[0]["category"] == "Month-to-month"


def test_missing_category_grouped_not_dropped(tiny_df):
    df = tiny_df.copy()
    df.loc[df.index[0], "Contract"] = None
    table = churn_by_category(df, "Contract")
    assert "(Missing)" in set(table["category"])
    assert table["customers"].sum() == 6


def test_tenure_band_churn(tiny_df):
    table = tenure_band_churn(tiny_df)
    assert table["customers"].sum() == 6
    # The 0-6 month band holds tenure {1, 3}: one churner of two customers.
    young = table[table["band"] == "0-6 mo"].iloc[0]
    assert young["churn_rate"] == pytest.approx(0.5)
