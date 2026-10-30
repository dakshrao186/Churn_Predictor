"""Tests for cleaning: blanks, TotalCharges conversion, zero-tenure fix, duplicates, raw immutability."""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_data, count_whitespace_blanks
from src import config


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.ID_COLUMN: ["a", "b", "c", "d", "e"],
            "tenure": [0, 12, 0, 5, 5],
            "MonthlyCharges": [20.0, 80.0, 30.0, 50.0, 50.0],
            "TotalCharges": [" ", "960.0", "30.5", "250.0", "250.0"],
            "Contract": ["Month-to-month", "Two year", "One year", " ", "One year"],
            "PaymentMethod": ["Mailed check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)", "Credit card (automatic)"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "Fiber optic"],
            config.TARGET_COLUMN: ["No", "Yes", "No", "No", "No"],
        }
    )


def test_raw_dataframe_unchanged(raw_df):
    snapshot = raw_df.copy(deep=True)
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_whitespace_blanks_counted(raw_df):
    # TotalCharges has one blank; Contract has one blank.
    assert count_whitespace_blanks(raw_df) == 2


def test_total_charges_converted_to_numeric(raw_df):
    cleaned = clean_data(raw_df).cleaned_df
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_zero_tenure_blank_correction(raw_df):
    result = clean_data(raw_df)
    cleaned = result.cleaned_df
    row_a = cleaned.loc[cleaned[config.ID_COLUMN] == "a"].iloc[0]
    assert row_a["TotalCharges"] == 0.0
    # Only row "a" is a zero-tenure blank; row "c" has tenure 0 but valid charges.
    assert result.quality_report.zero_tenure_charge_correction_count == 1


def test_whitespace_becomes_missing_then_reported(raw_df):
    result = clean_data(raw_df)
    cleaned = result.cleaned_df
    row_d = cleaned.loc[cleaned[config.ID_COLUMN] == "d"].iloc[0]
    assert pd.isna(row_d["Contract"])  # whitespace-only category became NaN
    assert result.quality_report.whitespace_blank_cells == 2
    assert "Contract" in result.quality_report.missing_after_by_column


def test_duplicate_counts_reported(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[1]]], ignore_index=True)
    report = clean_data(duplicated).quality_report
    assert report.duplicate_row_count == 1
    # Policy: duplicates are reported, not silently deleted.
    assert report.cleaned_shape[0] == 6


def test_invalid_target_labels_reported():
    df = pd.DataFrame(
        {
            config.ID_COLUMN: ["a", "b"],
            "tenure": [1, 2],
            "MonthlyCharges": [20.0, 20.0],
            "TotalCharges": [20.0, 40.0],
            "Contract": ["One year", "One year"],
            "PaymentMethod": ["Mailed check", "Mailed check"],
            "InternetService": ["DSL", "DSL"],
            config.TARGET_COLUMN: ["No", "MAYBE"],
        }
    )
    result = clean_data(df)
    assert result.quality_report.invalid_target_labels == ["MAYBE"]
    assert config.TARGET_COLUMN not in result.quality_report.missing_after_by_column


def test_missing_before_and_after_totals(raw_df):
    report = clean_data(raw_df).quality_report
    # Before: 1 blank TotalCharges + 1 blank Contract; after cleaning:
    # TotalCharges zero-tenure rows were corrected, Contract blank remains.
    assert report.missing_before_total == 2
    assert report.missing_after_total == 1
