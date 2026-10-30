"""Tests for data loading and schema validation."""

import pandas as pd
import pytest

from src.data_loader import DataLoadError, _validate_schema, load_raw_dataframe
from src import config


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 2],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 20.0],
            "TotalCharges": ["29.85", " "],
            "Churn": ["No", "Yes"],
        }
    )


def test_validate_schema_ok():
    status, messages = _validate_schema(_make_valid_df())
    assert status == "valid"
    assert messages == []


def test_validate_schema_missing_required_columns():
    df = _make_valid_df().drop(columns=["Churn", "TotalCharges"])
    status, messages = _validate_schema(df)
    assert status == "invalid"
    assert any("Churn" in m and "TotalCharges" in m for m in messages)


def test_validate_schema_reports_extra_columns_but_valid():
    df = _make_valid_df()
    df["extra_note"] = "x"
    status, messages = _validate_schema(df)
    assert status == "valid"
    assert any("extra_note" in m for m in messages)


def test_load_raw_dataframe_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_CSV_PATH", tmp_path / "missing.csv")
    # Simulate an unavailable network so the download fallback also fails.
    import requests

    def refuse_get(*args, **kwargs):
        raise requests.ConnectionError("network disabled for test")

    monkeypatch.setattr("src.data_loader.requests.get", refuse_get)
    with pytest.raises(DataLoadError):
        load_raw_dataframe(str(tmp_path / "missing.csv"))


def test_load_raw_dataframe_reads_local_file(tmp_path, monkeypatch):
    path = tmp_path / "telco.csv"
    _make_valid_df().to_csv(path, index=False)
    df = load_raw_dataframe(str(path))
    assert len(df) == 2
