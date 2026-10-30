"""Shared fixtures: the real cleaned Telco dataset for pipeline tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import clean_data  # noqa: E402
from src.data_loader import load_raw_dataframe  # noqa: E402


@pytest.fixture(scope="session")
def real_cleaned_df() -> pd.DataFrame:
    raw = load_raw_dataframe()
    return clean_data(raw).cleaned_df
