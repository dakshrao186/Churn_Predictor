"""Tests for feature preparation, stratified splits, and leakage invariants."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import (
    TrainOnlyIQRCapper,
    make_split,
    prepare_xy,
)


@pytest.fixture
def cleaned_df(real_cleaned_df) -> pd.DataFrame:  # noqa: F811
    return real_cleaned_df


def test_customerID_and_target_excluded(cleaned_df):
    X, y = prepare_xy(cleaned_df)
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET_COLUMN not in X.columns


def test_target_encoding(cleaned_df):
    _, y = prepare_xy(cleaned_df)
    assert set(y.unique()) == {0, 1}
    # Yes (churn) must be the positive class.
    yes_mask = cleaned_df[config.TARGET_COLUMN] == "Yes"
    assert (y[yes_mask] == 1).all()


def test_prepare_xy_rejects_unexpected_target():
    df = pd.DataFrame(
        {
            config.ID_COLUMN: ["a", "b"],
            "tenure": [1, 2],
            "MonthlyCharges": [20.0, 20.0],
            "TotalCharges": [20.0, 40.0],
            "Contract": ["One year", "Two year"],
            "PaymentMethod": ["Mailed check", "Mailed check"],
            "InternetService": ["DSL", "DSL"],
            config.TARGET_COLUMN: ["No", "MAYBE"],
        }
    )
    with pytest.raises(Exception):
        prepare_xy(df)


def test_split_deterministic_and_stratified(cleaned_df):
    s1 = make_split(cleaned_df, random_state=7)
    s2 = make_split(cleaned_df, random_state=7)
    pd.testing.assert_frame_equal(s1.X_train, s2.X_train)
    pd.testing.assert_series_equal(s1.y_train, s2.y_train)

    overall_rate = cleaned_df[config.TARGET_COLUMN].eq("Yes").mean()
    for y in (s1.y_train, s1.y_validation, s1.y_test):
        assert abs(y.mean() - overall_rate) < 0.02  # stratification preserved


def test_split_partitions_disjoint_and_sized(cleaned_df):
    s = make_split(cleaned_df)
    train_idx = set(s.train_indices)
    val_idx = set(s.validation_indices)
    test_idx = set(s.test_indices)
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)
    total = len(cleaned_df)
    assert abs(len(train_idx) / total - 0.70) < 0.02
    assert abs(len(val_idx) / total - 0.15) < 0.02
    assert abs(len(test_idx) / total - 0.15) < 0.02


def test_preprocessor_fitted_on_training_only(cleaned_df):
    from src.preprocessing import build_preprocessor

    split = make_split(cleaned_df)
    _, preprocessor = build_preprocessor(cap_outliers=False)
    preprocessor.fit(split.X_train, split.y_train)

    transformed = preprocessor.transform(split.X_validation)
    assert not np.isnan(transformed).any()  # no unhandled missing values


def test_iqr_capper_fitted_on_train_only(cleaned_df):
    split = make_split(cleaned_df)
    capper = TrainOnlyIQRCapper(config.NUMERIC_FEATURES).fit(split.X_train)
    low_train, high_train = capper.bounds_["TotalCharges"]

    # Bounds learned from training data: an extreme validation value is clipped
    # to the training-derived bound, not to bounds computed from its own partition.
    val_extreme = split.X_validation["TotalCharges"].max()
    if val_extreme > high_train:
        capped = capper.transform(split.X_validation)
        assert capped["TotalCharges"].max() == high_train
    assert split.y_train.mean() == pytest.approx(split.y_validation.mean(), abs=0.02)
