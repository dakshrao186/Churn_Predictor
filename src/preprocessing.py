"""Feature/target definition, stratified splits, and sklearn preprocessing.

Owns: building X and y, creating ONE reproducible stratified
train/validation/test split shared by both models, and fitting a
ColumnTransformer pipeline on training data only.

Leakage invariants enforced here:
  * customerID and Churn never appear in the predictor matrix.
  * Imputers, encoders, and scalers are fitted on X_train only.
  * Optional IQR capping bounds are learned from training data only and
    applied unchanged to validation/test partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class SplitError(Exception):
    """Raised when feature/target preparation or splitting cannot proceed."""


@dataclass
class DataSplit:
    """Single source of truth for both model comparisons."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_indices: pd.Index
    validation_indices: pd.Index
    test_indices: pd.Index
    random_state: int
    feature_columns: list[str]
    preprocessing_configuration: dict = field(default_factory=dict)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the predictor matrix X and binary target y from cleaned data.

    y encodes Churn as Yes=1 (positive) / No=0. customerID and Churn are
    excluded from X by construction.
    """
    missing = [c for c in [config.TARGET_COLUMN, *config.NUMERIC_FEATURES] if c not in df.columns]
    if missing:
        raise SplitError(f"Cannot build features; missing column(s): {', '.join(missing)}")

    y_raw = df[config.TARGET_COLUMN]
    unexpected = sorted(set(y_raw.dropna().unique()) - set(config.EXPECTED_TARGET_LABELS))
    if unexpected:
        raise SplitError(
            "Target column contains unexpected labels: " + ", ".join(map(str, unexpected))
        )

    y = (y_raw == config.POSITIVE_LABEL).astype(int)
    if y.nunique() != 2:
        raise SplitError("Both churn classes must be present to train a classifier.")

    feature_columns = [c for c in df.columns if c not in (config.ID_COLUMN, config.TARGET_COLUMN)]
    X = df[feature_columns].copy()
    if config.TARGET_COLUMN in X.columns or config.ID_COLUMN in X.columns:
        raise SplitError("Invariant violated: identifier or target found in features.")
    return X, y


def make_split(
    df: pd.DataFrame,
    random_state: int = config.DEFAULT_RANDOM_SEED,
    cap_outliers: bool = False,
) -> DataSplit:
    """Create the shared stratified 70/15/15 split using two sequential splits."""
    X, y = prepare_xy(df)

    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, train_size=config.TRAIN_FRACTION, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout,
        y_holdout,
        train_size=config.VALIDATION_FRACTION,
        random_state=random_state,
        stratify=y_holdout,
    )

    return DataSplit(
        X_train=X_train_full,
        X_validation=X_valid,
        X_test=X_test,
        y_train=y_train_full,
        y_validation=y_valid,
        y_test=y_test,
        train_indices=X_train_full.index,
        validation_indices=X_valid.index,
        test_indices=X_test.index,
        random_state=random_state,
        feature_columns=list(X.columns),
        preprocessing_configuration={
            "cap_outliers": cap_outliers,
            "random_state": random_state,
            "train_fraction": config.TRAIN_FRACTION,
        },
    )


class TrainOnlyIQRCapper:
    """IQR capper whose bounds are learned from the training partition only.

    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are clipped to the
    bounds. Fitted on numeric predictor columns of X_train; applied
    unchanged to validation/test.
    """

    def __init__(self, numeric_columns: list[str]):
        self.numeric_columns = numeric_columns
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X_train: pd.DataFrame) -> "TrainOnlyIQRCapper":
        self.bounds_ = {}
        for col in self.numeric_columns:
            if col not in X_train.columns:
                continue
            q1, q3 = X_train[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (low, high) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low, upper=high)
        return X


def build_preprocessor(cap_outliers: bool = False, X_train: pd.DataFrame | None = None):
    """Build the (optionally outlier-capped) preprocessing pipeline.

    When ``cap_outliers`` is True, ``X_train`` is required: the IQR bounds
    are computed from training data only. The returned object is a tuple
    (capper_or_None, ColumnTransformer).
    """
    capper = None
    if cap_outliers:
        if X_train is None:
            raise ValueError("X_train is required to fit the IQR capper on training data only.")
        capper = TrainOnlyIQRCapper(config.NUMERIC_FEATURES).fit(X_train)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )
    return capper, preprocessor


def _apply_capper(capper: TrainOnlyIQRCapper | None, X: pd.DataFrame) -> pd.DataFrame:
    return capper.transform(X) if capper is not None else X


def make_logreg_pipeline(C: float = config.LOGREG_DEFAULT_C) -> Pipeline:
    """Logistic Regression pipeline; scaling fitted on training data only."""
    _, preprocessor = build_preprocessor(cap_outliers=False)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=config.LOGREG_MAX_ITER, C=C)),
        ]
    )


def make_lda_pipeline():
    """LDA pipeline using the same feature space as Logistic Regression.

    Uses a shrinkage-capable configuration ('lsqr' + 'auto' shrinkage) for
    stability with the one-hot encoded feature matrix.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    _, preprocessor = build_preprocessor(cap_outliers=False)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )
