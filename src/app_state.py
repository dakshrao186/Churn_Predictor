"""Shared application state: cached split, models, and evaluations.

Module-level cached functions so Streamlit's cache survives reruns without
redefining function objects. Keys include a dataset fingerprint, seed, and
model configuration — never the interactive threshold.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.evaluation import evaluate
from src.modeling import train_lda, train_logistic_regression
from src.preprocessing import DataSplit, make_split


def dataset_fingerprint(cleaned_df: pd.DataFrame) -> int:
    """Cheap stable fingerprint of the cleaned dataset for cache keys."""
    return int(
        pd.util.hash_pandas_object(cleaned_df.head(500), index=True).sum() & 0x7FFFFFFF
    ) ^ len(cleaned_df)


@st.cache_data(show_spinner="Preparing stratified split…")
def cached_split(_fingerprint: int, df: pd.DataFrame,
                 random_state: int = config.DEFAULT_RANDOM_SEED) -> DataSplit:
    """Reproducible stratified split, keyed on dataset identity + seed."""
    return make_split(df, random_state=random_state)


@st.cache_resource(show_spinner="Training Logistic Regression and LDA…")
def cached_models(
    _fingerprint: int,
    df: pd.DataFrame,
    cap_outliers: bool = False,
    C: float = config.LOGREG_DEFAULT_C,
    random_state: int = config.DEFAULT_RANDOM_SEED,
) -> dict:
    """Fit both models on the shared split; keyed on the training configuration.

    The threshold is deliberately absent from the key: threshold changes
    reuse these cached probabilities without refitting.
    """
    split = cached_split(_fingerprint, df, random_state)
    if cap_outliers:
        from src.preprocessing import TrainOnlyIQRCapper

        capper = TrainOnlyIQRCapper(config.NUMERIC_FEATURES).fit(split.X_train)
        split = DataSplit(
            X_train=capper.transform(split.X_train),
            X_validation=capper.transform(split.X_validation),
            X_test=capper.transform(split.X_test),
            y_train=split.y_train, y_validation=split.y_validation, y_test=split.y_test,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            random_state=split.random_state,
            feature_columns=split.feature_columns,
            preprocessing_configuration={**split.preprocessing_configuration, "cap_outliers": True},
        )
    return {
        "logreg": train_logistic_regression(split, C=C, cap_outliers=False),
        "lda": train_lda(split, cap_outliers=False),
        "split": split,
    }


def models_and_evaluations(
    cleaned_df: pd.DataFrame,
    cap_outliers: bool = False,
    C: float = config.LOGREG_DEFAULT_C,
) -> tuple[dict, dict]:
    """Bundle + validation/test EvaluationResults for both models (or None)."""
    fingerprint = dataset_fingerprint(cleaned_df)
    bundle = cached_models(fingerprint, cleaned_df, cap_outliers, C)
    split: DataSplit = bundle["split"]

    evaluations: dict = {}
    for key, result in [("logreg", bundle["logreg"]), ("lda", bundle["lda"])]:
        if result.is_valid:
            evaluations[key + "_val"] = evaluate(
                split.y_validation, result.validation_probabilities,
                result.model_name, "Validation / Interactive",
            )
            evaluations[key + "_test"] = evaluate(
                split.y_test, result.test_probabilities,
                result.model_name, "Test / Final Evaluation",
                config.DEFAULT_THRESHOLD,
            )
        else:
            evaluations[key + "_val"] = None
            evaluations[key + "_test"] = None
    return bundle, evaluations
