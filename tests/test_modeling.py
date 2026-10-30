"""Tests for model training, probability validity, and coefficient extraction."""

import numpy as np
import pytest

from src.modeling import (
    logistic_regression_coefficients,
    train_lda,
    train_logistic_regression,
)
from src.preprocessing import make_split


@pytest.fixture(scope="module")
def split(real_cleaned_df):
    return make_split(real_cleaned_df)


@pytest.fixture(scope="module")
def logreg(split):
    return train_logistic_regression(split)


@pytest.fixture(scope="module")
def lda(split):
    return train_lda(split)


def test_both_models_succeed(logreg, lda):
    assert logreg.training_status == "success"
    assert lda.training_status == "success"


def test_probability_dimensions_match_labels(logreg, lda, split):
    assert len(logreg.validation_probabilities) == len(split.y_validation)
    assert len(logreg.test_probabilities) == len(split.y_test)
    assert len(lda.validation_probabilities) == len(split.y_validation)
    assert len(lda.test_probabilities) == len(split.y_test)


def test_probabilities_finite_and_in_range(logreg, lda):
    for result in (logreg, lda):
        for probs in (result.validation_probabilities, result.test_probabilities):
            assert np.all(np.isfinite(probs))
            assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_positive_class_probability_is_churn(logreg, split):
    """Higher observed churn should correlate with higher predicted probability."""
    val_churn = (split.y_validation == 1).to_numpy()
    probs = logreg.validation_probabilities
    assert probs[val_churn].mean() > probs[~val_churn].mean()


def test_both_models_share_the_same_split(logreg, lda):
    assert (
        logreg.configuration["random_state"] == lda.configuration["random_state"]
    )
    assert logreg.configuration["cap_outliers"] == lda.configuration["cap_outliers"]


def test_coefficients_extraction(logreg):
    coefs = logistic_regression_coefficients(logreg)
    assert len(coefs) > 0
    values = [c["coefficient"] for c in coefs]
    assert all(np.isfinite(values))
    # Ranked by absolute value, descending.
    abs_values = [abs(v) for v in values]
    assert abs_values == sorted(abs_values, reverse=True)


def test_coefficients_rejected_for_lda(lda):
    with pytest.raises(ValueError):
        logistic_regression_coefficients(lda)


def test_training_failure_is_transparent_not_fabricated(split):
    """A model that cannot fit returns failure status with a message, not fake data."""
    from src.modeling import _fit_model

    def broken_factory():
        raise RuntimeError("synthetic failure")

    result = _fit_model("Broken", split, broken_factory, {}, cap_outliers=False)
    assert result.training_status == "failure"
    assert "synthetic failure" in result.error_message
    assert result.validation_probabilities is None
