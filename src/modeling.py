"""Fitting Logistic Regression and LDA, plus positive-class probability generation.

Owns: building model bundles from a shared DataSplit. Both models are fitted
on the identical training partition; probabilities are aligned positionally
with the y arrays. Optional IQR capping is fitted on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config
from src.preprocessing import DataSplit, build_preprocessor


@dataclass
class ModelResult:
    """Fitted model plus validation/test probabilities (aligned with y arrays)."""

    model_name: str
    fitted_pipeline: object | None
    validation_probabilities: np.ndarray | None
    test_probabilities: np.ndarray | None
    configuration: dict = field(default_factory=dict)
    training_status: str = "success"
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.training_status == "success"


def _positive_class_probabilities(pipeline, X) -> np.ndarray:
    """Extract P(churn = 1), checking classes_ instead of assuming column order."""
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_) if hasattr(pipeline, "classes_") else list(
        pipeline.named_steps["classifier"].classes_
    )
    if config.POSITIVE_CLASS not in classes:
        raise ValueError("Positive churn class missing from model classes_.")
    return proba[:, classes.index(config.POSITIVE_CLASS)]


def _validate_probabilities(proba: np.ndarray, expected_size: int, model_name: str) -> None:
    if proba.shape[0] != expected_size:
        raise ValueError(
            f"{model_name}: probability count {proba.shape[0]} != label count {expected_size}."
        )
    if not np.all(np.isfinite(proba)):
        raise ValueError(f"{model_name}: probabilities contain non-finite values.")
    if proba.min() < 0.0 or proba.max() > 1.0:
        raise ValueError(f"{model_name}: probabilities fall outside [0, 1].")


def _fit_model(
    model_name: str,
    split: DataSplit,
    pipeline_factory,
    configuration: dict,
    cap_outliers: bool,
) -> ModelResult:
    """Shared fitting path for both classifiers."""
    try:
        if cap_outliers:
            capper, preprocessor = build_preprocessor(cap_outliers=True, X_train=split.X_train)
            X_train = capper.transform(split.X_train)
            X_validation = capper.transform(split.X_validation)
            X_test = capper.transform(split.X_test)
        else:
            _, preprocessor = build_preprocessor(cap_outliers=False)
            X_train, X_validation, X_test = split.X_train, split.X_validation, split.X_test

        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", pipeline_factory())])
        pipeline.fit(X_train, split.y_train)

        val_proba = _positive_class_probabilities(pipeline, X_validation)
        test_proba = _positive_class_probabilities(pipeline, X_test)
        _validate_probabilities(val_proba, len(split.y_validation), model_name)
        _validate_probabilities(test_proba, len(split.y_test), model_name)

        return ModelResult(
            model_name=model_name,
            fitted_pipeline=pipeline,
            validation_probabilities=val_proba,
            test_probabilities=test_proba,
            configuration=configuration,
        )
    except Exception as exc:  # model fitting failure — transparent, not fabricated
        return ModelResult(
            model_name=model_name,
            fitted_pipeline=None,
            validation_probabilities=None,
            test_probabilities=None,
            configuration=configuration,
            training_status="failure",
            error_message=f"{model_name} could not be fitted: {exc}",
        )


def train_logistic_regression(
    split: DataSplit, C: float = config.LOGREG_DEFAULT_C, cap_outliers: bool = False
) -> ModelResult:
    """Fit Logistic Regression on the shared training partition."""

    def factory():
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER, C=C, random_state=config.DEFAULT_RANDOM_SEED
        )

    return _fit_model(
        "Logistic Regression",
        split,
        factory,
        configuration={"model": "logreg", "C": C, "cap_outliers": cap_outliers,
                       "random_state": split.random_state},
        cap_outliers=cap_outliers,
    )


def train_lda(split: DataSplit, cap_outliers: bool = False) -> ModelResult:
    """Fit LDA (lsqr solver, auto shrinkage) on the shared training partition."""

    def factory():
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    return _fit_model(
        "LDA",
        split,
        factory,
        configuration={"model": "lda", "cap_outliers": cap_outliers,
                       "random_state": split.random_state},
        cap_outliers=cap_outliers,
    )


def logistic_regression_coefficients(model_result: ModelResult) -> list[dict]:
    """Ranked encoded-feature coefficients for the fitted Logistic Regression.

    Returns [{feature, coefficient}] sorted by absolute coefficient value,
    descending. Names combine numeric columns with one-hot encoded levels
    (e.g. 'Contract_Month-to-month'). Raises if the model is not Logistic
    Regression or not fitted.
    """
    if not model_result.is_valid or model_result.fitted_pipeline is None:
        raise ValueError("Model is not fitted; no coefficients available.")
    if model_result.configuration.get("model") != "logreg":
        raise ValueError("Coefficients are only extracted from Logistic Regression.")

    pipeline = model_result.fitted_pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split("__", 1)[-1] for name in feature_names]
    coefficients = classifier.coef_[0]

    ranked = sorted(
        (
            {"feature": name, "coefficient": float(coef)}
            for name, coef in zip(feature_names, coefficients)
        ),
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )
    return ranked
