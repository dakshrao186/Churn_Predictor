"""Reproducible raw-to-cleaned transformation with an explicit audit trail.

Owns: turning the raw Telco snapshot into a separate cleaned analysis
DataFrame plus a machine-readable quality report. The raw input is never
mutated. Model-time imputation stays in the scikit-learn pipeline
(src/preprocessing.py); this module only makes dataset-level repairs visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config


@dataclass
class QualityReport:
    """Machine-readable record of what cleaning actually did."""

    raw_shape: tuple[int, int] = (0, 0)
    cleaned_shape: tuple[int, int] = (0, 0)
    missing_before_total: int = 0
    missing_after_total: int = 0
    missing_before_by_column: dict[str, int] = field(default_factory=dict)
    missing_after_by_column: dict[str, int] = field(default_factory=dict)
    whitespace_blank_cells: int = 0
    duplicate_row_count: int = 0
    duplicate_customer_id_count: int = 0
    total_charges_conversion_count: int = 0
    zero_tenure_charge_correction_count: int = 0
    invalid_target_labels: list[str] = field(default_factory=list)
    duplicate_policy: str = "Retained (reported only)"
    actions: list[str] = field(default_factory=list)

    def to_actions_table(self) -> pd.DataFrame:
        """Compact (Check / Count / Action) table for the UI."""
        rows = [
            (
                "Whitespace-only cells treated as missing",
                self.whitespace_blank_cells,
                "Blank strings normalized to NaN before missingness checks",
            ),
            (
                "TotalCharges values converted to numeric",
                self.total_charges_conversion_count,
                "Text column parsed; invalid/blank entries became NaN",
            ),
            (
                "Zero-tenure blank TotalCharges set to 0",
                self.zero_tenure_charge_correction_count,
                "Contextual correction: 0 months tenure implies no charges yet",
            ),
            (
                "Duplicate rows",
                self.duplicate_row_count,
                "Retained and reported; no silent deletion",
            ),
            (
                "Duplicate customerID values",
                self.duplicate_customer_id_count,
                "Retained and reported; no silent deletion",
            ),
            (
                "Missing cells after cleaning (for pipeline imputation)",
                self.missing_after_total,
                "Handled at model time by train-fitted imputation only",
            ),
        ]
        if self.invalid_target_labels:
            rows.append(
                (
                    "Unexpected target labels in Churn",
                    len(self.invalid_target_labels),
                    "Reported; rows excluded from binary encoding",
                )
            )
        return pd.DataFrame(rows, columns=["Check", "Count", "Action / Policy"])


@dataclass
class CleaningResult:
    cleaned_df: pd.DataFrame
    quality_report: QualityReport


def _elementwise(df: pd.DataFrame, func) -> pd.DataFrame:
    """Apply ``func`` elementwise (pandas 2.2+ ``.map``, older ``.applymap``)."""
    mapper = getattr(df, "map", None)
    if mapper is None:  # pandas < 2.2
        mapper = df.applymap
    return mapper(func)


def count_whitespace_blanks(raw_df: pd.DataFrame) -> int:
    """Count cells that are non-null strings consisting only of whitespace."""
    mask = _elementwise(raw_df, lambda v: isinstance(v, str) and v.strip() == "")
    return int(mask.to_numpy().sum())


def clean_data(raw_df: pd.DataFrame) -> CleaningResult:
    """Produce a cleaned copy of ``raw_df`` plus a quality report.

    Steps (each reported, none silent):
      1. Record raw shape/missingness (whitespace counts as missing).
      2. Normalize whitespace-only strings to NaN.
      3. Convert TotalCharges to numeric (coercing invalid text to NaN).
      4. Apply the documented zero-tenure correction for blank TotalCharges.
      5. Validate target labels; keep only the two expected classes.
      6. Count duplicate rows and duplicate customer IDs (policy: retain).
    """
    if config.ID_COLUMN not in raw_df.columns or config.TARGET_COLUMN not in raw_df.columns:
        raise ValueError(
            "Cleaning requires at least the identifier and target columns."
        )

    df = raw_df.copy(deep=True)
    report = QualityReport(raw_shape=raw_df.shape)

    # --- 1. Missingness before cleaning (whitespace counted as missing) ---
    report.whitespace_blank_cells = count_whitespace_blanks(raw_df)
    missing_before = raw_df.isna().sum()
    missing_before = missing_before + (
        _elementwise(raw_df, lambda v: isinstance(v, str) and v.strip() == "").sum()
    )
    report.missing_before_by_column = {
        k: int(v) for k, v in missing_before.items() if v > 0
    }
    report.missing_before_total = int(missing_before.sum())

    # --- 2. Normalize whitespace-only strings to NaN ---
    text_columns = [
        col for col in df.columns
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
    ]
    for col in text_columns:
        df[col] = df[col].apply(lambda v: np.nan if isinstance(v, str) and v.strip() == "" else v)

    # --- 3. TotalCharges -> numeric ---
    if "TotalCharges" in df.columns:
        # Count raw entries that were not directly numeric-parseable
        # (blank whitespace strings or invalid text) before conversion.
        raw_col = raw_df["TotalCharges"]
        directly_numeric = pd.to_numeric(raw_col, errors="coerce")
        report.total_charges_conversion_count = int(
            (directly_numeric.isna() & raw_col.notna()).sum()
            + raw_col.isna().sum()
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # --- 4. Zero-tenure blank TotalCharges correction ---
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        zero_tenure_blank = (df["tenure"] == 0) & (df["TotalCharges"].isna())
        report.zero_tenure_charge_correction_count = int(zero_tenure_blank.sum())
        df.loc[zero_tenure_blank, "TotalCharges"] = 0.0

    # --- 5. Target validation ---
    invalid_mask = ~df[config.TARGET_COLUMN].isin(config.EXPECTED_TARGET_LABELS)
    report.invalid_target_labels = sorted(
        str(v) for v in df.loc[invalid_mask, config.TARGET_COLUMN].dropna().unique()
    )
    if report.invalid_target_labels:
        df = df[df[config.TARGET_COLUMN].isin(config.EXPECTED_TARGET_LABELS)].copy()
        report.actions.append(
            f"Removed {len(report.invalid_target_labels)} unexpected target label(s): "
            + ", ".join(report.invalid_target_labels)
        )

    # --- 6. Duplicate accounting (reported, retained) ---
    report.duplicate_row_count = int(df.duplicated().sum())
    report.duplicate_customer_id_count = int(
        df[config.ID_COLUMN].duplicated(keep=False).sum()
    )

    # --- Final missingness after cleaning ---
    missing_after = df.isna().sum()
    report.missing_after_by_column = {
        k: int(v) for k, v in missing_after.items() if v > 0
    }
    report.missing_after_total = int(missing_after.sum())
    report.cleaned_shape = df.shape

    report.actions.extend(
        [
            "Normalized whitespace-only cells to missing values",
            f"Parsed TotalCharges to numeric ({report.total_charges_conversion_count} blank/invalid value(s) became NaN)",
            f"Set TotalCharges to 0 for {report.zero_tenure_charge_correction_count} zero-tenure customer(s)",
            "Retained duplicate rows/IDs and reported them (no silent deletion)",
            "Remaining missing predictor values are handled at model time by train-fitted imputation",
        ]
    )
    return CleaningResult(cleaned_df=df, quality_report=report)
