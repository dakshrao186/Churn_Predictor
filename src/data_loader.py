"""CSV loading and schema validation.

Owns: accessing the Telco CSV (local file first, documented public URL as a
controlled fallback) and validating that the required columns exist. Loading
is cached so Streamlit reruns do not re-read or re-download the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or fails schema validation."""


@dataclass
class LoadedDataset:
    """Raw, untouched dataset snapshot plus source metadata."""

    raw_df: pd.DataFrame
    source_label: str
    source_location: str
    schema_status: str = "valid"
    schema_messages: list[str] = field(default_factory=list)


def _validate_schema(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Return (status, messages) describing schema conformance."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return "invalid", [
            "Missing required column(s): " + ", ".join(missing),
            "The app cannot continue without these fields.",
        ]
    extra = [c for c in df.columns if c not in config.REQUIRED_COLUMNS]
    messages = []
    if extra:
        messages.append(
            "Note: the file has extra column(s) not required by the schema: "
            + ", ".join(extra)
        )
    return "valid", messages


def load_raw_dataframe(path: str | None = None) -> pd.DataFrame:
    """Load the Telco CSV into a DataFrame, preferring the local copy.

    If the local file is missing, attempts a one-time download of the
    documented public IBM source. Raises DataLoadError with a user-safe
    message when neither path works. Never fabricates data.
    """
    local_path = Path(path) if path else config.RAW_CSV_PATH
    if local_path.exists():
        try:
            df = pd.read_csv(local_path)
        except Exception as exc:  # malformed CSV
            raise DataLoadError(
                f"The local CSV '{local_path.name}' could not be parsed ({exc}). "
                "Replace or delete the file and reload the app."
            ) from exc
        logger.info("Loaded dataset from %s", local_path)
        return df

    # Controlled fallback: download the documented public source once.
    logger.info("Local CSV not found; downloading from %s", config.DATASET_SOURCE_URL)
    try:
        response = requests.get(config.DATASET_SOURCE_URL, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise DataLoadError(
            "The dataset file was not found in the project 'data/' folder and "
            f"the public download failed ({exc}).\n\n"
            f"Fix: download the CSV manually from\n{config.DATASET_SOURCE_URL}\n"
            f"and save it as 'data/{config.RAW_CSV_PATH.name}'."
        ) from exc

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)
    df = pd.read_csv(local_path)
    logger.info("Downloaded and saved dataset to %s", local_path)
    return df


def load_dataset() -> LoadedDataset:
    """Cached entry point: load the raw dataset and validate its schema."""
    df = load_raw_dataframe()
    status, messages = _validate_schema(df)
    source_label = config.DATASET_SOURCE_LABEL
    return LoadedDataset(
        raw_df=df,
        source_label=source_label,
        source_location=str(config.RAW_CSV_PATH),
        schema_status=status,
        schema_messages=messages,
    )


# Streamlit-friendly cached wrapper (import-safe: streamlit may be absent in tests).
try:  # pragma: no cover - exercised only inside a Streamlit runtime
    import streamlit as _st

    @_st.cache_data(show_spinner="Loading dataset…")
    def load_dataset_cached() -> LoadedDataset:
        """Cache the raw load so Streamlit reruns do not re-read/re-download."""
        return load_dataset()

except ImportError:  # pragma: no cover
    load_dataset_cached = load_dataset  # type: ignore[assignment]
