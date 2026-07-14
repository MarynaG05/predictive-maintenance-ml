"""CSV dataset loading utilities.

The loader returns a defensive copy containing all configured REQUIRED_COLUMNS
in deterministic order. Identifier, target, and failure-mode columns are
included because later stages need them for traceability and supervised
learning. Extra source columns are allowed during validation but are excluded
from the returned DataFrame.
"""

import csv
from pathlib import Path

import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.exceptions import (
    DataLoadingError,
    DuplicateColumnsError,
    EmptyDatasetError,
)
from predictive_maintenance.validation import validate_dataframe_structure


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the AI4I dataset CSV and validate its structural schema."""
    dataset_path = Path(path) if path is not None else _default_dataset_path()

    if not dataset_path.exists():
        raise DataLoadingError(f"Dataset file does not exist: {dataset_path}")

    if not dataset_path.is_file():
        raise DataLoadingError(f"Dataset path is not a regular file: {dataset_path}")

    _validate_source_header_unique(dataset_path)

    try:
        dataframe = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("Dataset must contain at least one row.") from exc
    except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        raise DataLoadingError(f"Failed to load dataset CSV: {dataset_path}") from exc

    validate_dataframe_structure(dataframe)

    return dataframe.loc[:, config.REQUIRED_COLUMNS].copy()


def _default_dataset_path() -> Path:
    """Return the configured default raw dataset path."""
    return config.RAW_DATA_DIR / config.DATASET_FILENAME


def _read_csv_header(path: Path) -> tuple[str, ...]:
    """Read the first CSV record using standard CSV parsing rules."""
    try:
        with path.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            return tuple(next(reader, ()))
    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        raise DataLoadingError(f"Failed to read dataset CSV header: {path}") from exc


def _validate_source_header_unique(path: Path) -> None:
    """Raise if the source CSV header contains duplicate column names."""
    header = _read_csv_header(path)
    seen_columns: set[str] = set()
    duplicate_columns: list[str] = []
    duplicate_seen: set[str] = set()

    for column in header:
        if column in seen_columns and column not in duplicate_seen:
            duplicate_columns.append(column)
            duplicate_seen.add(column)
        seen_columns.add(column)

    if duplicate_columns:
        raise DuplicateColumnsError(tuple(duplicate_columns))
