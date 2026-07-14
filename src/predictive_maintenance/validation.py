"""Structural validation helpers for AI4I dataset inputs."""

import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.exceptions import (
    DuplicateColumnsError,
    EmptyDatasetError,
    MissingColumnsError,
)


def validate_not_empty(dataframe: pd.DataFrame) -> None:
    """Raise if the DataFrame has zero rows or zero columns."""
    if len(dataframe.index) == 0:
        raise EmptyDatasetError("Dataset must contain at least one row.")

    if len(dataframe.columns) == 0:
        raise EmptyDatasetError("Dataset must contain at least one column.")


def validate_unique_columns(dataframe: pd.DataFrame) -> None:
    """Raise if the DataFrame contains duplicated column names."""
    duplicated_columns = tuple(dataframe.columns[dataframe.columns.duplicated()])

    if duplicated_columns:
        raise DuplicateColumnsError(duplicated_columns)


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise if any configured required columns are missing."""
    missing_columns = tuple(
        column for column in config.REQUIRED_COLUMNS if column not in dataframe.columns
    )

    if missing_columns:
        raise MissingColumnsError(missing_columns)


def validate_dataframe_structure(dataframe: pd.DataFrame) -> None:
    """Run structural checks required before downstream data processing."""
    validate_not_empty(dataframe)
    validate_unique_columns(dataframe)
    validate_required_columns(dataframe)
