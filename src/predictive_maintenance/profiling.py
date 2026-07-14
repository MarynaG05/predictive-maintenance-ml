"""Reusable dataset profiling utilities for validated AI4I data."""

import json
import math
import numbers
from pathlib import Path
from typing import Any

import pandas as pd

from predictive_maintenance import config

HIGH_IMBALANCE_RATIO: float = 10.0


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable profile for an already validated DataFrame."""
    profile = {
        "dataset_summary": _dataset_summary(df),
        "schema_summary": _schema_summary(),
        "missing_values": _missing_values(df),
        "target_summary": _target_summary(df),
        "numerical_feature_summary": _numerical_feature_summary(df),
        "categorical_feature_summary": _categorical_feature_summary(df),
    }
    profile["ml_warnings"] = _ml_warnings(df, profile)
    return profile


def save_profile(profile: dict[str, Any], output_dir: Path | str) -> dict[str, Path]:
    """Write a dataset profile JSON file and a compact summary CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "dataset_profile.json"
    summary_path = output_path / "dataset_summary.csv"

    json_path.write_text(
        json.dumps(profile, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"metric": metric, "value": value}
            for metric, value in profile["dataset_summary"].items()
        ]
    ).to_csv(summary_path, index=False)

    return {
        "json": json_path,
        "summary_csv": summary_path,
    }


def _dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    """Summarize dimensions, deep memory usage, and duplicate rows."""
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def _schema_summary() -> dict[str, Any]:
    """Summarize configured schema groups used by the ML workflow."""
    return {
        "identifier_columns": list(config.IDENTIFIER_COLUMNS),
        "numerical_features": list(config.NUMERICAL_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "target_column": config.TARGET_COLUMN,
        "leakage_columns": list(config.FAILURE_MODE_COLUMNS),
    }


def _missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize missing values by column on a 0-100 percentage scale."""
    missing_counts = df.isna().sum()
    row_count = len(df)
    return {
        "missing_count_per_column": {
            column: int(missing_counts[column]) for column in df.columns
        },
        "missing_percentage_per_column": {
            column: _percentage(missing_counts[column], row_count)
            for column in df.columns
        },
        "total_missing_cells": int(missing_counts.sum()),
    }


def _target_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize target class distribution and imbalance ratio."""
    target = df[config.TARGET_COLUMN].dropna()
    counts = target.value_counts().sort_index()
    total = int(counts.sum())
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) > 1 else None

    return {
        "class_counts": {
            _json_key(label): int(count) for label, count in counts.items()
        },
        "class_percentages": {
            _json_key(label): _percentage(count, total)
            for label, count in counts.items()
        },
        "imbalance_ratio": imbalance_ratio,
    }


def _numerical_feature_summary(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Summarize numerical features using pandas sample standard deviation."""
    summary: dict[str, dict[str, float | None]] = {}
    for column in config.NUMERICAL_FEATURES:
        series = df[column]
        summary[column] = {
            "mean": _optional_float(series.mean()),
            "median": _optional_float(series.median()),
            "std": _optional_float(series.std()),
            "min": _optional_float(series.min()),
            "max": _optional_float(series.max()),
            "q1": _optional_float(series.quantile(0.25)),
            "q3": _optional_float(series.quantile(0.75)),
        }
    return summary


def _categorical_feature_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize categories as all-row 0-100 percentages, with NA as <NA>."""
    summary: dict[str, dict[str, Any]] = {}
    for column in config.CATEGORICAL_FEATURES:
        counts = df[column].value_counts(dropna=False).sort_index()
        total = int(counts.sum())
        summary[column] = {
            "unique_values": [_json_key(value) for value in counts.index],
            "counts": {_json_key(value): int(count) for value, count in counts.items()},
            "percentages": {
                _json_key(value): _percentage(count, total)
                for value, count in counts.items()
            },
        }
    return summary


def _ml_warnings(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
    """Create human-readable warnings for common ML data risks."""
    warnings: list[str] = []

    imbalance_ratio = profile["target_summary"]["imbalance_ratio"]
    missing_target_count = int(df[config.TARGET_COLUMN].isna().sum())
    observed_target_classes = len(profile["target_summary"]["class_counts"])

    if missing_target_count > 0:
        warnings.append(
            f"Target column contains {missing_target_count} missing values."
        )

    if observed_target_classes == 0:
        warnings.append(
            "Target column contains no observed classes and cannot support binary "
            "classification."
        )
    elif observed_target_classes == 1:
        warnings.append(
            "Target column contains only one observed class and cannot support binary "
            "classification."
        )

    if imbalance_ratio is not None and imbalance_ratio >= HIGH_IMBALANCE_RATIO:
        warnings.append(
            "Highly imbalanced target detected: majority/minority ratio is "
            f"{imbalance_ratio:.2f}."
        )

    duplicate_rows = profile["dataset_summary"]["duplicate_row_count"]
    if duplicate_rows > 0:
        warnings.append(f"Duplicate rows detected: {duplicate_rows} duplicate row(s).")

    total_missing = profile["missing_values"]["total_missing_cells"]
    if total_missing > 0:
        warnings.append(f"Missing values detected: {total_missing} missing cell(s).")

    constant_columns = _constant_columns(df)
    if constant_columns:
        warnings.append(
            "Constant columns detected: " + ", ".join(constant_columns) + "."
        )

    unexpected_categories = _unexpected_categories(df)
    if unexpected_categories:
        details = "; ".join(
            f"{column}: {', '.join(values)}"
            for column, values in unexpected_categories.items()
        )
        warnings.append(f"Unexpected categories detected: {details}.")

    return warnings


def _constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns with a single observed value, treating NaN as a value."""
    return [
        column for column in df.columns if int(df[column].nunique(dropna=False)) == 1
    ]


def _unexpected_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return configured categorical columns containing unexpected values."""
    unexpected: dict[str, list[str]] = {}
    for column, expected_values in config.EXPECTED_CATEGORICAL_VALUES.items():
        if column not in df.columns:
            continue
        actual_values = set(df[column].dropna().astype(str))
        unknown_values = sorted(actual_values - set(expected_values))
        if unknown_values:
            unexpected[column] = unknown_values
    return unexpected


def _percentage(count: float, total: int) -> float:
    """Return a percentage rounded to two decimal places."""
    if total == 0:
        return 0.0
    return round(float(count) / total * 100.0, 2)


def _optional_float(value: Any) -> float | None:
    """Convert pandas/numpy numeric output to a JSON-safe float or None."""
    if pd.isna(value):
        return None
    float_value = float(value)
    if not math.isfinite(float_value):
        return None
    return float_value


def _json_key(value: Any) -> str:
    """Convert scalar labels to stable JSON object keys."""
    if pd.isna(value):
        return "<NA>"
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value).is_integer()
    ):
        return str(int(value))
    return str(value)
