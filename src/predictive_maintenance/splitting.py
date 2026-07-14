"""Reproducible train-validation-test splitting utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_maintenance import config
from predictive_maintenance.exceptions import DatasetSplitError


@dataclass(frozen=True)
class SplitData:
    """Container for aligned train, validation, and final test splits."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(dataframe: pd.DataFrame) -> SplitData:
    """Split validated data into 60/20/20 splits, preserving source indices."""
    X = dataframe.loc[:, config.MODEL_FEATURES].copy()
    y = dataframe.loc[:, config.TARGET_COLUMN].copy()
    _validate_target_for_stratification(y)

    try:
        X_development, X_test, y_development, y_test = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=y,
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_development,
            y_development,
            test_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=y_development,
        )
    except ValueError as exc:
        raise DatasetSplitError(f"Unable to create stratified splits: {exc}") from exc

    return SplitData(
        X_train=X_train.copy(),
        X_validation=X_validation.copy(),
        X_test=X_test.copy(),
        y_train=y_train.copy(),
        y_validation=y_validation.copy(),
        y_test=y_test.copy(),
    )


def _validate_target_for_stratification(target: pd.Series) -> None:
    """Validate the binary target can support two stratified split operations."""
    if target.isna().any():
        raise DatasetSplitError(
            "Target column contains missing values; remove or resolve missing "
            "targets before splitting."
        )

    observed_labels = set(target.unique())
    if len(observed_labels) == 1:
        raise DatasetSplitError(
            "Target must contain both classes 0 and 1 for binary stratified splitting."
        )

    if observed_labels != {0, 1}:
        labels = ", ".join(str(label) for label in sorted(observed_labels, key=str))
        raise DatasetSplitError(
            "Target labels must be exactly {0, 1} for binary classification; "
            f"observed labels: {labels}."
        )

    class_counts = target.value_counts()

    minimum_class_count = int(class_counts.min())
    if minimum_class_count < 3:
        raise DatasetSplitError(
            "Each target class must contain at least 3 rows to support "
            "stratified train-validation-test splitting."
        )
