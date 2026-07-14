from pathlib import Path

import pytest

from predictive_maintenance import config

EXPECTED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}

EXPECTED_MODEL_FEATURES = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

EXPECTED_REQUIRED_COLUMNS = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)


def test_project_root_resolves_to_repository_root() -> None:
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / ".git").is_dir()


def test_paths_are_derived_from_project_root() -> None:
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.PROCESSED_DATA_DIR == config.DATA_DIR / "processed"
    assert config.REPORTS_DIR == config.PROJECT_ROOT / "reports"
    assert config.FIGURES_DIR == config.REPORTS_DIR / "figures"
    assert config.MODELS_DIR == config.PROJECT_ROOT / "models"


def test_target_column_is_not_a_model_feature() -> None:
    assert config.TARGET_COLUMN not in config.MODEL_FEATURES


def test_identifier_columns_are_not_model_features() -> None:
    assert set(config.IDENTIFIER_COLUMNS).isdisjoint(config.MODEL_FEATURES)


def test_failure_mode_columns_are_not_model_features() -> None:
    assert set(config.FAILURE_MODE_COLUMNS).isdisjoint(config.MODEL_FEATURES)


def test_categorical_and_numerical_features_equal_model_features() -> None:
    expected_features = config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES

    assert config.MODEL_FEATURES == expected_features


def test_model_features_match_expected_predictive_columns() -> None:
    assert config.MODEL_FEATURES == EXPECTED_MODEL_FEATURES


def test_model_features_are_unique() -> None:
    assert len(config.MODEL_FEATURES) == len(set(config.MODEL_FEATURES))


def test_required_columns_include_every_expected_dataset_column() -> None:
    assert set(config.REQUIRED_COLUMNS) == EXPECTED_COLUMNS


def test_required_columns_follow_expected_order() -> None:
    assert config.REQUIRED_COLUMNS == EXPECTED_REQUIRED_COLUMNS


def test_required_columns_are_unique() -> None:
    assert len(config.REQUIRED_COLUMNS) == len(set(config.REQUIRED_COLUMNS))


def test_split_values_are_valid() -> None:
    assert config.RANDOM_SEED == 42
    assert 0.0 < config.TEST_SIZE < 1.0
    assert 0.0 < config.VALIDATION_SIZE < 1.0


def test_split_values_represent_overall_60_20_20_split() -> None:
    test_proportion = config.TEST_SIZE
    validation_proportion = (1.0 - config.TEST_SIZE) * config.VALIDATION_SIZE
    train_proportion = 1.0 - test_proportion - validation_proportion

    assert train_proportion == pytest.approx(0.60)
    assert validation_proportion == pytest.approx(0.20)
    assert test_proportion == pytest.approx(0.20)
