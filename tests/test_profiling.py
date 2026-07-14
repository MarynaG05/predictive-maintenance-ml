import json
from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance import config
from predictive_maintenance.profiling import profile_dataset, save_profile


def make_profile_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "UDI": 1,
                "Product ID": "M14860",
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
            {
                "UDI": 2,
                "Product ID": "L47181",
                "Type": "L",
                "Air temperature [K]": 299.0,
                "Process temperature [K]": 309.2,
                "Rotational speed [rpm]": 1408,
                "Torque [Nm]": 46.3,
                "Tool wear [min]": 3,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
            {
                "UDI": 3,
                "Product ID": "H29424",
                "Type": "H",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1300,
                "Torque [Nm]": 50.0,
                "Tool wear [min]": 10,
                "Machine failure": 1,
                "TWF": 1,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
        ],
        columns=config.REQUIRED_COLUMNS,
    )


def test_profile_dataset_returns_expected_structure_for_valid_dataset() -> None:
    dataframe = make_profile_dataframe()

    profile = profile_dataset(dataframe)

    assert set(profile) == {
        "dataset_summary",
        "schema_summary",
        "missing_values",
        "target_summary",
        "numerical_feature_summary",
        "categorical_feature_summary",
        "ml_warnings",
    }
    assert profile["dataset_summary"]["row_count"] == 3
    assert profile["dataset_summary"]["column_count"] == len(config.REQUIRED_COLUMNS)
    assert profile["dataset_summary"]["memory_usage_bytes"] > 0
    assert profile["dataset_summary"]["duplicate_row_count"] == 0
    assert profile["schema_summary"]["identifier_columns"] == list(
        config.IDENTIFIER_COLUMNS
    )
    assert profile["schema_summary"]["target_column"] == config.TARGET_COLUMN
    assert profile["schema_summary"]["leakage_columns"] == list(
        config.FAILURE_MODE_COLUMNS
    )
    assert not any(
        "Leakage columns present" in warning for warning in profile["ml_warnings"]
    )


def test_profile_outputs_follow_deterministic_ordering() -> None:
    profile = profile_dataset(make_profile_dataframe())

    assert list(profile["missing_values"]["missing_count_per_column"]) == list(
        config.REQUIRED_COLUMNS
    )
    assert list(profile["numerical_feature_summary"]) == list(config.NUMERICAL_FEATURES)
    assert list(profile["categorical_feature_summary"]) == list(
        config.CATEGORICAL_FEATURES
    )


def test_profile_reports_missing_values() -> None:
    dataframe = make_profile_dataframe()
    dataframe.loc[0, "Torque [Nm]"] = pd.NA

    profile = profile_dataset(dataframe)

    assert profile["missing_values"]["missing_count_per_column"]["Torque [Nm]"] == 1
    assert (
        profile["missing_values"]["missing_percentage_per_column"]["Torque [Nm]"]
        == 33.33
    )
    assert profile["missing_values"]["total_missing_cells"] == 1
    assert any(
        "Missing values detected" in warning for warning in profile["ml_warnings"]
    )


def test_profile_reports_duplicate_rows() -> None:
    dataframe = make_profile_dataframe()
    dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)

    profile = profile_dataset(dataframe)

    assert profile["dataset_summary"]["duplicate_row_count"] == 1
    assert any(
        "Duplicate rows detected" in warning for warning in profile["ml_warnings"]
    )


def test_profile_reports_constant_columns() -> None:
    dataframe = make_profile_dataframe()

    profile = profile_dataset(dataframe)

    assert "Constant columns detected: HDF, PWF, OSF, RNF." in profile["ml_warnings"]


def test_all_missing_numerical_feature_is_missing_and_constant() -> None:
    dataframe = make_profile_dataframe()
    dataframe["Torque [Nm]"] = pd.NA

    profile = profile_dataset(dataframe)

    assert profile["missing_values"]["missing_count_per_column"]["Torque [Nm]"] == 3
    assert (
        profile["missing_values"]["missing_percentage_per_column"]["Torque [Nm]"]
        == 100.0
    )
    assert profile["numerical_feature_summary"]["Torque [Nm]"] == {
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
    }
    assert (
        "Constant columns detected: Torque [Nm], HDF, PWF, OSF, RNF."
        in profile["ml_warnings"]
    )


def test_profile_reports_all_missing_categorical_feature() -> None:
    dataframe = make_profile_dataframe()
    dataframe["Type"] = pd.NA

    profile = profile_dataset(dataframe)
    summary = profile["categorical_feature_summary"]["Type"]

    assert summary["unique_values"] == ["<NA>"]
    assert summary["counts"] == {"<NA>": 3}
    assert summary["percentages"] == {"<NA>": 100.0}
    assert (
        "Constant columns detected: Type, HDF, PWF, OSF, RNF." in profile["ml_warnings"]
    )


def test_profile_reports_two_class_imbalance() -> None:
    majority = pd.concat([make_profile_dataframe().iloc[[0]]] * 20, ignore_index=True)
    minority = make_profile_dataframe().iloc[[2]].copy()
    dataframe = pd.concat([majority, minority], ignore_index=True)

    profile = profile_dataset(dataframe)

    assert profile["target_summary"]["imbalance_ratio"] == 20.0
    assert any(
        "Highly imbalanced target detected" in warning
        for warning in profile["ml_warnings"]
    )


def test_profile_reports_one_class_target() -> None:
    dataframe = make_profile_dataframe()
    dataframe["Machine failure"] = 0

    profile = profile_dataset(dataframe)

    assert profile["target_summary"]["class_counts"] == {"0": 3}
    assert profile["target_summary"]["class_percentages"] == {"0": 100.0}
    assert profile["target_summary"]["imbalance_ratio"] is None
    assert (
        "Target column contains only one observed class and cannot support binary "
        "classification." in profile["ml_warnings"]
    )


def test_profile_excludes_missing_target_values_from_class_summary() -> None:
    dataframe = make_profile_dataframe()
    dataframe.loc[0, "Machine failure"] = pd.NA

    profile = profile_dataset(dataframe)

    assert profile["target_summary"]["class_counts"] == {"0": 1, "1": 1}
    assert profile["target_summary"]["class_percentages"] == {"0": 50.0, "1": 50.0}
    assert profile["target_summary"]["imbalance_ratio"] == 1.0
    assert "Target column contains 1 missing values." in profile["ml_warnings"]
    assert "<NA>" not in profile["target_summary"]["class_counts"]


def test_profile_reports_all_missing_target_values() -> None:
    dataframe = make_profile_dataframe()
    dataframe["Machine failure"] = pd.NA

    profile = profile_dataset(dataframe)

    assert profile["target_summary"]["class_counts"] == {}
    assert profile["target_summary"]["class_percentages"] == {}
    assert profile["target_summary"]["imbalance_ratio"] is None
    assert "Target column contains 3 missing values." in profile["ml_warnings"]
    assert (
        "Target column contains no observed classes and cannot support binary "
        "classification." in profile["ml_warnings"]
    )


def test_profile_reports_unexpected_categories() -> None:
    dataframe = make_profile_dataframe()
    dataframe.loc[0, "Type"] = "X"

    profile = profile_dataset(dataframe)

    assert any(
        "Unexpected categories detected: Type: X" in warning
        for warning in profile["ml_warnings"]
    )


def test_profile_reports_leakage_columns() -> None:
    profile = profile_dataset(make_profile_dataframe())

    assert profile["schema_summary"]["leakage_columns"] == list(
        config.FAILURE_MODE_COLUMNS
    )
    assert not any(
        "Leakage columns present" in warning for warning in profile["ml_warnings"]
    )


def test_numerical_feature_summary_contains_expected_statistics() -> None:
    profile = profile_dataset(make_profile_dataframe())
    summary = profile["numerical_feature_summary"]["Torque [Nm]"]

    assert set(summary) == {"mean", "median", "std", "min", "max", "q1", "q3"}
    assert summary["min"] == 42.8
    assert summary["max"] == 50.0


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_numerical_outputs_are_converted_to_none(value: float) -> None:
    dataframe = make_profile_dataframe()
    dataframe["Torque [Nm]"] = value

    profile = profile_dataset(dataframe)

    assert profile["numerical_feature_summary"]["Torque [Nm]"] == {
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
    }
    json.dumps(profile, allow_nan=False)


def test_categorical_feature_summary_contains_counts_and_percentages() -> None:
    profile = profile_dataset(make_profile_dataframe())
    summary = profile["categorical_feature_summary"]["Type"]

    assert summary["unique_values"] == ["H", "L", "M"]
    assert summary["counts"] == {"H": 1, "L": 1, "M": 1}
    assert summary["percentages"] == {"H": 33.33, "L": 33.33, "M": 33.33}


def test_profile_is_json_serializable() -> None:
    profile = profile_dataset(make_profile_dataframe())

    encoded = json.dumps(profile)

    assert "dataset_summary" in encoded


def test_save_profile_writes_strict_json_and_summary_csv(tmp_path: Path) -> None:
    profile = profile_dataset(make_profile_dataframe())

    output_paths = save_profile(profile, tmp_path)

    assert output_paths["json"] == tmp_path / "dataset_profile.json"
    assert output_paths["summary_csv"] == tmp_path / "dataset_summary.csv"
    assert output_paths["json"].is_file()
    assert output_paths["summary_csv"].is_file()
    with output_paths["json"].open(encoding="utf-8") as file:
        saved_profile = json.load(file)
    summary_csv = pd.read_csv(output_paths["summary_csv"])
    assert saved_profile["dataset_summary"]["row_count"] == 3
    assert set(summary_csv.columns) == {"metric", "value"}
    assert summary_csv.loc[summary_csv["metric"] == "row_count", "value"].item() == 3


def test_profile_dataset_does_not_mutate_dataframe() -> None:
    dataframe = make_profile_dataframe()
    original = dataframe.copy(deep=True)

    profile_dataset(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
