import pandas as pd
import pytest

from predictive_maintenance import config
from predictive_maintenance.exceptions import DatasetSplitError
from predictive_maintenance.splitting import split_dataset


def make_synthetic_dataframe(
    majority_count: int = 80,
    minority_count: int = 20,
) -> pd.DataFrame:
    rows = []
    labels = [0] * majority_count + [1] * minority_count
    type_values = ("L", "M", "H")

    for index, target in enumerate(labels, start=1):
        rows.append(
            {
                "UDI": index,
                "Product ID": f"ID{index:05d}",
                "Type": type_values[index % len(type_values)],
                "Air temperature [K]": 298.0 + index * 0.01,
                "Process temperature [K]": 308.0 + index * 0.01,
                "Rotational speed [rpm]": 1400 + index,
                "Torque [Nm]": 40.0 + target * 5.0 + index * 0.01,
                "Tool wear [min]": index % 200,
                "Machine failure": target,
                "TWF": target,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )

    return pd.DataFrame(rows, columns=config.REQUIRED_COLUMNS)


def test_split_dataset_produces_overall_60_20_20_proportions() -> None:
    dataframe = make_synthetic_dataframe()

    split_data = split_dataset(dataframe)

    assert len(split_data.X_train) == 60
    assert len(split_data.X_validation) == 20
    assert len(split_data.X_test) == 20


def test_split_dataset_accepts_valid_binary_target_labels() -> None:
    split_data = split_dataset(make_synthetic_dataframe())

    assert set(split_data.y_train.unique()) == {0, 1}


def test_split_dataset_preserves_target_stratification() -> None:
    split_data = split_dataset(make_synthetic_dataframe())

    assert split_data.y_train.value_counts().sort_index().to_dict() == {0: 48, 1: 12}
    assert split_data.y_validation.value_counts().sort_index().to_dict() == {
        0: 16,
        1: 4,
    }
    assert split_data.y_test.value_counts().sort_index().to_dict() == {0: 16, 1: 4}


def test_split_dataset_uses_exact_model_features() -> None:
    split_data = split_dataset(make_synthetic_dataframe())

    assert tuple(split_data.X_train.columns) == config.MODEL_FEATURES


def test_split_dataset_excludes_identifier_and_failure_mode_columns() -> None:
    split_data = split_dataset(make_synthetic_dataframe())

    excluded_columns = set(config.IDENTIFIER_COLUMNS + config.FAILURE_MODE_COLUMNS)
    assert excluded_columns.isdisjoint(split_data.X_train.columns)
    assert config.TARGET_COLUMN not in split_data.X_train.columns


def test_split_dataset_does_not_mutate_source_dataframe() -> None:
    dataframe = make_synthetic_dataframe()
    original = dataframe.copy(deep=True)

    split_dataset(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_split_dataset_preserves_X_y_index_alignment() -> None:
    split_data = split_dataset(make_synthetic_dataframe())

    assert split_data.X_train.index.equals(split_data.y_train.index)
    assert split_data.X_validation.index.equals(split_data.y_validation.index)
    assert split_data.X_test.index.equals(split_data.y_test.index)


def test_split_dataset_outputs_pairwise_disjoint_indices() -> None:
    dataframe = make_synthetic_dataframe()

    split_data = split_dataset(dataframe)

    train_indices = set(split_data.X_train.index)
    validation_indices = set(split_data.X_validation.index)
    test_indices = set(split_data.X_test.index)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)
    assert train_indices | validation_indices | test_indices == set(dataframe.index)


def test_split_dataset_is_reproducible() -> None:
    dataframe = make_synthetic_dataframe()

    first_split = split_dataset(dataframe)
    second_split = split_dataset(dataframe)

    pd.testing.assert_frame_equal(first_split.X_train, second_split.X_train)
    pd.testing.assert_series_equal(first_split.y_train, second_split.y_train)
    pd.testing.assert_frame_equal(
        first_split.X_validation,
        second_split.X_validation,
    )
    pd.testing.assert_series_equal(
        first_split.y_validation,
        second_split.y_validation,
    )
    pd.testing.assert_frame_equal(first_split.X_test, second_split.X_test)
    pd.testing.assert_series_equal(first_split.y_test, second_split.y_test)


def test_split_dataset_fails_for_one_class_target() -> None:
    dataframe = make_synthetic_dataframe(majority_count=100, minority_count=0)

    with pytest.raises(DatasetSplitError, match="both classes 0 and 1"):
        split_dataset(dataframe)


def test_split_dataset_fails_for_missing_target_values() -> None:
    dataframe = make_synthetic_dataframe()
    dataframe.loc[0, config.TARGET_COLUMN] = pd.NA

    with pytest.raises(DatasetSplitError, match="missing values"):
        split_dataset(dataframe)


def test_split_dataset_fails_for_unexpected_numeric_labels() -> None:
    dataframe = make_synthetic_dataframe()
    dataframe[config.TARGET_COLUMN] = dataframe[config.TARGET_COLUMN].map({0: 0, 1: 2})

    with pytest.raises(DatasetSplitError, match="exactly \\{0, 1\\}"):
        split_dataset(dataframe)


def test_split_dataset_fails_for_string_labels() -> None:
    dataframe = make_synthetic_dataframe()
    dataframe[config.TARGET_COLUMN] = dataframe[config.TARGET_COLUMN].map(
        {0: "no", 1: "yes"}
    )

    with pytest.raises(DatasetSplitError, match="exactly \\{0, 1\\}"):
        split_dataset(dataframe)


def test_split_dataset_fails_for_insufficient_minority_samples() -> None:
    dataframe = make_synthetic_dataframe(majority_count=8, minority_count=2)

    with pytest.raises(DatasetSplitError, match="at least 3 rows"):
        split_dataset(dataframe)
