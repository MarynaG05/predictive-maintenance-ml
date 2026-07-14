import pandas as pd
import pytest

from predictive_maintenance import config
from predictive_maintenance.exceptions import (
    DuplicateColumnsError,
    EmptyDatasetError,
    MissingColumnsError,
)
from predictive_maintenance.validation import validate_dataframe_structure


def make_valid_dataframe() -> pd.DataFrame:
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
            }
        ]
    )


def test_valid_dataframe_passes() -> None:
    validate_dataframe_structure(make_valid_dataframe())


def test_zero_row_dataframe_fails() -> None:
    dataframe = pd.DataFrame(columns=config.REQUIRED_COLUMNS)

    with pytest.raises(EmptyDatasetError, match="at least one row"):
        validate_dataframe_structure(dataframe)


def test_zero_column_dataframe_fails() -> None:
    dataframe = pd.DataFrame(index=[0])

    with pytest.raises(EmptyDatasetError, match="at least one column"):
        validate_dataframe_structure(dataframe)


def test_missing_required_column_fails() -> None:
    dataframe = make_valid_dataframe().drop(columns=["Torque [Nm]"])

    with pytest.raises(MissingColumnsError, match="Torque"):
        validate_dataframe_structure(dataframe)


def test_multiple_missing_required_columns_are_reported() -> None:
    dataframe = make_valid_dataframe().drop(columns=["Torque [Nm]", "TWF"])

    with pytest.raises(MissingColumnsError) as exc_info:
        validate_dataframe_structure(dataframe)

    assert exc_info.value.missing_columns == ("Torque [Nm]", "TWF")
    assert "Torque [Nm]" in str(exc_info.value)
    assert "TWF" in str(exc_info.value)


def test_duplicate_column_names_fail() -> None:
    columns = (*config.REQUIRED_COLUMNS, "UDI")
    values = [[*range(len(config.REQUIRED_COLUMNS)), 999]]
    dataframe = pd.DataFrame(values, columns=columns)

    with pytest.raises(DuplicateColumnsError, match="UDI") as exc_info:
        validate_dataframe_structure(dataframe)

    assert exc_info.value.duplicate_columns == ("UDI",)


def test_extra_columns_are_allowed() -> None:
    dataframe = make_valid_dataframe().assign(extra_sensor=123)

    validate_dataframe_structure(dataframe)


def test_input_dataframe_is_not_mutated() -> None:
    dataframe = make_valid_dataframe().assign(extra_sensor=123)
    original = dataframe.copy(deep=True)
    original_columns = tuple(dataframe.columns)

    validate_dataframe_structure(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
    assert tuple(dataframe.columns) == original_columns
