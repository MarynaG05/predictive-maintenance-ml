from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance import config, data
from predictive_maintenance.exceptions import (
    DataLoadingError,
    DuplicateColumnsError,
    EmptyDatasetError,
    MissingColumnsError,
)


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
            },
            {
                "UDI": 2,
                "Product ID": "L47181",
                "Type": "L",
                "Air temperature [K]": 298.2,
                "Process temperature [K]": 308.7,
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
        ]
    )


def write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    dataframe.to_csv(path, index=False)


def csv_row(values: tuple[object, ...]) -> str:
    return ",".join(str(value) for value in values)


def test_default_path_is_derived_from_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    dataset_path = raw_data_dir / config.DATASET_FILENAME
    expected = make_valid_dataframe()
    write_csv(dataset_path, expected)

    monkeypatch.setattr(data.config, "RAW_DATA_DIR", raw_data_dir)

    loaded = data.load_dataset()

    pd.testing.assert_frame_equal(loaded, expected.loc[:, config.REQUIRED_COLUMNS])


def test_explicit_path_input_works(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    expected = make_valid_dataframe()
    write_csv(dataset_path, expected)

    loaded = data.load_dataset(dataset_path)

    pd.testing.assert_frame_equal(loaded, expected.loc[:, config.REQUIRED_COLUMNS])


def test_explicit_string_input_works(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    expected = make_valid_dataframe()
    write_csv(dataset_path, expected)

    loaded = data.load_dataset(str(dataset_path))

    pd.testing.assert_frame_equal(loaded, expected.loc[:, config.REQUIRED_COLUMNS])


def test_missing_file_raises_data_loading_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(DataLoadingError, match="does not exist"):
        data.load_dataset(missing_path)


def test_directory_path_raises_data_loading_error(tmp_path: Path) -> None:
    with pytest.raises(DataLoadingError, match="not a regular file"):
        data.load_dataset(tmp_path)


def test_unreadable_csv_produces_data_loading_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    dataset_path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(DataLoadingError) as exc_info:
        data.load_dataset(dataset_path)

    assert exc_info.value.__cause__ is not None


def test_malformed_csv_parser_error_produces_data_loading_error(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "malformed.csv"
    dataset_path.write_text(
        f"{csv_row(config.REQUIRED_COLUMNS)}\n"
        '1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,"0\n'
    )

    with pytest.raises(DataLoadingError) as exc_info:
        data.load_dataset(dataset_path)

    assert isinstance(exc_info.value.__cause__, pd.errors.ParserError)


def test_empty_csv_produces_validation_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.csv"
    dataset_path.write_text("")

    with pytest.raises(EmptyDatasetError) as exc_info:
        data.load_dataset(dataset_path)

    assert isinstance(exc_info.value.__cause__, pd.errors.EmptyDataError)


def test_header_only_csv_produces_validation_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "header_only.csv"
    dataset_path.write_text(f"{csv_row(config.REQUIRED_COLUMNS)}\n")

    with pytest.raises(EmptyDatasetError) as exc_info:
        data.load_dataset(dataset_path)

    assert exc_info.value.__cause__ is None


def test_duplicate_source_headers_raise_duplicate_columns_error(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "duplicate_headers.csv"
    duplicated_header = (
        "UDI",
        "Product ID",
        "Type",
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
    dataset_path.write_text(
        f"{csv_row(duplicated_header)}\n"
        "1,M14860,M,L,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n"
    )

    with pytest.raises(DuplicateColumnsError, match="Type") as exc_info:
        data.load_dataset(dataset_path)

    assert exc_info.value.duplicate_columns == ("Type",)


def test_duplicate_source_header_names_preserve_first_duplicate_order(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "duplicate_headers.csv"
    duplicated_header = (
        "UDI",
        "Product ID",
        "Type",
        "UDI",
        "Air temperature [K]",
        "Type",
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
    dataset_path.write_text(
        f"{csv_row(duplicated_header)}\n"
        "1,M14860,M,2,298.1,L,308.6,1551,42.8,0,0,0,0,0,0,0\n"
    )

    with pytest.raises(DuplicateColumnsError) as exc_info:
        data.load_dataset(dataset_path)

    assert exc_info.value.duplicate_columns == ("UDI", "Type")


def test_quoted_csv_headers_are_parsed_correctly(tmp_path: Path) -> None:
    dataset_path = tmp_path / "quoted_headers.csv"
    expected = make_valid_dataframe()
    quoted_header = ",".join(f'"{column}"' for column in config.REQUIRED_COLUMNS)
    dataset_path.write_text(
        f"{quoted_header}\n"
        "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n"
        "2,L47181,L,298.2,308.7,1408,46.3,3,0,0,0,0,0,0\n"
    )

    loaded = data.load_dataset(dataset_path)

    pd.testing.assert_frame_equal(loaded, expected.loc[:, config.REQUIRED_COLUMNS])


def test_missing_columns_propagate_as_validation_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing_columns.csv"
    dataframe = make_valid_dataframe().drop(columns=["Machine failure"])
    write_csv(dataset_path, dataframe)

    with pytest.raises(MissingColumnsError, match="Machine failure"):
        data.load_dataset(dataset_path)


def test_returned_columns_match_required_columns_in_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataframe = make_valid_dataframe()
    write_csv(dataset_path, dataframe)

    loaded = data.load_dataset(dataset_path)

    assert tuple(loaded.columns) == config.REQUIRED_COLUMNS


def test_extra_columns_are_not_included_in_returned_dataframe(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataframe = make_valid_dataframe().assign(extra_sensor=123)
    write_csv(dataset_path, dataframe)

    loaded = data.load_dataset(dataset_path)

    assert "extra_sensor" not in loaded.columns
    assert tuple(loaded.columns) == config.REQUIRED_COLUMNS


def test_returned_dataframe_is_independent_copy(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataframe = make_valid_dataframe()
    write_csv(dataset_path, dataframe)

    loaded = data.load_dataset(dataset_path)
    loaded.loc[0, "Type"] = "changed"
    reloaded = data.load_dataset(dataset_path)

    assert reloaded.loc[0, "Type"] == "M"
