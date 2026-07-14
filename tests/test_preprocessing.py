import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from predictive_maintenance import config
from predictive_maintenance.preprocessing import build_preprocessor


def make_feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.1, 299.2, 300.3, 301.4],
            "Process temperature [K]": [308.1, 309.2, 310.3, 311.4],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1700],
            "Torque [Nm]": [40.0, 42.0, 44.0, 46.0],
            "Tool wear [min]": [0, 10, 20, 30],
        },
        columns=config.MODEL_FEATURES,
    )


def test_build_preprocessor_returns_column_transformer() -> None:
    assert isinstance(build_preprocessor(), ColumnTransformer)


def test_preprocessor_contains_expected_feature_groups() -> None:
    preprocessor = build_preprocessor()

    transformers = {name: columns for name, _, columns in preprocessor.transformers}

    assert transformers["numerical"] == list(config.NUMERICAL_FEATURES)
    assert transformers["categorical"] == list(config.CATEGORICAL_FEATURES)


def test_preprocessor_is_unfitted_when_returned() -> None:
    preprocessor = build_preprocessor()

    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)


def test_preprocessor_handles_unknown_type_after_fitting() -> None:
    preprocessor = build_preprocessor()
    features = make_feature_dataframe()

    preprocessor.fit(features)
    transformed = preprocessor.transform(
        pd.DataFrame(
            {
                "Type": ["Z"],
                "Air temperature [K]": [302.0],
                "Process temperature [K]": [312.0],
                "Rotational speed [rpm]": [1800],
                "Torque [Nm]": [48.0],
                "Tool wear [min]": [40],
            },
            columns=config.MODEL_FEATURES,
        )
    )

    assert transformed.shape[0] == 1


def test_preprocessor_transformed_output_has_expected_number_of_rows() -> None:
    preprocessor = build_preprocessor()
    features = make_feature_dataframe()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)


def test_preprocessor_does_not_require_non_model_columns() -> None:
    preprocessor = build_preprocessor()
    features = make_feature_dataframe()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
