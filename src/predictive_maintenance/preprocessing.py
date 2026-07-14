"""Preprocessing factory for baseline machine learning pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance import config


def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted preprocessor for configured predictive features."""
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), list(config.NUMERICAL_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(config.CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
