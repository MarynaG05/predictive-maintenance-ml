"""Central project configuration for predictive maintenance workflows.

This module defines repository paths, AI4I 2020 dataset schema constants, and
reproducible split settings. It intentionally contains no data loading,
preprocessing, training, or prediction logic.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"

DATASET_FILENAME: str = "ai4i2020.csv"
TARGET_COLUMN: str = "Machine failure"

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "UDI",
    "Product ID",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("Type",)
EXPECTED_CATEGORICAL_VALUES: dict[str, tuple[str, ...]] = {
    "Type": ("H", "L", "M"),
}

NUMERICAL_FEATURES: tuple[str, ...] = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

FAILURE_MODE_COLUMNS: tuple[str, ...] = (
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)

MODEL_FEATURES: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

REQUIRED_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *MODEL_FEATURES,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
)

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

# VALIDATION_SIZE is interpreted relative to the remaining training data after
# the test split. With TEST_SIZE = 0.20, this yields an overall 60/20/20
# train-validation-test split: 0.20 / (1.0 - 0.20) = 0.25.
VALIDATION_SIZE: float = 0.25
