"""Batch prediction utilities for persisted predictive-maintenance artifacts."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.artifacts import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    load_final_model_artifact,
)
from predictive_maintenance.data import _validate_source_header_unique
from predictive_maintenance.evaluation import positive_class_probability
from predictive_maintenance.exceptions import (
    ArtifactError,
    DataLoadingError,
    DataValidationError,
    DuplicateColumnsError,
    ModelEvaluationError,
    PredictionError,
)
from predictive_maintenance.validation import (
    validate_not_empty,
    validate_unique_columns,
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "failure_probability",
    "predicted_failure",
)


def load_prediction_input(data: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load and validate batch prediction features in configured model order.

    Missing values are passed through to the persisted artifact. The current
    final Pipeline supports NaNs, but compatibility ultimately depends on the
    loaded artifact rather than this input loader.
    """
    if isinstance(data, pd.DataFrame):
        dataframe = data.copy()
    else:
        dataframe = _load_prediction_csv(Path(data))

    _validate_prediction_features(dataframe)
    return dataframe.loc[:, config.MODEL_FEATURES].copy()


def predict_dataset(
    data: Path | str | pd.DataFrame,
    model_path: Path | str,
    metadata_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run batch predictions with a persisted final model artifact."""
    features = load_prediction_input(data)
    artifact = _load_prediction_artifact(model_path, metadata_path)
    return _predict_features(features, artifact)


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: Path | str,
) -> Path:
    """Save prediction results to predictions.csv using UTF-8 encoding.

    DataFrame input indices are preserved in returned predictions. CSV-loaded
    inputs receive pandas' default index unless the caller reconstructs a custom
    index before prediction. Exported predictions include this index as
    input_index for traceability.
    """
    _validate_prediction_output(predictions)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "predictions.csv"
    try:
        predictions.to_csv(
            csv_path,
            encoding="utf-8",
            index=True,
            index_label="input_index",
        )
    except OSError as exc:
        raise PredictionError(f"Unable to save predictions CSV: {csv_path}") from exc
    return csv_path


def run_batch_prediction(
    data: Path | str | pd.DataFrame,
    artifact_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load a persisted artifact, run batch prediction, and optionally save CSV."""
    resolved_model_path, resolved_metadata_path = _resolve_artifact_paths(
        artifact_dir=artifact_dir,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    features = load_prediction_input(data)
    artifact = _load_prediction_artifact(resolved_model_path, resolved_metadata_path)
    predictions = _predict_features(features, artifact)

    result: dict[str, Any] = {
        "prediction_count": int(len(predictions)),
        "threshold_used": _metadata_threshold(artifact["metadata"]),
        "model_version": artifact["metadata"]["project_package_version"],
        "project_package_version": artifact["metadata"]["project_package_version"],
        "artifact_version": artifact["metadata"]["artifact_version"],
        "prediction_dataframe": predictions,
    }
    if output_dir is not None:
        result["prediction_csv_path"] = save_predictions(predictions, output_dir)
    return result


def _predict_features(
    features: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    pipeline = artifact["fitted_pipeline"]
    metadata = artifact["metadata"]
    threshold = _metadata_threshold(metadata)

    try:
        probability_matrix = pipeline.predict_proba(features)
        probabilities = positive_class_probability(pipeline, probability_matrix)
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        ModelEvaluationError,
    ) as exc:
        raise PredictionError(
            "Unable to generate batch prediction probabilities."
        ) from exc

    probabilities = _validate_prediction_probabilities(
        probabilities,
        expected_length=len(features),
    )

    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "predicted_failure": (probabilities >= threshold).astype(int),
        },
        index=features.index.copy(),
        columns=list(PREDICTION_COLUMNS),
    )


def _load_prediction_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PredictionError(f"Prediction input CSV does not exist: {path}")
    if not path.is_file():
        raise PredictionError(f"Prediction input path is not a regular file: {path}")

    try:
        _validate_source_header_unique(path)
        dataframe = pd.read_csv(path)
    except DuplicateColumnsError as exc:
        raise PredictionError(
            f"Prediction input CSV contains duplicate columns: "
            f"{', '.join(exc.duplicate_columns)}."
        ) from exc
    except (DataLoadingError, DataValidationError, pd.errors.EmptyDataError) as exc:
        raise PredictionError(f"Unable to load prediction input CSV: {path}") from exc
    except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        raise PredictionError(f"Unable to load prediction input CSV: {path}") from exc

    return dataframe


def _validate_prediction_features(dataframe: pd.DataFrame) -> None:
    try:
        validate_not_empty(dataframe)
        validate_unique_columns(dataframe)
    except DataValidationError as exc:
        raise PredictionError("Prediction input schema is invalid.") from exc

    expected = config.MODEL_FEATURES
    actual = tuple(dataframe.columns)
    if actual != expected:
        missing_columns = tuple(column for column in expected if column not in actual)
        extra_columns = tuple(column for column in actual if column not in expected)
        if missing_columns:
            raise PredictionError(
                "Prediction input is missing required model feature columns: "
                f"{', '.join(missing_columns)}."
            )
        if extra_columns:
            raise PredictionError(
                "Prediction input contains unexpected model feature columns: "
                f"{', '.join(extra_columns)}."
            )
        raise PredictionError(
            "Prediction input columns must match config.MODEL_FEATURES exactly "
            "and in the same order."
        )


def _load_prediction_artifact(
    model_path: Path | str,
    metadata_path: Path | str | None,
) -> dict[str, Any]:
    try:
        return load_final_model_artifact(model_path, metadata_path)
    except ArtifactError as exc:
        raise PredictionError("Unable to load prediction model artifact.") from exc


def _metadata_threshold(metadata: dict[str, Any]) -> float:
    try:
        threshold = float(metadata["operating_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(
            "Artifact metadata must include an operating threshold."
        ) from exc
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise PredictionError(
            "Artifact operating threshold must be finite and strictly between 0 and 1."
        )
    return threshold


def _validate_prediction_probabilities(
    probabilities: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1:
        raise PredictionError(
            "Prediction probabilities must be a one-dimensional array."
        )
    if len(probability_array) != expected_length:
        raise PredictionError("Prediction probabilities must match input row count.")
    if not np.isfinite(probability_array).all():
        raise PredictionError("Prediction probabilities must be finite.")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise PredictionError("Prediction probabilities must be within [0, 1].")
    return probability_array


def _resolve_artifact_paths(
    artifact_dir: Path | str | None,
    model_path: Path | str | None,
    metadata_path: Path | str | None,
) -> tuple[Path, Path | None]:
    if artifact_dir is not None and model_path is not None:
        raise PredictionError("Provide either artifact_dir or model_path, not both.")
    if artifact_dir is not None:
        directory = Path(artifact_dir)
        return directory / MODEL_FILENAME, directory / METADATA_FILENAME
    if model_path is None:
        directory = config.MODELS_DIR / "final"
        return directory / MODEL_FILENAME, directory / METADATA_FILENAME
    return Path(model_path), Path(metadata_path) if metadata_path is not None else None


def _validate_prediction_output(predictions: pd.DataFrame) -> None:
    try:
        validate_not_empty(predictions)
        validate_unique_columns(predictions)
    except DataValidationError as exc:
        raise PredictionError("Prediction output is invalid.") from exc
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PredictionError(
            "Prediction output columns do not match the public schema."
        )
