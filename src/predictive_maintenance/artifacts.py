"""Final model artifact persistence and metadata validation.

Joblib artifacts use Python pickle under the hood and must only be loaded from
trusted sources. Loading untrusted pickle/joblib files can execute code. The
artifact produced here is intended for this repository environment; dependency
versions are recorded for reproducibility, not as a secure deserialization
guarantee.
"""

import hashlib
import json
import math
import platform
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from predictive_maintenance import __version__, config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.exceptions import ArtifactError, FinalEvaluationError
from predictive_maintenance.final_evaluation import (
    FINAL_MODEL_NAME,
    FINAL_OPERATING_PROFILE,
    _build_final_model,
    _combine_train_validation,
    _selected_balanced_threshold,
    _validate_final_split,
)
from predictive_maintenance.recommendations import run_business_threshold_recommendation
from predictive_maintenance.splitting import split_dataset
from predictive_maintenance.train import _fit_model

ARTIFACT_VERSION = "1.0.0"
MODEL_FILENAME = "predictive_maintenance_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "artifact_version",
    "model_name",
    "operating_profile",
    "operating_threshold",
    "target_column",
    "model_features",
    "numerical_features",
    "categorical_features",
    "excluded_identifier_columns",
    "excluded_leakage_columns",
    "random_seed",
    "development_row_count",
    "development_class_distribution",
    "training_timestamp_utc",
    "python_version",
    "pandas_version",
    "scikit_learn_version",
    "project_package_version",
    "final_evaluation_reference",
    "synthetic_dataset_limitation",
    "intended_use",
    "non_intended_use",
    "governance_statement",
    "model_sha256",
)


def build_final_model_artifact(
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit the frozen final pipeline on development data and return metadata.

    The deterministic final test split is validated for separation but is not
    used for fitting, sample weighting, threshold selection, prediction, or
    scoring in this workflow.
    """
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    recommendations = run_business_threshold_recommendation(
        dataset,
        model_name=FINAL_MODEL_NAME,
    )
    operating_threshold = _artifact_threshold(recommendations)

    split_data = split_dataset(dataset)
    X_development, y_development = _combine_train_validation(split_data)
    _validate_artifact_split(X_development, y_development, split_data)

    pipeline = _build_final_model()
    try:
        _fit_model(FINAL_MODEL_NAME, pipeline, X_development, y_development)
    except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        raise ArtifactError("Unable to fit the final model artifact.") from exc

    return {
        "fitted_pipeline": pipeline,
        "metadata": _build_metadata(
            operating_threshold=operating_threshold,
            development_row_count=len(X_development),
            development_class_distribution=_class_distribution(y_development),
            model_sha256=None,
        ),
    }


def save_final_model_artifact(
    artifact: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save the fitted final pipeline and strict JSON metadata."""
    pipeline = _artifact_pipeline(artifact)
    metadata = _artifact_metadata(artifact)
    output_path = (
        Path(output_dir) if output_dir is not None else config.MODELS_DIR / "final"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / MODEL_FILENAME
    metadata_path = output_path / METADATA_FILENAME

    try:
        joblib.dump(pipeline, model_path)
    except Exception as exc:
        raise ArtifactError("Unable to save fitted model artifact.") from exc

    metadata_to_save = dict(metadata)
    metadata_to_save["model_sha256"] = _sha256(model_path)
    _validate_metadata(metadata_to_save)

    try:
        metadata_path.write_text(
            json.dumps(metadata_to_save, allow_nan=False, indent=2),
            encoding="utf-8",
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ArtifactError("Unable to save model metadata JSON.") from exc

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_final_model_artifact(
    model_path: Path | str,
    metadata_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate a trusted final model artifact."""
    resolved_model_path = Path(model_path)
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else resolved_model_path.with_name(METADATA_FILENAME)
    )

    if not resolved_model_path.is_file():
        raise ArtifactError(f"Model artifact file not found: {resolved_model_path}")
    if not resolved_metadata_path.is_file():
        raise ArtifactError(f"Metadata file not found: {resolved_metadata_path}")

    metadata = _load_metadata(resolved_metadata_path)
    _validate_metadata(metadata)
    actual_checksum = _sha256(resolved_model_path)
    if actual_checksum != metadata["model_sha256"]:
        raise ArtifactError("Model artifact checksum does not match metadata.")

    try:
        pipeline = joblib.load(resolved_model_path)
    except Exception as exc:
        raise ArtifactError("Unable to load model artifact with joblib.") from exc
    _validate_loaded_pipeline(pipeline)

    return {"fitted_pipeline": pipeline, "metadata": metadata}


def _artifact_threshold(recommendations: dict[str, Any]) -> float:
    try:
        return _selected_balanced_threshold(recommendations)
    except FinalEvaluationError as exc:
        raise ArtifactError(
            "Unable to read the validation-derived balanced threshold."
        ) from exc


def _validate_artifact_split(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    split_data: Any,
) -> None:
    try:
        _validate_final_split(X_development, y_development, split_data)
    except FinalEvaluationError as exc:
        raise ArtifactError(
            "Final artifact development/test split is invalid."
        ) from exc


def _build_metadata(
    operating_threshold: float,
    development_row_count: int,
    development_class_distribution: dict[str, int],
    model_sha256: str | None,
) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": FINAL_MODEL_NAME,
        "operating_profile": FINAL_OPERATING_PROFILE,
        "operating_threshold": float(operating_threshold),
        "target_column": config.TARGET_COLUMN,
        "model_features": list(config.MODEL_FEATURES),
        "numerical_features": list(config.NUMERICAL_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "excluded_identifier_columns": list(config.IDENTIFIER_COLUMNS),
        "excluded_leakage_columns": list(config.FAILURE_MODE_COLUMNS),
        "random_seed": int(config.RANDOM_SEED),
        "development_row_count": int(development_row_count),
        "development_class_distribution": dict(development_class_distribution),
        "training_timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "project_package_version": __version__,
        "final_evaluation_reference": (
            "docs/results_summary.md documents final held-out test evidence; "
            "test metrics are not used as model-selection inputs."
        ),
        "synthetic_dataset_limitation": (
            "Trained on the synthetic UCI AI4I 2020 dataset; not production-certified "
            "for real industrial equipment."
        ),
        "intended_use": (
            "Batch machine-failure risk scoring for portfolio demonstration and "
            "maintenance-prioritization analysis."
        ),
        "non_intended_use": (
            "Autonomous maintenance decisions, safety-critical control, or production "
            "deployment without real operational validation."
        ),
        "governance_statement": (
            "Model and threshold were selected from validation workflows before final "
            "test evaluation. This artifact is fit on train plus validation data only; "
            "the held-out test split is excluded from fitting and sample weighting."
        ),
        "model_sha256": model_sha256,
    }


def _artifact_pipeline(artifact: dict[str, Any]) -> Pipeline:
    pipeline = artifact.get("fitted_pipeline")
    _validate_loaded_pipeline(pipeline)
    return pipeline


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactError("Artifact must contain metadata as a dictionary.")
    return metadata


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError("Model metadata JSON is malformed.") from exc
    except OSError as exc:
        raise ArtifactError("Unable to read model metadata JSON.") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("Model metadata JSON must contain an object.")
    return metadata


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing_fields = [
        field for field in REQUIRED_METADATA_FIELDS if field not in metadata
    ]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ArtifactError(f"Model metadata is missing required fields: {fields}.")

    if not re.fullmatch(r"\d+\.\d+\.\d+", str(metadata["artifact_version"])):
        raise ArtifactError("Artifact version must use semantic version format X.Y.Z.")
    if metadata["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactError(
            f"Unsupported artifact version: {metadata['artifact_version']}."
        )
    if metadata["model_name"] != FINAL_MODEL_NAME:
        raise ArtifactError("Model metadata references an unexpected model name.")
    if metadata["operating_profile"] != FINAL_OPERATING_PROFILE:
        raise ArtifactError(
            "Model metadata references an unexpected operating profile."
        )
    if metadata["target_column"] != config.TARGET_COLUMN:
        raise ArtifactError("Model metadata target column is incompatible.")
    _validate_metadata_sequence(
        metadata,
        "model_features",
        config.MODEL_FEATURES,
        "Model metadata features do not match project config.",
    )
    _validate_metadata_sequence(
        metadata,
        "numerical_features",
        config.NUMERICAL_FEATURES,
        "Model metadata numerical features are incompatible.",
    )
    _validate_metadata_sequence(
        metadata,
        "categorical_features",
        config.CATEGORICAL_FEATURES,
        "Model metadata categorical features are incompatible.",
    )
    _validate_metadata_sequence(
        metadata,
        "excluded_identifier_columns",
        config.IDENTIFIER_COLUMNS,
        "Model metadata identifier exclusions are incompatible.",
    )
    _validate_metadata_sequence(
        metadata,
        "excluded_leakage_columns",
        config.FAILURE_MODE_COLUMNS,
        "Model metadata leakage-column exclusions are incompatible.",
    )
    threshold = metadata["operating_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or threshold <= 0.0
        or threshold >= 1.0
    ):
        raise ArtifactError("Operating threshold must be finite and between 0 and 1.")
    if metadata["random_seed"] != config.RANDOM_SEED:
        raise ArtifactError("Model metadata random seed is incompatible.")
    _validate_development_summary(metadata)
    _validate_utc_timestamp(metadata["training_timestamp_utc"])
    for field in ("python_version", "pandas_version", "scikit_learn_version"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ArtifactError(f"Model metadata {field} must be a non-empty string.")
    if metadata["project_package_version"] != __version__:
        raise ArtifactError("Model metadata package version is incompatible.")
    checksum = metadata["model_sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ArtifactError("Model SHA-256 checksum is invalid.")


def _validate_loaded_pipeline(pipeline: Any) -> None:
    if not isinstance(pipeline, Pipeline):
        raise ArtifactError("Loaded model artifact must be a scikit-learn Pipeline.")
    if tuple(pipeline.named_steps) != ("preprocessor", "classifier"):
        raise ArtifactError(
            "Loaded model artifact must contain preprocessor and classifier steps."
        )
    predict_proba = getattr(pipeline, "predict_proba", None)
    if not callable(predict_proba):
        raise ArtifactError("Loaded model artifact must expose callable predict_proba.")
    try:
        check_is_fitted(pipeline)
    except NotFittedError as exc:
        raise ArtifactError("Loaded model artifact must be fitted.") from exc

    feature_names = getattr(pipeline, "feature_names_in_", None)
    if feature_names is None or tuple(feature_names) != config.MODEL_FEATURES:
        raise ArtifactError(
            "Loaded model artifact input features do not match project config."
        )

    classifier = pipeline.named_steps["classifier"]
    classes = getattr(classifier, "classes_", None)
    if classes is None or 1 not in set(classes):
        raise ArtifactError("Loaded model artifact classifier must include class 1.")


def _validate_metadata_sequence(
    metadata: dict[str, Any],
    field: str,
    expected: tuple[str, ...],
    message: str,
) -> None:
    value = metadata[field]
    if not isinstance(value, list) or tuple(value) != expected:
        raise ArtifactError(message)


def _validate_development_summary(metadata: dict[str, Any]) -> None:
    row_count = metadata["development_row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ArtifactError("Model metadata development row count is invalid.")

    class_distribution = metadata["development_class_distribution"]
    if not isinstance(class_distribution, dict) or set(class_distribution) != {
        "0",
        "1",
    }:
        raise ArtifactError(
            "Model metadata development class distribution must contain classes "
            "0 and 1."
        )
    counts = class_distribution.values()
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ):
        raise ArtifactError(
            "Model metadata development class distribution counts are invalid."
        )
    if sum(class_distribution.values()) != row_count:
        raise ArtifactError(
            "Model metadata development class distribution must sum to row count."
        )


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise ArtifactError("Model metadata training timestamp must be a string.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactError(
            "Model metadata training timestamp is not valid ISO 8601."
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ArtifactError("Model metadata training timestamp must be UTC-aware.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"Unable to read artifact for checksum: {path}") from exc
    return digest.hexdigest()


def _class_distribution(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}
