import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from predictive_maintenance import __version__, artifacts, config
from predictive_maintenance.artifacts import (
    ARTIFACT_VERSION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    build_final_model_artifact,
    load_final_model_artifact,
    save_final_model_artifact,
)
from predictive_maintenance.exceptions import ArtifactError
from predictive_maintenance.final_evaluation import (
    FINAL_MODEL_NAME,
    FINAL_OPERATING_PROFILE,
)
from predictive_maintenance.preprocessing import build_preprocessor
from predictive_maintenance.splitting import SplitData


class _ControlledFitModel:
    def fit(self, X, y, **fit_parameters):
        return self


def make_artifact_dataframe() -> pd.DataFrame:
    rows = []
    labels = [0] * 12 + [1] * 6
    for index, target in enumerate(labels, start=1):
        rows.append(
            {
                "UDI": index,
                "Product ID": f"ID{index:05d}",
                "Type": ("L", "M", "H")[index % 3],
                "Air temperature [K]": 298.0 + index,
                "Process temperature [K]": 308.0 + index,
                "Rotational speed [rpm]": 1400 + index,
                "Torque [Nm]": 40.0 + target,
                "Tool wear [min]": index,
                "Machine failure": target,
                "TWF": target,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )
    return pd.DataFrame(rows, columns=config.REQUIRED_COLUMNS)


def make_split_data() -> SplitData:
    columns = list(config.MODEL_FEATURES)
    X_train = pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298, 299, 300, 301],
            "Process temperature [K]": [308, 309, 310, 311],
            "Rotational speed [rpm]": [1400, 1410, 1420, 1430],
            "Torque [Nm]": [40, 41, 42, 43],
            "Tool wear [min]": [1, 2, 3, 4],
        },
        index=[10, 11, 12, 13],
        columns=columns,
    )
    X_validation = pd.DataFrame(
        {
            "Type": ["M", "H"],
            "Air temperature [K]": [302, 303],
            "Process temperature [K]": [312, 313],
            "Rotational speed [rpm]": [1440, 1450],
            "Torque [Nm]": [44, 45],
            "Tool wear [min]": [5, 6],
        },
        index=[20, 21],
        columns=columns,
    )
    X_test = pd.DataFrame(
        {
            "Type": ["L", "H"],
            "Air temperature [K]": [304, 305],
            "Process temperature [K]": [314, 315],
            "Rotational speed [rpm]": [1460, 1470],
            "Torque [Nm]": [46, 47],
            "Tool wear [min]": [7, 8],
        },
        index=[30, 31],
        columns=columns,
    )
    return SplitData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=pd.Series([0, 1, 0, 1], index=X_train.index),
        y_validation=pd.Series([0, 1], index=X_validation.index),
        y_test=pd.Series([0, 1], index=X_test.index),
    )


def make_recommendations(threshold: float = 0.8) -> dict[str, Any]:
    return {
        "profiles": {
            FINAL_OPERATING_PROFILE: {
                "threshold": threshold,
            },
        },
    }


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.0, 299.0, 300.0, 301.0],
            "Process temperature [K]": [308.0, 309.0, 310.0, 311.0],
            "Rotational speed [rpm]": [1400, 1410, 1420, 1430],
            "Torque [Nm]": [40.0, 41.0, 42.0, 43.0],
            "Tool wear [min]": [1, 2, 3, 4],
        },
        columns=list(config.MODEL_FEATURES),
    )


def make_fitted_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(make_feature_frame(), pd.Series([0, 1, 0, 1]))
    return pipeline


def make_metadata(model_sha256: str | None = None) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": FINAL_MODEL_NAME,
        "operating_profile": FINAL_OPERATING_PROFILE,
        "operating_threshold": 0.8,
        "target_column": config.TARGET_COLUMN,
        "model_features": list(config.MODEL_FEATURES),
        "numerical_features": list(config.NUMERICAL_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "excluded_identifier_columns": list(config.IDENTIFIER_COLUMNS),
        "excluded_leakage_columns": list(config.FAILURE_MODE_COLUMNS),
        "random_seed": config.RANDOM_SEED,
        "development_row_count": 6,
        "development_class_distribution": {"0": 3, "1": 3},
        "training_timestamp_utc": "2026-07-15T00:00:00+00:00",
        "python_version": "3.12.3",
        "pandas_version": pd.__version__,
        "scikit_learn_version": "1.0.0",
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
            "Batch machine-failure risk scoring for portfolio demonstration."
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


def test_build_final_model_artifact_uses_development_data_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    observed: dict[str, Any] = {}

    class ControlledModel:
        def fit(self, X, y, **fit_parameters):
            observed["fit_index"] = tuple(X.index)
            observed["fit_y_index"] = tuple(y.index)
            observed["sample_weight"] = tuple(
                fit_parameters["classifier__sample_weight"]
            )
            return self

    monkeypatch.setattr(
        artifacts,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(artifacts, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(artifacts, "_build_final_model", lambda: ControlledModel())

    source = make_artifact_dataframe()
    original = source.copy(deep=True)
    artifact = build_final_model_artifact(source)

    assert set(artifact) == {"fitted_pipeline", "metadata"}
    assert artifact["metadata"]["model_name"] == FINAL_MODEL_NAME
    assert artifact["metadata"]["operating_profile"] == FINAL_OPERATING_PROFILE
    assert artifact["metadata"]["operating_threshold"] == 0.8
    assert artifact["metadata"]["development_row_count"] == 6
    assert artifact["metadata"]["development_class_distribution"] == {"0": 3, "1": 3}
    assert observed["fit_index"] == tuple(
        split_data.X_train.index.append(split_data.X_validation.index)
    )
    assert observed["fit_y_index"] == tuple(
        split_data.y_train.index.append(split_data.y_validation.index)
    )
    assert set(observed["fit_index"]).isdisjoint(set(split_data.X_test.index))
    assert len(observed["sample_weight"]) == len(split_data.y_train) + len(
        split_data.y_validation
    )
    pd.testing.assert_frame_equal(source, original)


def test_build_final_model_artifact_uses_balanced_recommendation_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.75),
    )
    monkeypatch.setattr(artifacts, "split_dataset", lambda dataframe: make_split_data())
    monkeypatch.setattr(artifacts, "_build_final_model", lambda: _ControlledFitModel())

    artifact = build_final_model_artifact(make_artifact_dataframe())

    assert artifact["metadata"]["operating_threshold"] == 0.75


def test_save_final_model_artifact_writes_joblib_and_metadata(tmp_path: Path) -> None:
    artifact = {
        "fitted_pipeline": make_fitted_pipeline(),
        "metadata": make_metadata(),
    }

    paths = save_final_model_artifact(artifact, tmp_path)

    assert paths == {
        "model_path": tmp_path / MODEL_FILENAME,
        "metadata_path": tmp_path / METADATA_FILENAME,
    }
    assert paths["model_path"].is_file()
    assert paths["metadata_path"].is_file()
    metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
    json.dumps(metadata, allow_nan=False)
    assert metadata["model_sha256"]
    assert len(metadata["model_sha256"]) == 64
    assert "fitted_pipeline" not in metadata
    assert "raw_data" not in json.dumps(metadata)
    assert "raw_labels" not in json.dumps(metadata)
    assert "raw_probabilities" not in json.dumps(metadata)
    assert "/Volumes" not in json.dumps(metadata)
    assert "/Users" not in json.dumps(metadata)


def test_saved_artifact_reloads_and_predicts(tmp_path: Path) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )

    loaded = load_final_model_artifact(paths["model_path"], paths["metadata_path"])

    assert set(loaded) == {"fitted_pipeline", "metadata"}
    assert isinstance(loaded["fitted_pipeline"], Pipeline)
    probabilities = loaded["fitted_pipeline"].predict_proba(
        make_feature_frame().iloc[:2]
    )
    assert probabilities.shape == (2, 2)
    assert (
        loaded["metadata"]["model_sha256"]
        == json.loads(paths["metadata_path"].read_text("utf-8"))["model_sha256"]
    )


def test_load_final_model_artifact_uses_default_metadata_path(tmp_path: Path) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )

    loaded = load_final_model_artifact(paths["model_path"])

    assert loaded["metadata"]["artifact_version"] == ARTIFACT_VERSION


def test_load_final_model_artifact_rejects_checksum_mismatch(tmp_path: Path) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )
    paths["model_path"].write_bytes(paths["model_path"].read_bytes() + b"changed")

    with pytest.raises(ArtifactError, match="checksum"):
        load_final_model_artifact(paths["model_path"], paths["metadata_path"])


def test_load_final_model_artifact_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Model artifact file not found"):
        load_final_model_artifact(
            tmp_path / MODEL_FILENAME, tmp_path / METADATA_FILENAME
        )

    model_path = tmp_path / MODEL_FILENAME
    joblib.dump(make_fitted_pipeline(), model_path)
    with pytest.raises(ArtifactError, match="Metadata file not found"):
        load_final_model_artifact(model_path, tmp_path / METADATA_FILENAME)


def test_load_final_model_artifact_rejects_malformed_json(tmp_path: Path) -> None:
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(make_fitted_pipeline(), model_path)
    metadata_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ArtifactError, match="malformed") as exc_info:
        load_final_model_artifact(model_path, metadata_path)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_features", ["Type"], "features"),
        ("operating_threshold", 1.0, "threshold"),
        ("artifact_version", "2.0.0", "Unsupported artifact version"),
        ("artifact_version", "version-one", "semantic version"),
        ("target_column", "target", "target"),
        ("excluded_identifier_columns", ["UDI"], "identifier exclusions"),
        ("excluded_leakage_columns", ["TWF"], "leakage-column exclusions"),
        ("random_seed", 7, "random seed"),
        ("development_row_count", 5, "sum to row count"),
        ("development_class_distribution", {"0": 6}, "class distribution"),
        ("training_timestamp_utc", "2026-07-15T00:00:00", "UTC-aware"),
        ("training_timestamp_utc", "not-a-timestamp", "ISO 8601"),
        ("python_version", "", "python_version"),
        ("project_package_version", "9.9.9", "package version"),
        ("model_sha256", None, "SHA-256"),
        ("model_sha256", "not-a-checksum", "SHA-256"),
    ],
)
def test_load_final_model_artifact_rejects_invalid_metadata(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )
    metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
    metadata[field] = value
    paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match=message):
        load_final_model_artifact(paths["model_path"], paths["metadata_path"])


def test_load_final_model_artifact_rejects_missing_metadata_field(
    tmp_path: Path,
) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )
    metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
    del metadata["model_features"]
    paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="missing required fields"):
        load_final_model_artifact(paths["model_path"], paths["metadata_path"])


@pytest.mark.parametrize(
    ("bad_object", "message"),
    [
        (object(), "Pipeline"),
        (_ControlledFitModel(), "Pipeline"),
        (Pipeline([("regressor", LinearRegression())]), "preprocessor and classifier"),
    ],
)
def test_load_final_model_artifact_rejects_non_pipeline_or_invalid_object(
    tmp_path: Path,
    bad_object: Any,
    message: str,
) -> None:
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(bad_object, model_path)
    metadata = make_metadata(model_sha256=artifacts._sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match=message):
        load_final_model_artifact(model_path, metadata_path)


def test_load_final_model_artifact_rejects_unfitted_pipeline(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(pipeline, model_path)
    metadata = make_metadata(model_sha256=artifacts._sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="fitted"):
        load_final_model_artifact(model_path, metadata_path)


def test_load_final_model_artifact_rejects_missing_predict_proba(
    tmp_path: Path,
) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LinearRegression()),
        ]
    )
    pipeline.fit(make_feature_frame(), pd.Series([0.0, 1.0, 0.0, 1.0]))
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(pipeline, model_path)
    metadata = make_metadata(model_sha256=artifacts._sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="predict_proba"):
        load_final_model_artifact(model_path, metadata_path)


def test_load_final_model_artifact_rejects_missing_class_one(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(make_feature_frame(), pd.Series([0, 0, 0, 0]))
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(pipeline, model_path)
    metadata = make_metadata(model_sha256=artifacts._sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="class 1"):
        load_final_model_artifact(model_path, metadata_path)


def test_load_final_model_artifact_rejects_incompatible_input_features(
    tmp_path: Path,
) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", "passthrough"),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(pd.DataFrame({"unexpected": [0, 1, 2, 3]}), pd.Series([0, 1, 0, 1]))
    model_path = tmp_path / MODEL_FILENAME
    metadata_path = tmp_path / METADATA_FILENAME
    joblib.dump(pipeline, model_path)
    metadata = make_metadata(model_sha256=artifacts._sha256(model_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="input features"):
        load_final_model_artifact(model_path, metadata_path)


def test_load_final_model_artifact_checks_checksum_before_joblib_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )
    metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
    metadata["model_sha256"] = "0" * 64
    paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")

    def fail_if_called(path: Path) -> Any:
        raise AssertionError(f"joblib.load should not be called for {path}")

    monkeypatch.setattr(artifacts.joblib, "load", fail_if_called)

    with pytest.raises(ArtifactError, match="checksum"):
        load_final_model_artifact(paths["model_path"], paths["metadata_path"])


def test_save_final_model_artifact_rejects_invalid_contract(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Pipeline"):
        save_final_model_artifact(
            {"fitted_pipeline": object(), "metadata": {}}, tmp_path
        )
    with pytest.raises(ArtifactError, match="metadata"):
        save_final_model_artifact({"fitted_pipeline": make_fitted_pipeline()}, tmp_path)


def test_metadata_fields_are_deterministically_ordered(tmp_path: Path) -> None:
    paths = save_final_model_artifact(
        {"fitted_pipeline": make_fitted_pipeline(), "metadata": make_metadata()},
        tmp_path,
    )
    metadata = json.loads(paths["metadata_path"].read_text("utf-8"))

    assert tuple(metadata) == artifacts.REQUIRED_METADATA_FIELDS


def test_metadata_contains_security_and_reproducibility_wording() -> None:
    metadata = make_metadata()
    text = json.dumps(metadata)

    assert "synthetic UCI AI4I 2020" in text
    assert "not production-certified" in text
    assert "development" in text
    assert "held-out test split is excluded" in text
    assert metadata["project_package_version"] == __version__


def test_models_directory_artifacts_are_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "models/final/example.joblib"],
        cwd=config.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "models/" in result.stdout
