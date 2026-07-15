from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance import __version__, config, predict
from predictive_maintenance.artifacts import (
    ARTIFACT_VERSION,
    save_final_model_artifact,
)
from predictive_maintenance.exceptions import ArtifactError, PredictionError
from predictive_maintenance.predict import (
    load_prediction_input,
    predict_dataset,
    run_batch_prediction,
    save_predictions,
)
from predictive_maintenance.preprocessing import build_preprocessor


class RecordingPipeline:
    def __init__(
        self,
        probabilities: np.ndarray | None = None,
        classes: tuple[int, ...] = (0, 1),
    ) -> None:
        self.classes_ = np.array(classes)
        self.probabilities = (
            probabilities
            if probabilities is not None
            else np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        )
        self.predict_proba_call_count = 0
        self.fit_called = False
        self.seen_columns: tuple[str, ...] | None = None
        self.seen_index: tuple[Any, ...] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_proba_call_count += 1
        self.seen_columns = tuple(X.columns)
        self.seen_index = tuple(X.index)
        return self.probabilities

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.fit_called = True
        raise AssertionError("Batch prediction must not retrain the model.")


def make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H"],
            "Air temperature [K]": [298.1, 299.2, 300.3],
            "Process temperature [K]": [308.1, 309.2, 310.3],
            "Rotational speed [rpm]": [1400, 1500, 1600],
            "Torque [Nm]": [40.0, 42.0, 44.0],
            "Tool wear [min]": [10, 20, 30],
        },
        index=[101, 105, 109],
        columns=list(config.MODEL_FEATURES),
    )


def make_metadata(threshold: float = 0.8) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "project_package_version": __version__,
        "operating_threshold": threshold,
    }


def make_artifact(
    pipeline: RecordingPipeline | None = None,
    threshold: float = 0.8,
) -> dict[str, Any]:
    return {
        "fitted_pipeline": pipeline if pipeline is not None else RecordingPipeline(),
        "metadata": make_metadata(threshold),
    }


def patch_artifact_loader(
    monkeypatch: pytest.MonkeyPatch,
    artifact: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        predict,
        "load_final_model_artifact",
        lambda model_path, metadata_path=None: artifact,
    )


def test_load_prediction_input_accepts_valid_dataframe() -> None:
    source = make_features()
    original = source.copy(deep=True)

    loaded = load_prediction_input(source)

    pd.testing.assert_frame_equal(loaded, source)
    assert loaded is not source
    pd.testing.assert_frame_equal(source, original)


def test_load_prediction_input_accepts_valid_csv_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "batch.csv"
    make_features().to_csv(csv_path, index=False)

    loaded = load_prediction_input(csv_path)

    assert tuple(loaded.columns) == config.MODEL_FEATURES
    assert len(loaded) == 3


def test_load_prediction_input_accepts_valid_csv_string_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "batch.csv"
    make_features().to_csv(csv_path, index=False)

    loaded = load_prediction_input(str(csv_path))

    assert tuple(loaded.columns) == config.MODEL_FEATURES


def test_load_prediction_input_rejects_missing_column() -> None:
    dataframe = make_features().drop(columns=["Torque [Nm]"])

    with pytest.raises(PredictionError, match="missing"):
        load_prediction_input(dataframe)


def test_load_prediction_input_rejects_duplicate_columns() -> None:
    dataframe = make_features()
    dataframe.columns = [
        "Type",
        "Type",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    with pytest.raises(PredictionError, match="schema"):
        load_prediction_input(dataframe)


def test_load_prediction_input_rejects_wrong_feature_order() -> None:
    dataframe = make_features().loc[
        :,
        [
            "Air temperature [K]",
            "Type",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
    ]

    with pytest.raises(PredictionError, match="same order"):
        load_prediction_input(dataframe)


def test_load_prediction_input_rejects_extra_feature() -> None:
    dataframe = make_features().assign(extra=1)

    with pytest.raises(PredictionError, match="unexpected"):
        load_prediction_input(dataframe)


def test_load_prediction_input_allows_nan_values() -> None:
    dataframe = make_features()
    original = dataframe.copy(deep=True)
    dataframe.loc[105, "Torque [Nm]"] = np.nan

    loaded = load_prediction_input(dataframe)

    assert pd.isna(loaded.loc[105, "Torque [Nm]"])
    assert pd.isna(dataframe.loc[105, "Torque [Nm]"])
    pd.testing.assert_frame_equal(dataframe.drop(index=105), original.drop(index=105))


def test_load_prediction_input_rejects_duplicate_csv_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate_headers.csv"
    csv_path.write_text(
        "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
        "Torque [Nm],Torque [Nm]\n"
        "L,298.1,308.1,1400,40.0,41.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PredictionError,
        match="duplicate columns: Torque \\[Nm\\]",
    ) as exc:
        load_prediction_input(csv_path)

    assert ".1" not in str(exc.value)


def test_predict_dataset_uses_metadata_threshold_and_preserves_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline(
        np.array(
            [
                [0.95, 0.05],
                [0.19, 0.81],
                [0.2, 0.8],
            ]
        )
    )
    patch_artifact_loader(monkeypatch, make_artifact(pipeline, threshold=0.8))

    predictions = predict_dataset(make_features(), "model.joblib", "metadata.json")

    assert tuple(predictions.columns) == predict.PREDICTION_COLUMNS
    assert tuple(predictions.index) == (101, 105, 109)
    assert predictions["failure_probability"].tolist() == [0.05, 0.81, 0.8]
    assert predictions["predicted_failure"].tolist() == [0, 1, 1]
    assert pipeline.predict_proba_call_count == 1
    assert pipeline.seen_columns == config.MODEL_FEATURES
    assert pipeline.seen_index == (101, 105, 109)
    assert not pipeline.fit_called


def test_predict_dataset_extracts_class_one_probability_by_class_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline(
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        classes=(1, 0),
    )
    patch_artifact_loader(monkeypatch, make_artifact(pipeline, threshold=0.5))

    predictions = predict_dataset(make_features().iloc[:2], "model.joblib")

    assert predictions["failure_probability"].tolist() == [0.9, 0.2]
    assert predictions["predicted_failure"].tolist() == [1, 0]
    assert pipeline.predict_proba_call_count == 1


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[0.5, 0.5], [0.5, 0.5]]), "input row count"),
        (np.array([[0.5, 0.5]] * 4), "input row count"),
        (np.array([[0.9, 0.1], [np.nan, np.nan], [0.6, 0.4]]), "finite"),
        (np.array([[0.9, 0.1], [-np.inf, np.inf], [0.6, 0.4]]), "finite"),
        (np.array([[0.9, 0.1], [1.1, -0.1], [0.6, 0.4]]), r"\[0, 1\]"),
    ],
)
def test_predict_dataset_rejects_invalid_probabilities(
    probabilities: np.ndarray,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline(probabilities=probabilities)
    patch_artifact_loader(monkeypatch, make_artifact(pipeline))

    with pytest.raises(PredictionError, match=message):
        predict_dataset(make_features(), "model.joblib")


def test_prediction_probability_validation_rejects_invalid_shape() -> None:
    with pytest.raises(PredictionError, match="one-dimensional"):
        predict._validate_prediction_probabilities(
            np.array([[0.1], [0.2], [0.3]]),
            expected_length=3,
        )


@pytest.mark.parametrize(
    "threshold",
    [np.nan, np.inf, -np.inf, 0.0, 1.0, -0.1, 1.1, "high"],
)
def test_predict_dataset_rejects_invalid_metadata_thresholds(
    threshold: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline()
    patch_artifact_loader(monkeypatch, make_artifact(pipeline, threshold=threshold))

    with pytest.raises(PredictionError, match="threshold"):
        predict_dataset(make_features(), "model.joblib")
    assert pipeline.predict_proba_call_count == 0


def test_predict_dataset_wraps_probability_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPipeline(RecordingPipeline):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            raise RuntimeError("broken")

    patch_artifact_loader(monkeypatch, make_artifact(BrokenPipeline()))

    with pytest.raises(PredictionError, match="probabilities") as exc_info:
        predict_dataset(make_features(), "model.joblib")
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_predict_dataset_wraps_artifact_loading_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(model_path: Path | str, metadata_path: Path | str | None = None) -> None:
        raise ArtifactError("bad artifact")

    monkeypatch.setattr(predict, "load_final_model_artifact", fail)

    with pytest.raises(PredictionError, match="artifact") as exc_info:
        predict_dataset(make_features(), "model.joblib")
    assert isinstance(exc_info.value.__cause__, ArtifactError)


def test_save_predictions_writes_csv(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "failure_probability": [0.1, 0.9],
            "predicted_failure": [0, 1],
        },
        index=[7, 9],
    )

    csv_path = save_predictions(predictions, tmp_path)
    reloaded = pd.read_csv(csv_path)

    assert csv_path == tmp_path / "predictions.csv"
    assert reloaded.columns.tolist() == [
        "input_index",
        "failure_probability",
        "predicted_failure",
    ]
    assert reloaded["input_index"].tolist() == [7, 9]
    assert reloaded["failure_probability"].tolist() == [0.1, 0.9]
    assert reloaded["predicted_failure"].tolist() == [0, 1]


def test_run_batch_prediction_returns_report_and_saves_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline()
    patch_artifact_loader(monkeypatch, make_artifact(pipeline, threshold=0.7))

    report = run_batch_prediction(
        make_features(),
        artifact_dir=tmp_path / "artifact",
        output_dir=tmp_path / "predictions",
    )

    assert report["prediction_count"] == 3
    assert report["threshold_used"] == 0.7
    assert report["model_version"] == __version__
    assert report["project_package_version"] == __version__
    assert report["artifact_version"] == ARTIFACT_VERSION
    assert isinstance(report["prediction_dataframe"], pd.DataFrame)
    assert report["prediction_csv_path"] == tmp_path / "predictions" / "predictions.csv"
    assert report["prediction_csv_path"].is_file()
    assert pipeline.predict_proba_call_count == 1
    assert not pipeline.fit_called


def test_run_batch_prediction_rejects_conflicting_artifact_paths() -> None:
    with pytest.raises(PredictionError, match="either artifact_dir or model_path"):
        run_batch_prediction(
            make_features(),
            artifact_dir="models/final",
            model_path="model.joblib",
        )


def test_run_batch_prediction_does_not_call_evaluation_or_threshold_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = RecordingPipeline()
    patch_artifact_loader(monkeypatch, make_artifact(pipeline))

    forbidden_names = (
        "run_business_threshold_recommendation",
        "run_threshold_analysis",
        "run_final_model_evaluation",
    )

    for name in forbidden_names:
        assert not hasattr(predict, name)

    report = run_batch_prediction(make_features(), model_path="model.joblib")

    assert report["prediction_count"] == 3
    assert set(report) == {
        "prediction_count",
        "threshold_used",
        "model_version",
        "project_package_version",
        "artifact_version",
        "prediction_dataframe",
    }
    assert pipeline.predict_proba_call_count == 1
    assert not pipeline.fit_called


def test_predict_dataset_uses_reload_artifact(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(make_features(), pd.Series([0, 1, 0], index=make_features().index))
    artifact = {
        "fitted_pipeline": pipeline,
        "metadata": _full_metadata(development_row_count=3),
    }
    paths = save_final_model_artifact(artifact, tmp_path)

    predictions = predict_dataset(
        make_features(),
        model_path=paths["model_path"],
        metadata_path=paths["metadata_path"],
    )

    assert tuple(predictions.columns) == predict.PREDICTION_COLUMNS
    assert len(predictions) == 3


def _full_metadata(development_row_count: int) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": "hist_gradient_boosting_balanced",
        "operating_profile": "balanced",
        "operating_threshold": 0.8,
        "target_column": config.TARGET_COLUMN,
        "model_features": list(config.MODEL_FEATURES),
        "numerical_features": list(config.NUMERICAL_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "excluded_identifier_columns": list(config.IDENTIFIER_COLUMNS),
        "excluded_leakage_columns": list(config.FAILURE_MODE_COLUMNS),
        "random_seed": config.RANDOM_SEED,
        "development_row_count": development_row_count,
        "development_class_distribution": {"0": 2, "1": 1},
        "training_timestamp_utc": "2026-07-15T00:00:00+00:00",
        "python_version": "3.12.3",
        "pandas_version": pd.__version__,
        "scikit_learn_version": "1.9.0",
        "project_package_version": __version__,
        "final_evaluation_reference": (
            "docs/results_summary.md documents final held-out test evidence; "
            "test metrics are not used as model-selection inputs."
        ),
        "synthetic_dataset_limitation": (
            "Trained on the synthetic UCI AI4I 2020 dataset; not production-certified "
            "for real industrial equipment."
        ),
        "intended_use": "Batch machine-failure risk scoring.",
        "non_intended_use": "Production deployment without operational validation.",
        "governance_statement": (
            "Model and threshold were selected from validation workflows before final "
            "test evaluation. This artifact is fit on train plus validation data only; "
            "the held-out test split is excluded from fitting and sample weighting."
        ),
        "model_sha256": None,
    }
