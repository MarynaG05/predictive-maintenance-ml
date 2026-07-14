import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance import config, train
from predictive_maintenance.models import (
    DUMMY_PRIOR_MODEL_NAME,
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
)
from predictive_maintenance.train import (
    MODEL_COMPARISON_COLUMNS,
    _best_model_name,
    _fit_model,
    run_baseline_training,
    run_model_comparison,
    save_baseline_report,
    save_model_comparison_report,
)


def make_training_dataframe() -> pd.DataFrame:
    rows = []
    labels = [0] * 80 + [1] * 20
    for index, target in enumerate(labels, start=1):
        rows.append(
            {
                "UDI": index,
                "Product ID": f"ID{index:05d}",
                "Type": ("L", "M", "H")[index % 3],
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


def test_run_baseline_training_fits_and_evaluates_both_models() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert set(results["validation_metrics"]) == {
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
    }
    assert results["best_validation_model"] in results["validation_metrics"]


def test_run_model_comparison_fits_and_evaluates_all_models() -> None:
    results = run_model_comparison(make_training_dataframe())

    assert tuple(results["validation_metrics"]) == (
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
        RANDOM_FOREST_MODEL_NAME,
        HIST_GRADIENT_BOOSTING_MODEL_NAME,
    )
    assert results["best_validation_model"] in results["validation_metrics"]


def test_run_baseline_training_reports_only_validation_metrics() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert "validation_metrics" in results
    assert "test_metrics" not in results
    assert "no test-set metrics were calculated" in results["test_set_status"]


def test_run_model_comparison_reports_only_validation_metrics() -> None:
    results = run_model_comparison(make_training_dataframe())

    assert "validation_metrics" in results
    assert "test_metrics" not in results
    assert "no test-set metrics were calculated" in results["test_set_status"]


def test_run_baseline_training_reports_split_sizes_and_class_distributions() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert results["class_distributions"] == {
        "train": {"0": 48, "1": 12},
        "validation": {"0": 16, "1": 4},
        "test": {"0": 16, "1": 4},
    }


def test_run_model_comparison_reports_split_sizes_and_class_distributions() -> None:
    results = run_model_comparison(make_training_dataframe())

    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert results["class_distributions"] == {
        "train": {"0": 48, "1": 12},
        "validation": {"0": 16, "1": 4},
        "test": {"0": 16, "1": 4},
    }


def test_run_model_comparison_reports_non_negative_training_durations() -> None:
    results = run_model_comparison(make_training_dataframe())

    assert set(results["training_durations"]) == set(results["validation_metrics"])
    for model_name, metrics in results["validation_metrics"].items():
        assert metrics["training_seconds"] >= 0.0
        assert results["training_durations"][model_name] == metrics["training_seconds"]


def test_run_model_comparison_does_not_mutate_source_dataframe() -> None:
    dataframe = make_training_dataframe()
    original = dataframe.copy(deep=True)

    run_model_comparison(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_run_model_comparison_uses_provided_dataframe_without_disk_loading(
    monkeypatch,
) -> None:
    def fail_load_dataset():
        raise AssertionError("load_dataset should not be called")

    monkeypatch.setattr(train, "load_dataset", fail_load_dataset)

    results = run_model_comparison(make_training_dataframe())

    assert results["split_row_counts"]["train"] == 60


def test_run_model_comparison_uses_same_split_for_all_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_calls = []
    evaluation_call = {}

    class CaptureModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, X, y, **fit_parameters):
            fit_calls.append((self.name, tuple(X.index), tuple(y.index)))
            return self

    models = {
        DUMMY_PRIOR_MODEL_NAME: CaptureModel(DUMMY_PRIOR_MODEL_NAME),
        LOGISTIC_REGRESSION_MODEL_NAME: CaptureModel(LOGISTIC_REGRESSION_MODEL_NAME),
        RANDOM_FOREST_MODEL_NAME: CaptureModel(RANDOM_FOREST_MODEL_NAME),
        HIST_GRADIENT_BOOSTING_MODEL_NAME: CaptureModel(
            HIST_GRADIENT_BOOSTING_MODEL_NAME
        ),
    }

    def fake_evaluate_models(models, X_validation, y_validation):
        evaluation_call["models"] = tuple(models)
        evaluation_call["X_validation_index"] = tuple(X_validation.index)
        evaluation_call["y_validation_index"] = tuple(y_validation.index)
        return {
            model_name: {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "roc_auc": 1.0,
                "average_precision": 1.0,
                "confusion_matrix": {
                    "true_negatives": 1,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "true_positives": 1,
                },
            }
            for model_name in models
        }

    monkeypatch.setattr(train, "build_model_comparison_models", lambda: models)
    monkeypatch.setattr(train, "evaluate_models", fake_evaluate_models)

    run_model_comparison(make_training_dataframe())

    assert [name for name, _, _ in fit_calls] == list(models)
    X_train_indices = {indices for _, indices, _ in fit_calls}
    y_train_indices = {indices for _, _, indices in fit_calls}
    assert len(X_train_indices) == 1
    assert len(y_train_indices) == 1
    assert next(iter(X_train_indices)) == next(iter(y_train_indices))
    assert evaluation_call["models"] == tuple(models)
    assert (
        evaluation_call["X_validation_index"] == evaluation_call["y_validation_index"]
    )


def test_run_model_comparison_propagates_model_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def fit(self, X, y, **fit_parameters):
            raise RuntimeError("intentional fit failure")

    monkeypatch.setattr(
        train,
        "build_model_comparison_models",
        lambda: {DUMMY_PRIOR_MODEL_NAME: FailingModel()},
    )

    with pytest.raises(RuntimeError, match="intentional fit failure"):
        run_model_comparison(make_training_dataframe())


def test_save_baseline_report_writes_json_and_model_comparison(
    tmp_path: Path,
) -> None:
    results = run_baseline_training(make_training_dataframe())

    output_paths = save_baseline_report(results, tmp_path)

    assert output_paths["metrics_json"] == tmp_path / "baseline_metrics.json"
    assert output_paths["model_comparison_csv"] == tmp_path / "model_comparison.csv"
    assert output_paths["metrics_json"].is_file()
    assert output_paths["model_comparison_csv"].is_file()
    json.loads(output_paths["metrics_json"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(output_paths["model_comparison_csv"])
    assert set(comparison["model"]) == {
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
    }


def test_save_model_comparison_report_writes_exact_json_and_csv(
    tmp_path: Path,
) -> None:
    results = run_model_comparison(make_training_dataframe())

    output_paths = save_model_comparison_report(results, tmp_path)

    assert output_paths["metrics_json"] == tmp_path / "model_comparison_metrics.json"
    assert output_paths["model_comparison_csv"] == tmp_path / "model_comparison.csv"
    saved_results = json.loads(output_paths["metrics_json"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(output_paths["model_comparison_csv"])
    assert "test_metrics" not in saved_results
    assert tuple(comparison.columns) == MODEL_COMPARISON_COLUMNS
    assert tuple(comparison["model"]) == (
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
        RANDOM_FOREST_MODEL_NAME,
        HIST_GRADIENT_BOOSTING_MODEL_NAME,
    )


def test_save_model_comparison_report_preserves_json_model_order(
    tmp_path: Path,
) -> None:
    results = run_model_comparison(make_training_dataframe())

    output_paths = save_model_comparison_report(results, tmp_path)

    saved_results = json.loads(output_paths["metrics_json"].read_text(encoding="utf-8"))
    assert tuple(saved_results["validation_metrics"]) == (
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
        RANDOM_FOREST_MODEL_NAME,
        HIST_GRADIENT_BOOSTING_MODEL_NAME,
    )


def test_best_validation_model_breaks_average_precision_ties_by_name() -> None:
    validation_metrics = {
        "z_model": {"average_precision": 0.5},
        "a_model": {"average_precision": 0.5},
    }

    assert _best_model_name(validation_metrics) == "a_model"


def test_hist_gradient_boosting_sample_weights_use_only_training_target() -> None:
    class CaptureFitModel:
        def __init__(self) -> None:
            self.fit_parameters = None

        def fit(self, X, y, **fit_parameters):
            self.fit_parameters = fit_parameters
            return self

    X_train = make_training_dataframe().loc[:59, config.MODEL_FEATURES]
    y_train = pd.Series([0, 1, 0, 0, 1, 0])
    model = CaptureFitModel()

    training_seconds = _fit_model(
        HIST_GRADIENT_BOOSTING_MODEL_NAME,
        model,
        X_train,
        y_train,
    )

    sample_weight = model.fit_parameters["classifier__sample_weight"]
    assert training_seconds >= 0.0
    assert len(sample_weight) == len(y_train)
    expected_weights = np.array([0.75, 1.5, 0.75, 0.75, 1.5, 0.75])
    np.testing.assert_allclose(sample_weight, expected_weights)
    assert np.isfinite(sample_weight).all()
    assert (sample_weight > 0).all()


def test_sample_weight_is_passed_only_to_hist_gradient_boosting() -> None:
    class CaptureFitModel:
        def __init__(self) -> None:
            self.fit_parameters = None

        def fit(self, X, y, **fit_parameters):
            self.fit_parameters = fit_parameters
            return self

    X_train = make_training_dataframe().loc[:5, config.MODEL_FEATURES]
    y_train = pd.Series([0, 1, 0, 0, 1, 0])
    dummy_model = CaptureFitModel()
    hgb_model = CaptureFitModel()

    _fit_model(DUMMY_PRIOR_MODEL_NAME, dummy_model, X_train, y_train)
    _fit_model(HIST_GRADIENT_BOOSTING_MODEL_NAME, hgb_model, X_train, y_train)

    assert dummy_model.fit_parameters == {}
    assert "classifier__sample_weight" in hgb_model.fit_parameters
