import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from predictive_maintenance import config, final_evaluation
from predictive_maintenance.exceptions import FinalEvaluationError
from predictive_maintenance.final_evaluation import (
    DEFAULT_THRESHOLD,
    FINAL_EVALUATION_COLUMNS,
    FINAL_MODEL_NAME,
    FINAL_OPERATING_PROFILE,
    plot_final_confusion_matrices,
    run_final_model_evaluation,
    save_final_evaluation_outputs,
    save_final_evaluation_report,
)
from predictive_maintenance.splitting import SplitData


def make_final_dataframe() -> pd.DataFrame:
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
                "Torque [Nm]": 40.0 + target * 8.0 + index * 0.01,
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


def make_split_data() -> SplitData:
    columns = list(config.MODEL_FEATURES)
    X_train = pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [298, 299, 300, 301, 302, 303],
            "Process temperature [K]": [308, 309, 310, 311, 312, 313],
            "Rotational speed [rpm]": [1400, 1410, 1420, 1430, 1440, 1450],
            "Torque [Nm]": [40, 41, 42, 43, 44, 45],
            "Tool wear [min]": [1, 2, 3, 4, 5, 6],
        },
        index=[10, 11, 12, 13, 14, 15],
        columns=columns,
    )
    X_validation = pd.DataFrame(
        {
            "Type": ["L", "M"],
            "Air temperature [K]": [304, 305],
            "Process temperature [K]": [314, 315],
            "Rotational speed [rpm]": [1460, 1470],
            "Torque [Nm]": [46, 47],
            "Tool wear [min]": [7, 8],
        },
        index=[20, 21],
        columns=columns,
    )
    X_test = pd.DataFrame(
        {
            "Type": ["H", "L"],
            "Air temperature [K]": [306, 307],
            "Process temperature [K]": [316, 317],
            "Rotational speed [rpm]": [1480, 1490],
            "Torque [Nm]": [48, 49],
            "Tool wear [min]": [9, 10],
        },
        index=[30, 31],
        columns=columns,
    )
    return SplitData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=pd.Series([0, 1, 0, 1, 0, 1], index=X_train.index),
        y_validation=pd.Series([0, 1], index=X_validation.index),
        y_test=pd.Series([0, 1], index=X_test.index),
    )


def make_recommendations(threshold: float = 0.8) -> dict[str, Any]:
    return {
        "selected_model": FINAL_MODEL_NAME,
        "profiles": {
            FINAL_OPERATING_PROFILE: {
                "threshold": threshold,
            }
        },
    }


def test_run_final_model_evaluation_uses_fixed_model_threshold_and_dev_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    observed: dict[str, Any] = {"events": []}

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            observed["events"].append("fit")
            observed["fit_index"] = tuple(X.index)
            observed["fit_y_index"] = tuple(y.index)
            observed["sample_weight_length"] = len(
                fit_parameters["classifier__sample_weight"]
            )
            return self

        def predict_proba(self, X):
            observed["events"].append("predict_proba")
            observed["predict_index"] = tuple(X.index)
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    def recommendation(dataframe, model_name):
        observed["events"].append("recommendation")
        assert model_name == FINAL_MODEL_NAME
        return make_recommendations(0.8)

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        recommendation,
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ControlledModel()},
    )

    results = run_final_model_evaluation(make_final_dataframe())

    assert results["model_name"] == FINAL_MODEL_NAME
    assert results["operating_profile"] == FINAL_OPERATING_PROFILE
    assert results["selected_threshold"] == 0.8
    assert results["development_row_count"] == 8
    assert results["test_row_count"] == 2
    assert observed["fit_index"] == tuple(
        split_data.X_train.index.append(split_data.X_validation.index)
    )
    assert observed["fit_y_index"] == tuple(
        split_data.y_train.index.append(split_data.y_validation.index)
    )
    assert observed["sample_weight_length"] == 8
    assert observed["predict_index"] == tuple(split_data.X_test.index)
    assert observed["events"] == ["recommendation", "fit", "predict_proba"]
    assert results["governance"] == {
        "final_configuration_frozen_before_test": True,
        "test_evaluation_count": 1,
        "test_driven_changes_allowed": False,
        "process_note": results["governance"]["process_note"],
    }
    json.dumps(results, allow_nan=False)
    _assert_no_model_objects(results)
    assert "probabilities" not in json.dumps(results)


def test_run_final_model_evaluation_handles_reversed_class_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()

    class ReversedClassModel:
        classes_ = np.array([1, 0])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array(
                [
                    [0.4, 0.6],
                    [0.7, 0.3],
                ]
            )

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ReversedClassModel()},
    )

    results = run_final_model_evaluation(make_final_dataframe())

    assert results["test_roc_auc"] == 1.0
    assert results["test_average_precision"] == 1.0
    assert results["default_threshold_metrics"]["true_negative"] == 1
    assert results["default_threshold_metrics"]["true_positive"] == 1
    assert results["selected_threshold_metrics"]["false_negative"] == 1
    assert results["selected_threshold_metrics"]["predicted_positive_count"] == 0


def test_final_evaluation_metrics_are_exact_for_controlled_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ControlledModel()},
    )

    results = run_final_model_evaluation(make_final_dataframe())

    assert results["test_roc_auc"] == 1.0
    assert results["test_average_precision"] == 1.0
    assert results["default_threshold_metrics"] == {
        "threshold": DEFAULT_THRESHOLD,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 1,
        "predicted_positive_count": 1,
        "predicted_positive_rate": 0.5,
    }
    assert results["selected_threshold_metrics"] == {
        "threshold": 0.8,
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 1,
        "true_positive": 0,
        "predicted_positive_count": 0,
        "predicted_positive_rate": 0.0,
    }


def test_run_final_model_evaluation_does_not_mutate_source_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_final_dataframe()
    original = dataframe.copy(deep=True)
    split_data = make_split_data()

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    def mutate_recommendation_dataframe(dataframe, model_name):
        dataframe.iloc[0, dataframe.columns.get_loc("Type")] = "H"
        return make_recommendations(0.8)

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        mutate_recommendation_dataframe,
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ControlledModel()},
    )

    run_final_model_evaluation(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_run_final_model_evaluation_uses_provided_dataframe_without_disk_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    monkeypatch.setattr(
        final_evaluation,
        "load_dataset",
        lambda: (_ for _ in ()).throw(AssertionError("load_dataset should not run")),
    )
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ControlledModel()},
    )

    results = run_final_model_evaluation(make_final_dataframe())

    assert results["test_row_count"] == 2


@pytest.mark.parametrize("threshold", [None, -0.1, 0.0, 1.0, np.nan, np.inf])
def test_run_final_model_evaluation_rejects_invalid_recommendation_threshold(
    monkeypatch: pytest.MonkeyPatch,
    threshold: float | None,
) -> None:
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(threshold),
    )

    with pytest.raises(FinalEvaluationError, match="threshold"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_rejects_missing_balanced_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: {"profiles": {}},
    )

    with pytest.raises(FinalEvaluationError, match="balanced profile threshold"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_rejects_one_class_test_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    split_data = SplitData(
        X_train=split_data.X_train,
        X_validation=split_data.X_validation,
        X_test=split_data.X_test,
        y_train=split_data.y_train,
        y_validation=split_data.y_validation,
        y_test=pd.Series([0, 0], index=split_data.y_test.index),
    )
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)

    with pytest.raises(FinalEvaluationError, match="both classes"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_rejects_duplicate_development_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    X_train = split_data.X_train.copy()
    y_train = split_data.y_train.copy()
    X_train.index = [10, 10, 12, 13, 14, 15]
    y_train.index = X_train.index
    split_data = SplitData(
        X_train=X_train,
        X_validation=split_data.X_validation,
        X_test=split_data.X_test,
        y_train=y_train,
        y_validation=split_data.y_validation,
        y_test=split_data.y_test,
    )
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)

    with pytest.raises(FinalEvaluationError, match="Development feature indices"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_rejects_duplicate_test_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    X_test = split_data.X_test.copy()
    y_test = split_data.y_test.copy()
    X_test.index = [30, 30]
    y_test.index = X_test.index
    split_data = SplitData(
        X_train=split_data.X_train,
        X_validation=split_data.X_validation,
        X_test=X_test,
        y_train=split_data.y_train,
        y_validation=split_data.y_validation,
        y_test=y_test,
    )
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)

    with pytest.raises(FinalEvaluationError, match="Test feature indices"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_rejects_development_test_index_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = make_split_data()
    X_test = split_data.X_test.copy()
    y_test = split_data.y_test.copy()
    X_test.index = [20, 31]
    y_test.index = X_test.index
    split_data = SplitData(
        X_train=split_data.X_train,
        X_validation=split_data.X_validation,
        X_test=X_test,
        y_train=split_data.y_train,
        y_validation=split_data.y_validation,
        y_test=y_test,
    )
    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(final_evaluation, "split_dataset", lambda dataframe: split_data)

    with pytest.raises(FinalEvaluationError, match="disjoint"):
        run_final_model_evaluation(make_final_dataframe())


def test_run_final_model_evaluation_wraps_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def fit(self, X, y, **fit_parameters):
            raise ValueError("fit failed")

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: FailingModel()},
    )

    with pytest.raises(FinalEvaluationError, match="fit") as exc_info:
        run_final_model_evaluation(make_final_dataframe())
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_run_final_model_evaluation_wraps_runtime_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeFailingModel:
        def fit(self, X, y, **fit_parameters):
            raise RuntimeError("runtime fit failed")

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: RuntimeFailingModel()},
    )

    with pytest.raises(FinalEvaluationError, match="fit") as exc_info:
        run_final_model_evaluation(make_final_dataframe())
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_run_final_model_evaluation_wraps_predict_proba_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProbabilityModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            raise RuntimeError("predict failed")

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: FailingProbabilityModel()},
    )

    with pytest.raises(FinalEvaluationError, match="probabilities") as exc_info:
        run_final_model_evaluation(make_final_dataframe())
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_run_final_model_evaluation_wraps_missing_class_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingClassModel:
        classes_ = np.array([0, 2])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: MissingClassModel()},
    )

    with pytest.raises(FinalEvaluationError, match="probabilities") as exc_info:
        run_final_model_evaluation(make_final_dataframe())
    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize(
    ("probability_matrix", "message"),
    [
        (np.array([[0.6, np.nan], [0.3, 0.7]]), "finite"),
        (np.array([[0.6, np.inf], [0.3, 0.7]]), "finite"),
        (np.array([[0.6, -np.inf], [0.3, 0.7]]), "finite"),
        (np.array([[1.1, -0.1], [0.3, 0.7]]), r"\[0, 1\]"),
        (np.array([[0.6, 0.4], [-0.2, 1.2]]), r"\[0, 1\]"),
        (np.array([[0.6, 0.4]]), "number of test rows"),
    ],
)
def test_run_final_model_evaluation_rejects_invalid_final_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    probability_matrix: np.ndarray,
    message: str,
) -> None:
    class InvalidProbabilityModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return probability_matrix

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: InvalidProbabilityModel()},
    )

    with pytest.raises(FinalEvaluationError, match=message):
        run_final_model_evaluation(make_final_dataframe())


def test_save_final_evaluation_report_writes_reloadable_json_and_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = _controlled_final_results(monkeypatch)

    paths = save_final_evaluation_report(results, tmp_path)

    saved_json = json.loads(paths["final_evaluation_json"].read_text("utf-8"))
    saved_csv = pd.read_csv(paths["final_evaluation_csv"])
    assert saved_json["model_name"] == FINAL_MODEL_NAME
    assert saved_csv["operating_point"].tolist() == [
        "default_0_5",
        "selected_balanced_threshold",
    ]
    assert len(saved_csv) == 2
    assert tuple(saved_csv.columns) == FINAL_EVALUATION_COLUMNS


def test_final_evaluation_output_contract_contains_no_raw_test_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _controlled_final_results(monkeypatch)

    assert results["model_name"] == FINAL_MODEL_NAME
    assert results["operating_profile"] == FINAL_OPERATING_PROFILE
    assert results["selected_threshold"] == 0.8
    assert set(results) == {
        "model_name",
        "operating_profile",
        "selected_threshold",
        "model_selection_source",
        "threshold_selection_source",
        "development_row_count",
        "test_row_count",
        "development_class_distribution",
        "test_class_distribution",
        "test_roc_auc",
        "test_average_precision",
        "default_threshold_metrics",
        "selected_threshold_metrics",
        "final_evaluation_statement",
        "governance",
        "limitations",
        "reproducibility",
    }
    assert results["governance"] == {
        "final_configuration_frozen_before_test": True,
        "test_evaluation_count": 1,
        "test_driven_changes_allowed": False,
        "process_note": results["governance"]["process_note"],
    }
    assert _threshold_metric_keys(results) == {
        "default_threshold_metrics",
        "selected_threshold_metrics",
    }
    assert {
        results["default_threshold_metrics"]["threshold"],
        results["selected_threshold_metrics"]["threshold"],
    } == {DEFAULT_THRESHOLD, 0.8}
    json.dumps(results, allow_nan=False)
    _assert_no_model_objects(results)
    _assert_no_raw_test_artifacts(results)


def test_plot_final_confusion_matrices_creates_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = _controlled_final_results(monkeypatch)
    output_path = tmp_path / "confusion.png"

    path = plot_final_confusion_matrices(results, output_path)

    assert path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_final_evaluation_outputs_writes_reports_and_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = _controlled_final_results(monkeypatch)

    paths = save_final_evaluation_outputs(
        results,
        output_dir=tmp_path,
        figure_path=tmp_path / "confusion.png",
    )

    assert paths["final_evaluation_json"].is_file()
    assert paths["final_evaluation_csv"].is_file()
    assert paths["confusion_matrices_figure"].is_file()


def _controlled_final_results(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.3, 0.7]])

    monkeypatch.setattr(
        final_evaluation,
        "run_business_threshold_recommendation",
        lambda dataframe, model_name: make_recommendations(0.8),
    )
    monkeypatch.setattr(
        final_evaluation,
        "split_dataset",
        lambda dataframe: make_split_data(),
    )
    monkeypatch.setattr(
        final_evaluation,
        "build_model_comparison_models",
        lambda: {FINAL_MODEL_NAME: ControlledModel()},
    )
    return run_final_model_evaluation(make_final_dataframe())


def _assert_no_model_objects(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_model_objects(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_no_model_objects(item)
        return

    assert not isinstance(value, Pipeline)
    assert not isinstance(value, BaseEstimator)
    assert not hasattr(value, "predict_proba")


def _assert_no_raw_test_artifacts(value: Any) -> None:
    if isinstance(value, dict):
        forbidden_keys = {
            "X_test",
            "y_test",
            "test_features",
            "test_labels",
            "feature_rows",
            "labels",
            "probabilities",
            "raw_probabilities",
        }
        assert forbidden_keys.isdisjoint(value)
        for item in value.values():
            _assert_no_raw_test_artifacts(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_no_raw_test_artifacts(item)
        return

    assert not isinstance(value, pd.DataFrame | pd.Series | np.ndarray)


def _threshold_metric_keys(results: dict[str, Any]) -> set[str]:
    return {key for key in results if key.endswith("_threshold_metrics")}
