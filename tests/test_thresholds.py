import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from predictive_maintenance import config, thresholds
from predictive_maintenance.exceptions import ThresholdAnalysisError
from predictive_maintenance.models import HIST_GRADIENT_BOOSTING_MODEL_NAME
from predictive_maintenance.splitting import SplitData
from predictive_maintenance.thresholds import (
    DEFAULT_THRESHOLDS,
    THRESHOLD_REPORT_COLUMNS,
    calculate_threshold_costs,
    evaluate_thresholds,
    run_threshold_analysis,
    save_threshold_analysis_report,
    select_best_f1_threshold,
    select_lowest_cost_threshold,
    select_threshold_for_minimum_recall,
)


def make_threshold_dataframe() -> pd.DataFrame:
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


def test_default_threshold_grid_is_deterministic_and_includes_half() -> None:
    assert DEFAULT_THRESHOLDS[0] == 0.05
    assert DEFAULT_THRESHOLDS[-1] == 0.95
    assert 0.5 in DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS == tuple(sorted(DEFAULT_THRESHOLDS))


def test_evaluate_thresholds_returns_expected_metrics_and_order() -> None:
    results = evaluate_thresholds(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.6, 0.7, 0.4]),
        thresholds=[0.25, 0.5, 0.75],
    )

    assert [result["threshold"] for result in results] == [0.25, 0.5, 0.75]
    assert results[1] == {
        "threshold": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
        "predicted_positive_count": 2,
        "predicted_positive_rate": 0.5,
    }
    json.dumps(results, allow_nan=False)


def test_custom_thresholds_are_deduplicated_and_sorted() -> None:
    results = evaluate_thresholds(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.6, 0.7, 0.4]),
        thresholds=[0.75, 0.25, 0.5, 0.25],
    )

    assert [result["threshold"] for result in results] == [0.25, 0.5, 0.75]


def test_custom_threshold_half_is_included_only_when_supplied() -> None:
    without_half = evaluate_thresholds(
        pd.Series([0, 1]),
        np.array([0.2, 0.8]),
        thresholds=[0.25, 0.75],
    )
    with_half = evaluate_thresholds(
        pd.Series([0, 1]),
        np.array([0.2, 0.8]),
        thresholds=[0.75, 0.5, 0.25],
    )
    default_results = evaluate_thresholds(pd.Series([0, 1]), np.array([0.2, 0.8]))

    assert [result["threshold"] for result in without_half] == [0.25, 0.75]
    assert [result["threshold"] for result in with_half] == [0.25, 0.5, 0.75]
    assert 0.5 in [result["threshold"] for result in default_results]


@pytest.mark.parametrize(
    ("y_true", "message"),
    [
        ([0, 1, np.nan], "missing values"),
        ([0, 0, 0], "exactly \\{0, 1\\}"),
        ([0, 2, 2], "exactly \\{0, 1\\}"),
        (["0", "1", "1"], "exactly \\{0, 1\\}"),
    ],
)
def test_evaluate_thresholds_rejects_invalid_targets(
    y_true: list[Any],
    message: str,
) -> None:
    with pytest.raises(ThresholdAnalysisError, match=message):
        evaluate_thresholds(y_true, np.array([0.1, 0.2, 0.3]), thresholds=[0.5])


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([0.1, np.inf], "finite"),
        ([0.1, -np.inf], "finite"),
        ([0.1, np.nan], "finite"),
        ([-0.1, 0.5], "\\[0, 1\\]"),
        ([0.1, 1.1], "\\[0, 1\\]"),
    ],
)
def test_evaluate_thresholds_rejects_invalid_probabilities(
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(ThresholdAnalysisError, match=message):
        evaluate_thresholds([0, 1], np.array(probabilities), thresholds=[0.5])


def test_evaluate_thresholds_rejects_length_mismatch() -> None:
    with pytest.raises(ThresholdAnalysisError, match="same length"):
        evaluate_thresholds([0, 1], np.array([0.2]), thresholds=[0.5])


@pytest.mark.parametrize(
    ("custom_thresholds", "message"),
    [
        ([], "At least one threshold"),
        ([0.0], "strictly between 0 and 1"),
        ([1.0], "strictly between 0 and 1"),
        ([-0.1], "strictly between 0 and 1"),
        ([1.1], "strictly between 0 and 1"),
        ([np.nan], "finite"),
        ([np.inf], "finite"),
        ([-np.inf], "finite"),
        (["not-a-threshold"], "numeric"),
    ],
)
def test_evaluate_thresholds_rejects_invalid_custom_thresholds(
    custom_thresholds: list[Any],
    message: str,
) -> None:
    with pytest.raises(ThresholdAnalysisError, match=message):
        evaluate_thresholds([0, 1], np.array([0.2, 0.8]), thresholds=custom_thresholds)


def test_select_best_f1_threshold_uses_deterministic_tie_breaking() -> None:
    results = [
        {"threshold": 0.2, "f1": 0.8, "recall": 0.7, "precision": 0.9},
        {"threshold": 0.4, "f1": 0.8, "recall": 0.8, "precision": 0.6},
        {"threshold": 0.6, "f1": 0.8, "recall": 0.8, "precision": 0.6},
    ]

    selected = select_best_f1_threshold(results)

    assert selected["threshold"] == 0.6


def test_select_threshold_for_minimum_recall_maximizes_precision() -> None:
    results = [
        {"threshold": 0.2, "recall": 1.0, "precision": 0.2, "f1": 0.33},
        {"threshold": 0.4, "recall": 0.8, "precision": 0.7, "f1": 0.75},
        {"threshold": 0.6, "recall": 0.7, "precision": 0.9, "f1": 0.79},
    ]

    selected = select_threshold_for_minimum_recall(results, minimum_recall=0.8)

    assert selected["threshold"] == 0.4


def test_select_threshold_for_minimum_recall_raises_when_no_threshold_qualifies() -> (
    None
):
    with pytest.raises(ThresholdAnalysisError, match="No threshold satisfies"):
        select_threshold_for_minimum_recall(
            [{"threshold": 0.5, "recall": 0.3, "precision": 1.0, "f1": 0.46}],
            minimum_recall=0.8,
        )


def test_calculate_threshold_costs_requires_false_negative_cost_to_be_higher() -> None:
    with pytest.raises(ThresholdAnalysisError, match="false_negative_cost"):
        calculate_threshold_costs(
            [{"threshold": 0.5, "false_negative": 1, "false_positive": 1}],
            false_negative_cost=1.0,
            false_positive_cost=10.0,
        )


@pytest.mark.parametrize(
    ("false_negative_cost", "false_positive_cost", "message"),
    [
        (0.0, 1.0, "positive"),
        (10.0, 0.0, "positive"),
        (-1.0, 1.0, "positive"),
        (10.0, -1.0, "positive"),
        (1.0, 1.0, "greater than false_positive_cost"),
        (1.0, 2.0, "greater than false_positive_cost"),
        (np.inf, 1.0, "finite"),
        (10.0, np.inf, "finite"),
        (np.nan, 1.0, "finite"),
    ],
)
def test_calculate_threshold_costs_rejects_invalid_cost_assumptions(
    false_negative_cost: float,
    false_positive_cost: float,
    message: str,
) -> None:
    with pytest.raises(ThresholdAnalysisError, match=message):
        calculate_threshold_costs(
            [{"threshold": 0.5, "false_negative": 1, "false_positive": 1}],
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )


def test_calculate_threshold_costs_adds_total_cost() -> None:
    results = calculate_threshold_costs(
        [{"threshold": 0.5, "false_negative": 2, "false_positive": 3}],
        false_negative_cost=10.0,
        false_positive_cost=1.0,
    )

    assert results[0]["total_cost"] == 23.0


def test_select_lowest_cost_threshold_uses_deterministic_tie_breaking() -> None:
    results = [
        {
            "threshold": 0.2,
            "total_cost": 10.0,
            "recall": 0.8,
            "precision": 0.5,
            "f1": 0.62,
        },
        {
            "threshold": 0.4,
            "total_cost": 10.0,
            "recall": 0.8,
            "precision": 0.6,
            "f1": 0.69,
        },
    ]

    selected = select_lowest_cost_threshold(results)

    assert selected["threshold"] == 0.4


def test_select_lowest_cost_threshold_requires_costs() -> None:
    with pytest.raises(ThresholdAnalysisError, match="total_cost"):
        select_lowest_cost_threshold([{"threshold": 0.5}])


def test_run_threshold_analysis_returns_validation_only_results() -> None:
    results = run_threshold_analysis(make_threshold_dataframe())

    assert results["model"] == HIST_GRADIENT_BOOSTING_MODEL_NAME
    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert results["validation_class_distribution"] == {"0": 16, "1": 4}
    assert "test_metrics" not in results
    assert "untouched" in results["test_set_status"]
    assert results["selected_thresholds"]["default_0_5"]["threshold"] == 0.5
    assert len(results["threshold_results"]) == len(DEFAULT_THRESHOLDS)
    json.dumps(results, allow_nan=False)


def test_run_threshold_analysis_output_contains_no_models_or_probabilities() -> None:
    results = run_threshold_analysis(make_threshold_dataframe())

    json.dumps(results, allow_nan=False)
    assert "test_metrics" not in results
    assert "probabilities" not in json.dumps(results)
    _assert_no_model_objects(results)


def test_run_threshold_analysis_does_not_mutate_source_dataframe() -> None:
    dataframe = make_threshold_dataframe()
    original = dataframe.copy(deep=True)

    run_threshold_analysis(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_run_threshold_analysis_never_scores_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = thresholds.split_dataset(make_threshold_dataframe())
    test_index = tuple(split_data.X_test.index)
    scored_indices = []

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            scored_indices.append(tuple(X.index))
            if tuple(X.index) == test_index:
                raise AssertionError("X_test must not be scored")
            return np.column_stack(
                [
                    np.full(len(X), 0.7),
                    np.full(len(X), 0.3),
                ]
            )

    monkeypatch.setattr(
        thresholds,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ControlledModel()},
        raising=False,
    )

    run_threshold_analysis(make_threshold_dataframe())

    assert scored_indices == [tuple(split_data.X_validation.index)]


def test_run_threshold_analysis_uses_reversed_class_order_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = SplitData(
        X_train=pd.DataFrame({"feature": [1, 2, 3, 4]}, index=[10, 11, 12, 13]),
        X_validation=pd.DataFrame({"feature": [5, 6, 7, 8]}, index=[20, 21, 22, 23]),
        X_test=pd.DataFrame({"feature": [9, 10]}, index=[30, 31]),
        y_train=pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13]),
        y_validation=pd.Series([0, 0, 1, 1], index=[20, 21, 22, 23]),
        y_test=pd.Series([0, 1], index=[30, 31]),
    )

    class ReversedClassModel:
        classes_ = np.array([1, 0])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array(
                [
                    [0.1, 0.9],
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.4, 0.6],
                ]
            )

    monkeypatch.setattr(thresholds, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        thresholds,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ReversedClassModel()},
    )

    results = run_threshold_analysis(make_threshold_dataframe())

    assert results["selected_thresholds"]["default_0_5"] == {
        "threshold": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
        "predicted_positive_count": 2,
        "predicted_positive_rate": 0.5,
        "total_cost": 11.0,
    }


def test_run_threshold_analysis_passes_sample_weights_only_from_training_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_fit = {}
    split_data = thresholds.split_dataset(make_threshold_dataframe())

    class ControlledModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            observed_fit["X_index"] = tuple(X.index)
            observed_fit["y_index"] = tuple(y.index)
            observed_fit["sample_weight_length"] = len(
                fit_parameters["classifier__sample_weight"]
            )
            return self

        def predict_proba(self, X):
            return np.column_stack(
                [
                    np.full(len(X), 0.7),
                    np.full(len(X), 0.3),
                ]
            )

    monkeypatch.setattr(
        thresholds,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ControlledModel()},
        raising=False,
    )

    run_threshold_analysis(make_threshold_dataframe())

    assert observed_fit == {
        "X_index": tuple(split_data.X_train.index),
        "y_index": tuple(split_data.y_train.index),
        "sample_weight_length": len(split_data.y_train),
    }


def test_run_threshold_analysis_rejects_unknown_model_name() -> None:
    with pytest.raises(ThresholdAnalysisError, match="Unknown model_name"):
        run_threshold_analysis(make_threshold_dataframe(), model_name="missing_model")


def test_run_threshold_analysis_wraps_predict_proba_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProbabilityModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            raise RuntimeError("probability failure")

    monkeypatch.setattr(
        thresholds,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: FailingProbabilityModel()},
    )

    with pytest.raises(ThresholdAnalysisError, match="generate validation") as exc_info:
        run_threshold_analysis(make_threshold_dataframe())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "probability failure"


def test_run_threshold_analysis_wraps_missing_class_one_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingClassOneModel:
        classes_ = np.array([0, 2])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.column_stack(
                [
                    np.full(len(X), 0.8),
                    np.full(len(X), 0.2),
                ]
            )

    monkeypatch.setattr(
        thresholds,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: MissingClassOneModel()},
    )

    with pytest.raises(ThresholdAnalysisError, match="class-1") as exc_info:
        run_threshold_analysis(make_threshold_dataframe())

    assert exc_info.value.__cause__ is not None
    assert "include class 1" in str(exc_info.value.__cause__)


def test_save_threshold_analysis_report_writes_strict_json_and_ordered_csv(
    tmp_path: Path,
) -> None:
    results = run_threshold_analysis(make_threshold_dataframe())

    output_paths = save_threshold_analysis_report(results, tmp_path)

    assert output_paths["threshold_analysis_json"] == (
        tmp_path / "threshold_analysis.json"
    )
    assert output_paths["threshold_metrics_csv"] == tmp_path / "threshold_metrics.csv"
    saved_results = json.loads(
        output_paths["threshold_analysis_json"].read_text(encoding="utf-8")
    )
    saved_metrics = pd.read_csv(output_paths["threshold_metrics_csv"])
    assert tuple(saved_results) == (
        "model",
        "split_row_counts",
        "validation_class_distribution",
        "assumptions",
        "selected_thresholds",
        "threshold_results",
        "test_set_status",
    )
    assert tuple(saved_metrics.columns) == THRESHOLD_REPORT_COLUMNS
    assert tuple(saved_metrics["threshold"]) == DEFAULT_THRESHOLDS


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
    assert not isinstance(value, np.ndarray)
    assert not hasattr(value, "predict_proba")
