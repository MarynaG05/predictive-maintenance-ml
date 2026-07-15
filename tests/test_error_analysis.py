import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from predictive_maintenance import config, error_analysis
from predictive_maintenance.error_analysis import (
    ERROR_CASE_COLUMNS,
    OUTCOME_COLUMNS,
    build_error_analysis_table,
    classify_prediction_outcomes,
    run_validation_error_analysis,
    save_error_analysis_report,
    summarize_categorical_features_by_outcome,
    summarize_false_negatives,
    summarize_false_positives,
    summarize_numerical_features_by_outcome,
    summarize_outcomes,
)
from predictive_maintenance.exceptions import ErrorAnalysisError
from predictive_maintenance.models import HIST_GRADIENT_BOOSTING_MODEL_NAME
from predictive_maintenance.splitting import SplitData


def make_error_dataframe() -> pd.DataFrame:
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


def make_validation_features(index: list[int] | None = None) -> pd.DataFrame:
    index = [10, 11, 12, 13] if index is None else index
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [300.0, 301.0, 302.0, 303.0],
            "Process temperature [K]": [310.0, 311.0, 312.0, 313.0],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1700],
            "Torque [Nm]": [40.0, 41.0, 42.0, 43.0],
            "Tool wear [min]": [10, 20, 30, 40],
            "UDI": [1, 2, 3, 4],
            "Product ID": ["a", "b", "c", "d"],
            "TWF": [0, 0, 1, 1],
            "HDF": [0, 0, 0, 0],
            "PWF": [0, 0, 0, 0],
            "OSF": [0, 0, 0, 0],
            "RNF": [0, 0, 0, 0],
        },
        index=index,
    )


def make_error_table() -> pd.DataFrame:
    X_validation = make_validation_features()
    y_validation = pd.Series([0, 0, 1, 1], index=X_validation.index)
    probabilities = np.array([0.1, 0.6, 0.4, 0.8])
    return build_error_analysis_table(
        X_validation,
        y_validation,
        probabilities,
        threshold=0.5,
    )


def test_classify_prediction_outcomes_returns_exact_labels_and_columns() -> None:
    y_true = pd.Series([0, 0, 1, 1], index=[10, 11, 12, 13])
    probabilities = np.array([0.1, 0.6, 0.4, 0.8])

    outcomes = classify_prediction_outcomes(y_true, probabilities, threshold=0.5)

    assert tuple(outcomes.columns) == OUTCOME_COLUMNS
    assert outcomes.index.tolist() == [10, 11, 12, 13]
    assert outcomes.to_dict(orient="list") == {
        "actual": [0, 0, 1, 1],
        "predicted": [0, 1, 0, 1],
        "probability": [0.1, 0.6, 0.4, 0.8],
        "outcome": [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
    }


def test_classify_prediction_outcomes_uses_greater_equal_threshold_rule() -> None:
    outcomes = classify_prediction_outcomes(
        pd.Series([0, 1]),
        np.array([0.5, 0.5]),
        threshold=0.5,
    )

    assert outcomes["predicted"].tolist() == [1, 1]
    assert outcomes["outcome"].tolist() == ["false_positive", "true_positive"]


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (0.0, "strictly between 0 and 1"),
        (1.0, "strictly between 0 and 1"),
        (-0.1, "strictly between 0 and 1"),
        (1.1, "strictly between 0 and 1"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        ("not-a-threshold", "numeric"),
    ],
)
def test_classify_prediction_outcomes_rejects_invalid_thresholds(
    threshold: Any,
    message: str,
) -> None:
    with pytest.raises(ErrorAnalysisError, match=message):
        classify_prediction_outcomes([0, 1], np.array([0.1, 0.9]), threshold=threshold)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([0.1], "same length"),
        ([np.nan, 0.2], "finite"),
        ([np.inf, 0.2], "finite"),
        ([-np.inf, 0.2], "finite"),
        ([-0.1, 0.2], "\\[0, 1\\]"),
        ([1.1, 0.2], "\\[0, 1\\]"),
    ],
)
def test_classify_prediction_outcomes_rejects_invalid_probabilities(
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(ErrorAnalysisError, match=message):
        classify_prediction_outcomes([0, 1], np.array(probabilities), threshold=0.5)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([0, 0], "exactly \\{0, 1\\}"),
        ([0, 2], "exactly \\{0, 1\\}"),
        (["0", "1"], "exactly \\{0, 1\\}"),
        ([0, np.nan], "missing"),
    ],
)
def test_classify_prediction_outcomes_rejects_invalid_targets(
    target: list[Any],
    message: str,
) -> None:
    with pytest.raises(ErrorAnalysisError, match=message):
        classify_prediction_outcomes(target, np.array([0.1, 0.9]), threshold=0.5)


def test_classify_prediction_outcomes_rejects_empty_inputs() -> None:
    with pytest.raises(ErrorAnalysisError, match="at least one row"):
        classify_prediction_outcomes([], np.array([]), threshold=0.5)


def test_classify_prediction_outcomes_does_not_mutate_inputs() -> None:
    y_true = pd.Series([0, 1], index=[10, 11])
    probabilities = np.array([0.2, 0.8])
    original_target = y_true.copy(deep=True)
    original_probabilities = probabilities.copy()

    classify_prediction_outcomes(y_true, probabilities, threshold=0.5)

    pd.testing.assert_series_equal(y_true, original_target)
    np.testing.assert_array_equal(probabilities, original_probabilities)


def test_build_error_analysis_table_uses_only_model_features_and_outcomes() -> None:
    X_validation = make_validation_features()
    y_validation = pd.Series([0, 0, 1, 1], index=X_validation.index)

    table = build_error_analysis_table(
        X_validation,
        y_validation,
        np.array([0.1, 0.6, 0.4, 0.8]),
        threshold=0.5,
    )

    assert tuple(table.columns) == (*config.MODEL_FEATURES, *OUTCOME_COLUMNS)
    assert "UDI" not in table.columns
    assert "Product ID" not in table.columns
    for column in config.FAILURE_MODE_COLUMNS:
        assert column not in table.columns
    assert table["outcome"].tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]


def test_build_error_analysis_table_preserves_non_default_indices() -> None:
    X_validation = make_validation_features(index=[101, 103, 107, 109])
    y_validation = pd.Series([0, 0, 1, 1], index=X_validation.index)

    table = build_error_analysis_table(
        X_validation,
        y_validation,
        np.array([0.1, 0.6, 0.4, 0.8]),
        threshold=0.5,
    )

    assert table.index.tolist() == [101, 103, 107, 109]


def test_build_error_analysis_table_allows_duplicate_aligned_indices() -> None:
    X_validation = make_validation_features(index=[10, 10, 12, 12])
    y_validation = pd.Series([0, 0, 1, 1], index=X_validation.index)

    table = build_error_analysis_table(
        X_validation,
        y_validation,
        np.array([0.1, 0.6, 0.4, 0.8]),
        threshold=0.5,
    )

    assert table.index.tolist() == [10, 10, 12, 12]
    assert table["outcome"].tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]


def test_build_error_analysis_table_rejects_mismatched_indices() -> None:
    X_validation = make_validation_features(index=[10, 11, 12, 13])
    y_validation = pd.Series([0, 0, 1, 1], index=[10, 11, 12, 99])

    with pytest.raises(ErrorAnalysisError, match="indices must match"):
        build_error_analysis_table(
            X_validation,
            y_validation,
            np.array([0.1, 0.6, 0.4, 0.8]),
            threshold=0.5,
        )


def test_build_error_analysis_table_does_not_mutate_source_inputs() -> None:
    X_validation = make_validation_features()
    y_validation = pd.Series([0, 0, 1, 1], index=X_validation.index)
    original_X = X_validation.copy(deep=True)
    original_y = y_validation.copy(deep=True)

    build_error_analysis_table(
        X_validation,
        y_validation,
        np.array([0.1, 0.6, 0.4, 0.8]),
        threshold=0.5,
    )

    pd.testing.assert_frame_equal(X_validation, original_X)
    pd.testing.assert_series_equal(y_validation, original_y)


def test_summarize_outcomes_returns_exact_counts_and_percentages() -> None:
    summary = summarize_outcomes(make_error_table())

    assert summary["true_negative"]["count"] == 1
    assert summary["false_positive"]["count"] == 1
    assert summary["false_negative"]["count"] == 1
    assert summary["true_positive"]["count"] == 1
    assert summary["false_positive"]["percentage"] == 25.0
    assert summary["false_positive"]["mean_probability"] == 0.6


def test_summarize_outcomes_handles_empty_error_table() -> None:
    empty_table = pd.DataFrame(columns=(*config.MODEL_FEATURES, *OUTCOME_COLUMNS))

    summary = summarize_outcomes(empty_table)

    assert summary["true_negative"]["count"] == 0
    assert summary["true_negative"]["percentage"] == 0.0
    assert summary["true_negative"]["mean_probability"] is None


def test_summarize_numerical_features_by_outcome_returns_expected_values() -> None:
    summary = summarize_numerical_features_by_outcome(make_error_table())

    torque_summary = summary["false_positive"]["Torque [Nm]"]
    assert torque_summary["mean"] == 41.0
    assert torque_summary["median"] == 41.0
    assert torque_summary["min"] == 41.0
    assert torque_summary["max"] == 41.0


def test_summarize_numerical_features_handles_all_missing_feature() -> None:
    table = make_error_table()
    table["Torque [Nm]"] = np.nan

    summary = summarize_numerical_features_by_outcome(table)

    assert summary["false_positive"]["Torque [Nm]"] == {
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
    }


def test_summarize_categorical_features_by_outcome_returns_distribution() -> None:
    summary = summarize_categorical_features_by_outcome(make_error_table())

    assert summary["true_negative"]["Type"] == {"L": {"count": 1, "percentage": 100.0}}
    assert summary["false_positive"]["Type"] == {"M": {"count": 1, "percentage": 100.0}}


def test_summarize_categorical_features_handles_all_missing_feature() -> None:
    table = make_error_table()
    table["Type"] = None

    summary = summarize_categorical_features_by_outcome(table)

    assert summary["false_positive"]["Type"] == {
        "None": {"count": 1, "percentage": 100.0}
    }


def test_summaries_handle_no_false_positives_and_no_false_negatives() -> None:
    table = build_error_analysis_table(
        make_validation_features(),
        pd.Series([0, 0, 1, 1], index=[10, 11, 12, 13]),
        np.array([0.1, 0.2, 0.7, 0.8]),
        threshold=0.5,
    )

    fp_summary = summarize_false_positives(table, top_n=3)
    fn_summary = summarize_false_negatives(table, top_n=3)

    assert fp_summary["count"] == 0
    assert fn_summary["count"] == 0
    assert fp_summary["probability_summary"]["mean"] is None
    assert fn_summary["top_false_negatives"] == []


def test_focused_error_summary_uses_feature_wise_differences_without_ranking() -> None:
    fp_summary = summarize_false_positives(make_error_table(), top_n=3)

    assert "strongest_numerical_differences_vs_correct" not in fp_summary
    differences = fp_summary["feature_wise_mean_differences_vs_correct"]
    assert [row["feature"] for row in differences] == list(config.NUMERICAL_FEATURES)
    assert all("absolute_difference" not in row for row in differences)
    assert all("native units" in row["unit_note"] for row in differences)
    json.dumps(differences, allow_nan=False)


def test_focused_error_summaries_return_top_n_in_deterministic_order() -> None:
    table = build_error_analysis_table(
        make_validation_features(index=[10, 11, 12, 13]),
        pd.Series([0, 0, 1, 1], index=[10, 11, 12, 13]),
        np.array([0.9, 0.7, 0.2, 0.1]),
        threshold=0.5,
    )

    fp_summary = summarize_false_positives(table, top_n=1)
    fn_summary = summarize_false_negatives(table, top_n=1)

    assert fp_summary["top_false_positives"][0]["probability"] == 0.9
    assert fn_summary["top_false_negatives"][0]["probability"] == 0.2


def test_summarize_false_positives_rejects_invalid_top_n() -> None:
    with pytest.raises(ErrorAnalysisError, match="top_n"):
        summarize_false_positives(make_error_table(), top_n=0)


def test_summaries_are_json_serializable_when_all_predictions_wrong() -> None:
    table = build_error_analysis_table(
        make_validation_features(),
        pd.Series([0, 0, 1, 1], index=[10, 11, 12, 13]),
        np.array([0.9, 0.8, 0.2, 0.1]),
        threshold=0.5,
    )

    payload = {
        "outcomes": summarize_outcomes(table),
        "numerical": summarize_numerical_features_by_outcome(table),
        "categorical": summarize_categorical_features_by_outcome(table),
        "fp": summarize_false_positives(table),
        "fn": summarize_false_negatives(table),
    }

    json.dumps(payload, allow_nan=False)


def test_run_validation_error_analysis_returns_validation_only_output() -> None:
    results = run_validation_error_analysis(make_error_dataframe(), top_n=3)

    assert results["model"] == HIST_GRADIENT_BOOSTING_MODEL_NAME
    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert tuple(results["thresholds_analyzed"]) == (
        "default_0_5",
        "best_f1",
        "minimum_recall",
        "lowest_cost",
    )
    assert "test_metrics" not in results
    assert "untouched" in results["test_set_status"]
    json.dumps({key: value for key, value in results.items() if key != "error_cases"})
    _assert_no_model_objects(results)


def test_run_validation_error_analysis_uses_provided_dataframe_without_disk_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load_dataset():
        raise AssertionError("load_dataset should not be called")

    monkeypatch.setattr(error_analysis, "load_dataset", fail_load_dataset)

    results = run_validation_error_analysis(make_error_dataframe())

    assert results["split_row_counts"]["train"] == 60


def test_run_validation_error_analysis_does_not_mutate_source_dataframe() -> None:
    dataframe = make_error_dataframe()
    original = dataframe.copy(deep=True)

    run_validation_error_analysis(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_run_validation_error_analysis_scores_only_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = error_analysis.split_dataset(make_error_dataframe())
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
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ControlledModel()},
    )

    run_validation_error_analysis(make_error_dataframe())

    assert scored_indices == [tuple(split_data.X_validation.index)]


def test_run_validation_error_analysis_fits_only_training_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = error_analysis.split_dataset(make_error_dataframe())
    observed_fit = {}

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
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ControlledModel()},
    )

    run_validation_error_analysis(make_error_dataframe())

    assert observed_fit == {
        "X_index": tuple(split_data.X_train.index),
        "y_index": tuple(split_data.y_train.index),
        "sample_weight_length": len(split_data.y_train),
    }


def test_run_validation_error_analysis_uses_custom_thresholds_in_sorted_order() -> None:
    results = run_validation_error_analysis(
        make_error_dataframe(),
        thresholds=[0.8, 0.2, 0.2],
    )

    assert results["thresholds_analyzed"] == {
        "threshold_0_20": 0.2,
        "threshold_0_80": 0.8,
    }


def test_run_validation_error_analysis_wraps_probability_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProbabilityModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            raise RuntimeError("probability failure")

    monkeypatch.setattr(
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: FailingProbabilityModel()},
    )

    with pytest.raises(ErrorAnalysisError, match="validation probabilities") as exc:
        run_validation_error_analysis(make_error_dataframe())

    assert isinstance(exc.value.__cause__, RuntimeError)


def test_run_validation_error_analysis_wraps_model_fit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingFitModel:
        def fit(self, X, y, **fit_parameters):
            raise RuntimeError("fit failure")

    monkeypatch.setattr(
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: FailingFitModel()},
    )

    with pytest.raises(ErrorAnalysisError, match="fit selected model") as exc:
        run_validation_error_analysis(make_error_dataframe())

    assert isinstance(exc.value.__cause__, RuntimeError)
    assert str(exc.value.__cause__) == "fit failure"


def test_run_validation_error_analysis_wraps_missing_class_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingClassOneModel:
        classes_ = np.array([0, 2])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.column_stack(
                [
                    np.full(len(X), 0.7),
                    np.full(len(X), 0.3),
                ]
            )

    monkeypatch.setattr(
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: MissingClassOneModel()},
    )

    with pytest.raises(ErrorAnalysisError, match="class-1") as exc:
        run_validation_error_analysis(make_error_dataframe())

    assert exc.value.__cause__ is not None


def test_run_validation_error_analysis_rejects_unknown_model_name() -> None:
    with pytest.raises(ErrorAnalysisError, match="Unknown model_name"):
        run_validation_error_analysis(make_error_dataframe(), model_name="missing")


def test_save_error_analysis_report_writes_reloadable_json_and_csv(
    tmp_path: Path,
) -> None:
    results = run_validation_error_analysis(make_error_dataframe(), top_n=2)

    paths = save_error_analysis_report(results, tmp_path)

    assert paths["error_analysis_json"] == tmp_path / "error_analysis.json"
    assert paths["error_cases_csv"] == tmp_path / "error_cases.csv"
    saved_json = json.loads(paths["error_analysis_json"].read_text(encoding="utf-8"))
    saved_csv = pd.read_csv(paths["error_cases_csv"])
    assert "error_cases" not in saved_json
    assert tuple(saved_csv.columns) == ERROR_CASE_COLUMNS
    assert set(saved_csv["outcome"]).issubset({"false_positive", "false_negative"})


def test_run_validation_error_analysis_with_controlled_split_threshold_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data = SplitData(
        X_train=make_validation_features(index=[1, 2, 3, 4]).loc[
            :,
            config.MODEL_FEATURES,
        ],
        X_validation=make_validation_features(index=[10, 11, 12, 13]).loc[
            :,
            config.MODEL_FEATURES,
        ],
        X_test=make_validation_features(index=[20, 21, 22, 23]).loc[
            :,
            config.MODEL_FEATURES,
        ],
        y_train=pd.Series([0, 0, 1, 1], index=[1, 2, 3, 4]),
        y_validation=pd.Series([0, 0, 1, 1], index=[10, 11, 12, 13]),
        y_test=pd.Series([0, 0, 1, 1], index=[20, 21, 22, 23]),
    )

    class ControlledModel:
        classes_ = np.array([1, 0])

        def fit(self, X, y, **fit_parameters):
            return self

        def predict_proba(self, X):
            return np.array(
                [
                    [0.1, 0.9],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.8, 0.2],
                ]
            )

    monkeypatch.setattr(error_analysis, "split_dataset", lambda dataframe: split_data)
    monkeypatch.setattr(
        error_analysis,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: ControlledModel()},
    )

    results = run_validation_error_analysis(make_error_dataframe(), thresholds=[0.5])

    summary = results["per_threshold"]["threshold_0_50"]
    assert summary["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert summary["predicted_positive_count"] == 2
    assert summary["false_positive_rate"] == 25.0
    assert summary["false_negative_rate_actual_positives"] == 50.0


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
