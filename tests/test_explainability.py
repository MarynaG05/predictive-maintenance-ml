import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from predictive_maintenance import config, explainability
from predictive_maintenance.exceptions import ExplainabilityError
from predictive_maintenance.explainability import (
    FEATURE_IMPORTANCE_COLUMNS,
    calculate_permutation_importance,
    interpret_feature_importance,
    plot_feature_importance,
    rank_feature_importance,
    run_validation_explainability,
    save_explainability_outputs,
    save_explainability_report,
)
from predictive_maintenance.models import HIST_GRADIENT_BOOSTING_MODEL_NAME


def make_explainability_dataframe() -> pd.DataFrame:
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


def test_calculate_permutation_importance_uses_complete_pipeline_and_raw_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES].copy()
    y_validation = dataframe.loc[:, config.TARGET_COLUMN].copy()
    captured: dict[str, Any] = {}
    model = object()

    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        captured["estimator"] = estimator
        captured["X"] = X.copy()
        captured["y"] = y.copy()
        captured["scoring"] = scoring
        captured["n_repeats"] = n_repeats
        captured["random_state"] = random_state
        captured["n_jobs"] = n_jobs
        return SimpleNamespace(
            importances_mean=np.array([0.01, -0.02, 0.03, 0.04, 0.05, 0.06]),
            importances_std=np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006]),
        )

    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )

    importance = calculate_permutation_importance(
        model,
        X_validation,
        y_validation,
        n_repeats=7,
    )

    assert captured["estimator"] is model
    assert tuple(captured["X"].columns) == config.MODEL_FEATURES
    assert captured["X"].equals(X_validation)
    assert captured["y"].equals(y_validation)
    assert captured["scoring"] == "average_precision"
    assert captured["n_repeats"] == 7
    assert captured["random_state"] == config.RANDOM_SEED
    assert captured["n_jobs"] == 1
    assert {row["feature"] for row in importance} == set(config.MODEL_FEATURES)
    assert sum(row["feature"] == "Type" for row in importance) == 1
    assert any(row["importance_mean"] < 0 for row in importance)
    assert all("categorical__" not in row["feature"] for row in importance)
    assert all("Type_" not in row["feature"] for row in importance)


def test_calculate_permutation_importance_ignores_unexpected_extra_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES].copy()
    X_validation["extra_column"] = 1
    y_validation = dataframe.loc[:, config.TARGET_COLUMN].copy()
    captured = {}

    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        captured["columns"] = tuple(X.columns)
        return SimpleNamespace(
            importances_mean=np.zeros(len(config.MODEL_FEATURES)),
            importances_std=np.zeros(len(config.MODEL_FEATURES)),
        )

    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )

    importance = calculate_permutation_importance(
        object(),
        X_validation,
        y_validation,
        n_repeats=2,
    )

    assert captured["columns"] == config.MODEL_FEATURES
    assert [row["feature"] for row in importance] == sorted(config.MODEL_FEATURES)


def test_calculate_permutation_importance_rejects_missing_model_feature() -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES].drop(columns=["Torque [Nm]"])
    y_validation = dataframe.loc[:, config.TARGET_COLUMN]

    with pytest.raises(
        ExplainabilityError, match="Missing required validation feature"
    ):
        calculate_permutation_importance(
            object(),
            X_validation,
            y_validation,
        )


def test_calculate_permutation_importance_rejects_misaligned_indices() -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES]
    y_validation = dataframe.loc[:, config.TARGET_COLUMN].copy()
    y_validation.index = range(1, len(y_validation) + 1)

    with pytest.raises(ExplainabilityError, match="indices must match"):
        calculate_permutation_importance(
            object(),
            X_validation,
            y_validation,
        )


def test_calculate_permutation_importance_rejects_invalid_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES]
    y_validation = dataframe.loc[:, config.TARGET_COLUMN]

    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        return SimpleNamespace(
            importances_mean=np.array([0.1]),
            importances_std=np.array([0.01]),
        )

    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )

    with pytest.raises(ExplainabilityError, match="does not match"):
        calculate_permutation_importance(
            object(),
            X_validation,
            y_validation,
        )


def test_calculate_permutation_importance_wraps_scoring_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES]
    y_validation = dataframe.loc[:, config.TARGET_COLUMN]

    def fail_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        raise ValueError("scoring failed")

    monkeypatch.setattr(
        explainability,
        "permutation_importance",
        fail_permutation_importance,
    )

    with pytest.raises(ExplainabilityError, match="Unable to calculate") as exc_info:
        calculate_permutation_importance(
            object(),
            X_validation,
            y_validation,
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_calculate_permutation_importance_does_not_mutate_validation_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    X_validation = dataframe.loc[:, config.MODEL_FEATURES].copy()
    original = X_validation.copy(deep=True)
    y_validation = dataframe.loc[:, config.TARGET_COLUMN]

    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        X.iloc[0, 0] = "H"
        return SimpleNamespace(
            importances_mean=np.zeros(len(config.MODEL_FEATURES)),
            importances_std=np.zeros(len(config.MODEL_FEATURES)),
        )

    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )

    calculate_permutation_importance(object(), X_validation, y_validation)

    pd.testing.assert_frame_equal(X_validation, original)


def test_rank_feature_importance_is_deterministic() -> None:
    ranked = rank_feature_importance(
        [
            {"feature": "b_feature", "importance_mean": 0.2, "importance_std": 0.0},
            {"feature": "a_feature", "importance_mean": 0.2, "importance_std": 0.1},
            {"feature": "c_feature", "importance_mean": 0.4, "importance_std": 0.2},
        ]
    )

    assert [row["feature"] for row in ranked] == ["c_feature", "a_feature", "b_feature"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_interpret_feature_importance_is_json_serializable_and_non_causal() -> None:
    ranked = [
        {
            "rank": 1,
            "feature": "Torque [Nm]",
            "importance_mean": 0.3,
            "importance_std": 0.1,
        }
    ]

    interpretation = interpret_feature_importance(ranked)

    json.dumps(interpretation, allow_nan=False)
    combined_text = " ".join(interpretation[0].values())
    assert "causality" in combined_text
    assert "effect direction" in combined_text
    assert "validation-split-specific" in combined_text
    assert "low importance does not prove a feature is useless" in combined_text
    assert "production-ready causal conclusions" in combined_text


def test_run_validation_explainability_returns_validation_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_permutation_importance(monkeypatch)

    results = run_validation_explainability(
        make_explainability_dataframe(),
        n_repeats=2,
    )

    assert results["model_name"] == HIST_GRADIENT_BOOSTING_MODEL_NAME
    assert results["scoring_metric"] == "average_precision"
    assert results["n_repeats"] == 2
    assert results["random_seed"] == config.RANDOM_SEED
    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert isinstance(results["baseline_validation_average_precision"], float)
    assert "test_metrics" not in results
    assert "raw_probabilities" not in results
    assert "untouched" in results["test_set_status"]
    assert {row["feature"] for row in results["feature_importance"]} == set(
        config.MODEL_FEATURES
    )
    json.dumps(results, allow_nan=False)
    _assert_no_model_objects(results)


def test_run_validation_explainability_uses_provided_dataframe_without_disk_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_permutation_importance(monkeypatch)

    def fail_load_dataset():
        raise AssertionError("load_dataset should not be called")

    monkeypatch.setattr(explainability, "load_dataset", fail_load_dataset)

    results = run_validation_explainability(
        make_explainability_dataframe(),
        n_repeats=1,
    )

    assert results["split_row_counts"]["train"] == 60


def test_run_validation_explainability_uses_train_fit_and_validation_permutation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_explainability_dataframe()
    split_data = explainability.split_dataset(dataframe)
    observed: dict[str, Any] = {}

    class ControlledPreprocessor:
        def fit_transform(self, X, y=None):
            observed["preprocessor_fit_transform_index"] = tuple(X.index)
            return X[["Torque [Nm]"]].to_numpy()

        def fit(self, X, y=None):
            observed["preprocessor_fit_index"] = tuple(X.index)
            return self

        def transform(self, X):
            observed.setdefault("transform_indices", []).append(tuple(X.index))
            return X[["Torque [Nm]"]].to_numpy()

    class ControlledClassifier(BaseEstimator):
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.array([0, 1])
            self.is_fitted_ = True
            observed["classifier_y_index"] = tuple(y.index)
            observed["sample_weight"] = np.asarray(sample_weight)
            return self

        def predict_proba(self, X):
            probabilities = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([1.0 - probabilities, probabilities])

    model = Pipeline(
        [
            ("preprocessor", ControlledPreprocessor()),
            ("classifier", ControlledClassifier()),
        ]
    )

    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        observed["permutation_estimator"] = estimator
        observed["permutation_X_index"] = tuple(X.index)
        observed["permutation_y_index"] = tuple(y.index)
        observed["permutation_columns"] = tuple(X.columns)
        return SimpleNamespace(
            importances_mean=np.zeros(len(config.MODEL_FEATURES)),
            importances_std=np.zeros(len(config.MODEL_FEATURES)),
        )

    monkeypatch.setattr(
        explainability,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: model},
    )
    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )

    run_validation_explainability(dataframe, n_repeats=1)

    assert observed["preprocessor_fit_transform_index"] == tuple(
        split_data.X_train.index
    )
    assert observed["classifier_y_index"] == tuple(split_data.y_train.index)
    assert observed["permutation_estimator"] is model
    assert observed["permutation_X_index"] == tuple(split_data.X_validation.index)
    assert observed["permutation_y_index"] == tuple(split_data.y_validation.index)
    assert observed["permutation_columns"] == config.MODEL_FEATURES
    assert tuple(split_data.X_test.index) not in observed.get("transform_indices", [])
    np.testing.assert_allclose(
        observed["sample_weight"],
        compute_sample_weight(class_weight="balanced", y=split_data.y_train),
    )


def test_run_validation_explainability_wraps_model_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def fit(self, X, y, **kwargs):
            raise ValueError("fit failed")

    monkeypatch.setattr(
        explainability,
        "build_model_comparison_models",
        lambda: {HIST_GRADIENT_BOOSTING_MODEL_NAME: FailingModel()},
    )

    with pytest.raises(ExplainabilityError, match="Unable to fit model") as exc_info:
        run_validation_explainability(make_explainability_dataframe())
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_save_explainability_report_writes_reloadable_json_and_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_permutation_importance(monkeypatch)
    results = run_validation_explainability(
        make_explainability_dataframe(),
        n_repeats=1,
    )

    paths = save_explainability_report(results, tmp_path)

    saved_json = json.loads(paths["feature_importance_json"].read_text("utf-8"))
    saved_csv = pd.read_csv(paths["feature_importance_csv"])
    assert tuple(saved_json) == (
        "model_name",
        "scoring_metric",
        "n_repeats",
        "random_seed",
        "baseline_validation_average_precision",
        "split_row_counts",
        "feature_importance",
        "interpretation",
        "test_set_status",
    )
    assert tuple(saved_csv.columns) == FEATURE_IMPORTANCE_COLUMNS
    assert saved_csv["feature"].tolist() == [
        row["feature"] for row in results["feature_importance"]
    ]


def test_plot_feature_importance_uses_uncertainty_bars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "importance.png"
    observed: dict[str, Any] = {}

    class FakeFigure:
        def tight_layout(self):
            observed["tight_layout"] = True

        def savefig(self, path, dpi):
            observed["savefig_path"] = path
            observed["dpi"] = dpi
            Path(path).write_bytes(b"png")

    class FakeAxis:
        def barh(self, labels, values, *, xerr, capsize):
            observed["labels"] = labels
            observed["values"] = values
            observed["xerr"] = xerr
            observed["capsize"] = capsize

        def axvline(self, *args, **kwargs):
            observed["axvline"] = args

        def set_title(self, title):
            observed["title"] = title

        def set_xlabel(self, label):
            observed["xlabel"] = label

        def set_ylabel(self, label):
            observed["ylabel"] = label

    monkeypatch.setattr(
        explainability.plt, "subplots", lambda figsize: (FakeFigure(), FakeAxis())
    )
    monkeypatch.setattr(explainability.plt, "close", lambda fig: None)

    path = plot_feature_importance(
        [
            {
                "rank": 1,
                "feature": "Torque [Nm]",
                "importance_mean": 0.3,
                "importance_std": 0.1,
            },
            {
                "rank": 2,
                "feature": "Type",
                "importance_mean": 0.2,
                "importance_std": 0.05,
            },
        ],
        output_path,
    )

    assert path == output_path
    assert output_path.is_file()
    assert observed["xerr"] == [0.05, 0.1]
    assert observed["capsize"] == 3
    assert "Validation Permutation" in observed["title"]
    assert "Average precision" in observed["xlabel"]


def test_save_explainability_outputs_writes_reports_and_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_permutation_importance(monkeypatch)
    results = run_validation_explainability(
        make_explainability_dataframe(),
        n_repeats=1,
    )
    figure_path = tmp_path / "figure.png"

    paths = save_explainability_outputs(results, tmp_path, figure_path)

    assert paths["feature_importance_json"].is_file()
    assert paths["feature_importance_csv"].is_file()
    assert paths["permutation_importance_png"].is_file()


def test_run_validation_explainability_rejects_unknown_model_name() -> None:
    with pytest.raises(ExplainabilityError, match="Unknown model_name"):
        run_validation_explainability(
            make_explainability_dataframe(),
            model_name="missing_model",
        )


def test_encoded_aggregation_api_is_removed() -> None:
    assert not hasattr(explainability, "aggregate_encoded_importance")


def _patch_permutation_importance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):
        return SimpleNamespace(
            importances_mean=np.linspace(0.01, 0.06, len(config.MODEL_FEATURES)),
            importances_std=np.linspace(0.001, 0.006, len(config.MODEL_FEATURES)),
        )

    monkeypatch.setattr(
        explainability, "permutation_importance", fake_permutation_importance
    )


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
