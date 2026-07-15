import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from predictive_maintenance import config, recommendations
from predictive_maintenance.recommendations import (
    PROFILE_CSV_COLUMNS,
    PROFILE_ORDER,
    build_threshold_recommendations,
    plot_business_threshold_profiles,
    run_business_threshold_recommendation,
    save_business_threshold_outputs,
    save_business_threshold_recommendation,
)


def make_threshold_analysis(
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model": "hist_gradient_boosting_balanced",
        "split_row_counts": {"train": 60, "validation": 20, "test": 20},
        "threshold_results": rows if rows is not None else make_threshold_rows(),
        "test_set_status": (
            "Final test set remains untouched; no test-set metrics were calculated."
        ),
    }


def make_threshold_rows() -> list[dict[str, Any]]:
    return [
        threshold_row(
            threshold=0.1,
            tn=10,
            fp=7,
            fn=1,
            tp=9,
        ),
        threshold_row(
            threshold=0.5,
            tn=15,
            fp=2,
            fn=5,
            tp=5,
        ),
        threshold_row(
            threshold=0.8,
            tn=16,
            fp=1,
            fn=4,
            tp=6,
        ),
    ]


def threshold_row(
    threshold: float,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, Any]:
    predicted_positive_count = fp + tp
    total_count = tn + fp + fn + tp
    actual_positive_count = fn + tp
    precision = tp / predicted_positive_count if predicted_positive_count else 0.0
    recall = tp / actual_positive_count if actual_positive_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tn + tp) / total_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_count / total_count,
        "total_cost": float(fn * 10 + fp),
    }


def make_recommendation_dataframe() -> pd.DataFrame:
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


def test_build_threshold_recommendations_selects_ordered_profiles() -> None:
    result = build_threshold_recommendations(make_threshold_analysis())

    assert tuple(result["profiles"]) == PROFILE_ORDER
    assert [row["profile"] for row in result["comparison_table"]] == list(PROFILE_ORDER)
    assert result["selected_model"] == "hist_gradient_boosting_balanced"
    assert result["profiles"]["high_recall"]["threshold"] == 0.1
    assert result["profiles"]["balanced"]["threshold"] == 0.8
    assert result["profiles"]["conservative"]["threshold"] == 0.8
    json.dumps(result, allow_nan=False)
    _assert_no_model_objects(result)


def test_high_recall_target_satisfied_uses_deterministic_ties() -> None:
    rows = [
        threshold_row(0.1, 12, 5, 1, 9),
        threshold_row(0.2, 14, 3, 1, 9),
        threshold_row(0.3, 15, 2, 1, 9),
    ]

    result = build_threshold_recommendations(make_threshold_analysis(rows))

    profile = result["profiles"]["high_recall"]
    assert profile["threshold"] == 0.3
    assert profile["recall_target_met"] is True


def test_high_recall_target_not_satisfied_falls_back_to_highest_recall() -> None:
    rows = [
        threshold_row(0.1, 12, 5, 4, 6),
        threshold_row(0.2, 13, 4, 2, 8),
    ]

    result = build_threshold_recommendations(
        make_threshold_analysis(rows),
        minimum_recall=0.95,
    )

    profile = result["profiles"]["high_recall"]
    assert profile["threshold"] == 0.2
    assert profile["recall_target_met"] is False
    assert "target was not met" in profile["limitations"]


def test_balanced_profile_uses_best_f1_tie_breaking() -> None:
    rows = [
        threshold_row(0.3, 14, 2, 2, 8),
        threshold_row(0.6, 14, 2, 2, 8),
    ]

    result = build_threshold_recommendations(make_threshold_analysis(rows))

    assert result["profiles"]["balanced"]["threshold"] == 0.6


def test_conservative_precision_target_satisfied_maximizes_recall() -> None:
    rows = [
        threshold_row(0.4, 15, 2, 4, 6),
        threshold_row(0.6, 16, 1, 5, 5),
    ]

    result = build_threshold_recommendations(make_threshold_analysis(rows))

    profile = result["profiles"]["conservative"]
    assert profile["threshold"] == 0.4
    assert profile["precision_target_met"] is True


def test_conservative_target_not_satisfied_uses_highest_precision() -> None:
    rows = [
        threshold_row(0.4, 11, 6, 1, 9),
        threshold_row(0.6, 14, 3, 5, 5),
    ]

    result = build_threshold_recommendations(
        make_threshold_analysis(rows),
        minimum_precision=0.9,
    )

    profile = result["profiles"]["conservative"]
    assert profile["threshold"] == 0.6
    assert profile["precision_target_met"] is False
    assert "target was not met" in profile["limitations"]


@pytest.mark.parametrize(
    ("minimum_recall", "minimum_precision"),
    [
        (-0.1, 0.7),
        (1.1, 0.7),
        (np.nan, 0.7),
        (np.inf, 0.7),
        (0.85, -0.1),
        (0.85, 1.1),
        (0.85, np.nan),
        (0.85, np.inf),
    ],
)
def test_build_threshold_recommendations_rejects_invalid_constraints(
    minimum_recall: float,
    minimum_precision: float,
) -> None:
    with pytest.raises(ValueError, match="finite and within"):
        build_threshold_recommendations(
            make_threshold_analysis(),
            minimum_recall=minimum_recall,
            minimum_precision=minimum_precision,
        )


@pytest.mark.parametrize(
    "analysis",
    [
        {},
        {"model": "model", "threshold_results": []},
        {"model": "model", "split_row_counts": {}, "threshold_results": []},
    ],
)
def test_build_threshold_recommendations_rejects_invalid_analysis_structure(
    analysis: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        build_threshold_recommendations(analysis)


def test_build_threshold_recommendations_rejects_missing_metric_fields() -> None:
    row = make_threshold_rows()[0]
    row.pop("total_cost")

    with pytest.raises(ValueError, match="missing fields"):
        build_threshold_recommendations(make_threshold_analysis([row]))


def test_build_threshold_recommendations_rejects_duplicated_thresholds() -> None:
    rows = [make_threshold_rows()[0], dict(make_threshold_rows()[0])]

    with pytest.raises(ValueError, match="Duplicate threshold"):
        build_threshold_recommendations(make_threshold_analysis(rows))


def test_convergence_metadata_reports_balanced_conservative_shared() -> None:
    result = build_threshold_recommendations(make_threshold_analysis())

    assert result["profiles_share_threshold"] is True
    assert result["shared_threshold_groups"] == [
        {
            "threshold": 0.8,
            "profiles": ["balanced", "conservative"],
            "explanation": (
                "Distinct selection rules converged on the same validation "
                "operating point. The profile objectives remain different."
            ),
        }
    ]
    assert (
        "Identical thresholds do not mean"
        in result["decision_guidance"]["profile_convergence"]
    )


def test_convergence_metadata_supports_all_profiles_sharing_one_threshold() -> None:
    rows = [threshold_row(0.5, 17, 0, 0, 10)]

    result = build_threshold_recommendations(make_threshold_analysis(rows))

    assert result["profiles_share_threshold"] is True
    assert result["shared_threshold_groups"][0]["profiles"] == list(PROFILE_ORDER)


def test_convergence_metadata_reports_no_shared_thresholds() -> None:
    rows = [
        threshold_row(0.1, 7, 10, 0, 10),
        threshold_row(0.5, 15, 2, 3, 7),
        threshold_row(0.9, 17, 0, 6, 4),
    ]

    result = build_threshold_recommendations(
        make_threshold_analysis(rows),
        minimum_recall=0.95,
        minimum_precision=0.99,
    )

    assert result["profiles_share_threshold"] is False
    assert result["shared_threshold_groups"] == []


def test_business_profile_contents_copy_metrics_and_rates() -> None:
    result = build_threshold_recommendations(make_threshold_analysis())

    profile = result["profiles"]["high_recall"]

    assert profile["true_negative"] == 10
    assert profile["false_positive"] == 7
    assert profile["false_negative"] == 1
    assert profile["true_positive"] == 9
    assert profile["predicted_positive_count"] == 16
    assert profile["false_positive_rate_overall"] == pytest.approx(7 / 27)
    assert profile["false_negative_rate_among_actual_positives"] == pytest.approx(
        1 / 10
    )
    assert profile["illustrative_total_cost"] == 17.0
    assert profile["business_objective"]
    assert profile["tradeoff_summary"]
    assert profile["recommended_when"]
    assert profile["limitations"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predicted_positive_count", 999, "predicted_positive_count"),
        ("predicted_positive_rate", 0.99, "predicted_positive_rate"),
        ("precision", 0.99, "precision"),
        ("recall", 0.99, "recall"),
        ("f1", 0.99, "f1"),
        ("true_negative", -1, "cannot be negative"),
        ("false_positive", 1.5, "must be an integer"),
    ],
)
def test_build_threshold_recommendations_rejects_inconsistent_threshold_rows(
    field: str,
    value: Any,
    message: str,
) -> None:
    row = make_threshold_rows()[0]
    row[field] = value

    with pytest.raises(ValueError, match=message):
        build_threshold_recommendations(make_threshold_analysis([row]))


def test_build_threshold_recommendations_rejects_inconsistent_row_totals() -> None:
    rows = make_threshold_rows()
    rows[1]["true_negative"] += 1
    rows[1]["predicted_positive_rate"] = rows[1]["predicted_positive_count"] / (
        rows[1]["true_negative"]
        + rows[1]["false_positive"]
        + rows[1]["false_negative"]
        + rows[1]["true_positive"]
    )

    with pytest.raises(ValueError, match="totals must be consistent"):
        build_threshold_recommendations(make_threshold_analysis(rows))


def test_output_contains_no_raw_probabilities_or_test_metrics() -> None:
    result = build_threshold_recommendations(make_threshold_analysis())
    encoded = json.dumps(result, allow_nan=False)

    assert "raw_probabilities" not in encoded
    assert "test_metrics" not in encoded
    assert "test-set predictions" in result["test_set_status"]


def test_run_business_threshold_recommendation_delegates_validation_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = make_recommendation_dataframe()
    original = dataframe.copy(deep=True)
    observed: dict[str, Any] = {}

    def fake_run_threshold_analysis(
        dataframe=None,
        model_name="hist_gradient_boosting_balanced",
        minimum_recall=0.0,
        false_negative_cost=10.0,
        false_positive_cost=1.0,
    ):
        observed["dataframe_is_copy"] = dataframe is not original
        observed["columns"] = tuple(dataframe.columns)
        observed["model_name"] = model_name
        observed["minimum_recall"] = minimum_recall
        observed["false_negative_cost"] = false_negative_cost
        observed["false_positive_cost"] = false_positive_cost
        dataframe.iloc[0, dataframe.columns.get_loc("Type")] = "H"
        return make_threshold_analysis()

    monkeypatch.setattr(
        recommendations,
        "run_threshold_analysis",
        fake_run_threshold_analysis,
    )

    result = run_business_threshold_recommendation(
        dataframe,
        minimum_recall=0.85,
        minimum_precision=0.70,
        false_negative_cost=12.0,
        false_positive_cost=2.0,
    )

    assert observed == {
        "dataframe_is_copy": True,
        "columns": config.REQUIRED_COLUMNS,
        "model_name": "hist_gradient_boosting_balanced",
        "minimum_recall": 0.0,
        "false_negative_cost": 12.0,
        "false_positive_cost": 2.0,
    }
    pd.testing.assert_frame_equal(dataframe, original)
    assert result["selected_model"] == "hist_gradient_boosting_balanced"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_recall": np.nan},
        {"minimum_precision": np.inf},
        {"false_negative_cost": 0.0},
        {"false_positive_cost": -1.0},
        {"false_negative_cost": 1.0, "false_positive_cost": 1.0},
    ],
)
def test_run_business_threshold_recommendation_rejects_invalid_inputs_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    def fail_run_threshold_analysis(**workflow_kwargs):
        raise AssertionError("threshold workflow should not be called")

    monkeypatch.setattr(
        recommendations,
        "run_threshold_analysis",
        fail_run_threshold_analysis,
    )

    with pytest.raises(ValueError):
        run_business_threshold_recommendation(
            make_recommendation_dataframe(),
            **kwargs,
        )


def test_run_business_threshold_recommendation_avoids_disk_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_threshold_analysis(**kwargs):
        assert kwargs["dataframe"] is not None
        return make_threshold_analysis()

    monkeypatch.setattr(
        recommendations,
        "run_threshold_analysis",
        fake_run_threshold_analysis,
    )

    result = run_business_threshold_recommendation(make_recommendation_dataframe())

    assert result["validation_assumptions"]["split_row_counts"]["validation"] == 20


def test_run_business_threshold_recommendation_propagates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run_threshold_analysis(**kwargs):
        raise RuntimeError("threshold workflow failed")

    monkeypatch.setattr(
        recommendations,
        "run_threshold_analysis",
        fail_run_threshold_analysis,
    )

    with pytest.raises(RuntimeError, match="threshold workflow failed"):
        run_business_threshold_recommendation(make_recommendation_dataframe())


def test_save_business_threshold_recommendation_writes_reloadable_json_and_csv(
    tmp_path: Path,
) -> None:
    result = build_threshold_recommendations(make_threshold_analysis())

    paths = save_business_threshold_recommendation(result, tmp_path)

    saved_json = json.loads(paths["recommendation_json"].read_text("utf-8"))
    saved_csv = pd.read_csv(paths["recommendation_csv"])
    assert tuple(saved_json["profiles"]) == PROFILE_ORDER
    assert tuple(saved_csv.columns) == PROFILE_CSV_COLUMNS
    assert saved_csv["profile"].tolist() == list(PROFILE_ORDER)


def test_plot_business_threshold_profiles_creates_figure(tmp_path: Path) -> None:
    result = build_threshold_recommendations(make_threshold_analysis())
    output_path = tmp_path / "profiles.png"

    path = plot_business_threshold_profiles(result, output_path)

    assert path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_business_threshold_profiles_notes_shared_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = build_threshold_recommendations(make_threshold_analysis())
    observed: dict[str, Any] = {}
    output_path = tmp_path / "profiles.png"

    class FakeFigure:
        def tight_layout(self):
            return None

        def savefig(self, path, dpi):
            Path(path).write_bytes(b"png")

    class FakeAxis:
        transAxes = object()

        def bar(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            observed["annotation"] = args[2]

        def set_title(self, title):
            observed["title"] = title

        def set_xlabel(self, label):
            return None

        def set_ylabel(self, label):
            return None

        def set_xticks(self, ticks):
            return None

        def set_xticklabels(self, labels):
            observed["labels"] = list(labels)

        def set_ylim(self, low, high):
            return None

        def legend(self):
            return None

    monkeypatch.setattr(
        recommendations.plt,
        "subplots",
        lambda figsize: (FakeFigure(), FakeAxis()),
    )
    monkeypatch.setattr(recommendations.plt, "close", lambda fig: None)

    plot_business_threshold_profiles(result, output_path)

    assert "share the same validation threshold" in observed["annotation"]
    assert observed["labels"] == list(PROFILE_ORDER)
    assert output_path.is_file()


def test_save_business_threshold_outputs_writes_reports_and_figure(
    tmp_path: Path,
) -> None:
    result = build_threshold_recommendations(make_threshold_analysis())

    paths = save_business_threshold_outputs(
        result,
        output_dir=tmp_path,
        figure_path=tmp_path / "profiles.png",
    )

    assert paths["recommendation_json"].is_file()
    assert paths["recommendation_csv"].is_file()
    assert paths["profile_figure"].is_file()


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
