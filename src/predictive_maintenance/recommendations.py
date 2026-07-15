"""Business-oriented validation threshold recommendations."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.models import HIST_GRADIENT_BOOSTING_MODEL_NAME
from predictive_maintenance.thresholds import (
    run_threshold_analysis,
    select_best_f1_threshold,
)

PROFILE_ORDER: tuple[str, ...] = ("high_recall", "balanced", "conservative")
PROFILE_CSV_COLUMNS: tuple[str, ...] = (
    "profile",
    "threshold",
    "precision",
    "recall",
    "f1",
    "false_positive",
    "false_negative",
    "predicted_positive_count",
    "predicted_positive_rate",
    "illustrative_total_cost",
    "recall_target_met",
    "precision_target_met",
)

_REQUIRED_THRESHOLD_FIELDS: tuple[str, ...] = (
    "threshold",
    "precision",
    "recall",
    "f1",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "predicted_positive_count",
    "predicted_positive_rate",
    "total_cost",
)


def build_threshold_recommendations(
    threshold_analysis: dict[str, Any],
    minimum_recall: float = 0.85,
    minimum_precision: float = 0.70,
) -> dict[str, Any]:
    """Build three client-facing operating profiles from validation thresholds."""
    _validate_rate(minimum_recall, "minimum_recall")
    _validate_rate(minimum_precision, "minimum_precision")
    threshold_results = _validate_threshold_analysis(threshold_analysis)

    high_recall_row, high_recall_met = _select_high_recall_profile(
        threshold_results,
        minimum_recall,
    )
    balanced_row = select_best_f1_threshold(threshold_results)
    conservative_row, conservative_met = _select_conservative_profile(
        threshold_results,
        minimum_precision,
    )

    profiles = {
        "high_recall": _build_profile(
            profile_name="high_recall",
            row=high_recall_row,
            business_objective=(
                "Prioritize catching as many likely failures as possible."
            ),
            tradeoff_summary=(
                "Catches more failures but can produce more false positives and "
                "increase inspection workload."
            ),
            recommended_when=(
                "Use when missed failures are very costly and maintenance teams "
                "can absorb extra inspections."
            ),
            limitations=(
                "Validation recall target met."
                if high_recall_met
                else "Recall target was not met; this is the highest-recall "
                "validation threshold found on the Version 1 grid."
            ),
            recall_target_met=high_recall_met,
            precision_target_met=None,
        ),
        "balanced": _build_profile(
            profile_name="balanced",
            row=balanced_row,
            business_objective=(
                "Balance precision and recall using the best validation F1 score."
            ),
            tradeoff_summary=(
                "Provides a statistical balance, but may not be the optimal "
                "business policy without capacity and cost inputs."
            ),
            recommended_when=(
                "Use as a general operating point when no stronger business "
                "preference has been selected."
            ),
            limitations=(
                "Best F1 is validation-set specific and still depends on business "
                "capacity and failure cost."
            ),
            recall_target_met=None,
            precision_target_met=None,
        ),
        "conservative": _build_profile(
            profile_name="conservative",
            row=conservative_row,
            business_objective=(
                "Reduce false alarms while preserving as much validation recall as "
                "possible."
            ),
            tradeoff_summary=(
                "Lowers maintenance workload and false alarms but may miss more "
                "actual failures."
            ),
            recommended_when=(
                "Use when inspections are expensive or maintenance capacity is limited."
            ),
            limitations=(
                "Validation precision target met."
                if conservative_met
                else "Precision target was not met; this is the highest-precision "
                "validation threshold found on the Version 1 grid."
            ),
            recall_target_met=None,
            precision_target_met=conservative_met,
        ),
    }

    return {
        "selected_model": str(threshold_analysis["model"]),
        "validation_assumptions": {
            "minimum_recall": float(minimum_recall),
            "minimum_precision": float(minimum_precision),
            "split_row_counts": dict(threshold_analysis["split_row_counts"]),
            "cost_note": (
                "Illustrative costs support threshold comparison only; they are "
                "not real monetary operating costs."
            ),
            "threshold_source": (
                "Profiles are selected only from validation-threshold results."
            ),
        },
        "profiles": {profile: profiles[profile] for profile in PROFILE_ORDER},
        "profiles_share_threshold": bool(_shared_threshold_groups(profiles)),
        "shared_threshold_groups": _shared_threshold_groups(profiles),
        "comparison_table": _comparison_table(profiles),
        "decision_guidance": {
            "high_recall": (
                "Choose when missed failures are the dominant risk and extra "
                "maintenance review is acceptable."
            ),
            "balanced": (
                "Choose as a neutral validation operating point before business "
                "capacity constraints are finalized."
            ),
            "conservative": (
                "Choose when reducing false alarms and inspection workload is the "
                "primary constraint."
            ),
            "profile_convergence": (
                "Different business rules can select the same threshold. Identical "
                "thresholds do not mean the business objectives are identical; they "
                "mean the validation results support the same operating point for "
                "those objectives. This can change with another dataset, split, "
                "model, or business constraint."
            ),
        },
        "limitations": (
            "Recommendations are based on validation data only and are not "
            "production-ready causal conclusions."
        ),
        "test_set_status": (
            "Final test set remains untouched; no test-set predictions, "
            "probabilities, metrics, or recommendations were calculated."
        ),
    }


def run_business_threshold_recommendation(
    dataframe: pd.DataFrame | None = None,
    model_name: str = HIST_GRADIENT_BOOSTING_MODEL_NAME,
    minimum_recall: float = 0.85,
    minimum_precision: float = 0.70,
    false_negative_cost: float = 10.0,
    false_positive_cost: float = 1.0,
) -> dict[str, Any]:
    """Run validation-only threshold analysis and return business profiles."""
    _validate_rate(minimum_recall, "minimum_recall")
    _validate_rate(minimum_precision, "minimum_precision")
    _validate_costs(false_negative_cost, false_positive_cost)

    threshold_analysis = run_threshold_analysis(
        dataframe=None if dataframe is None else dataframe.copy(),
        model_name=model_name,
        minimum_recall=0.0,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return build_threshold_recommendations(
        threshold_analysis,
        minimum_recall=minimum_recall,
        minimum_precision=minimum_precision,
    )


def save_business_threshold_recommendation(
    recommendations: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save business threshold recommendations as strict JSON and CSV."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "recommendations"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "business_threshold_recommendation.json"
    csv_path = report_dir / "business_threshold_recommendation.csv"

    json_path.write_text(
        json.dumps(recommendations, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(recommendations["comparison_table"]).to_csv(
        csv_path,
        columns=list(PROFILE_CSV_COLUMNS),
        index=False,
    )
    return {
        "recommendation_json": json_path,
        "recommendation_csv": csv_path,
    }


def plot_business_threshold_profiles(
    recommendations: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """Plot validation precision, recall, and F1 by operating profile."""
    figure_path = (
        Path(output_path)
        if output_path is not None
        else config.FIGURES_DIR / "business_threshold_profiles.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = list(PROFILE_ORDER)
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(profiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, metric in enumerate(metrics):
        values = [recommendations["profiles"][profile][metric] for profile in profiles]
        ax.bar(x + (offset - 1) * width, values, width, label=metric)

    ax.set_title("Validation Threshold Profile Comparison")
    if recommendations.get("profiles_share_threshold"):
        ax.text(
            0.5,
            1.02,
            "Some profiles share the same validation threshold.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Operating profile")
    ax.set_ylabel("Validation metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def save_business_threshold_outputs(
    recommendations: dict[str, Any],
    output_dir: Path | str | None = None,
    figure_path: Path | str | None = None,
) -> dict[str, Path]:
    """Save JSON, CSV, and profile comparison figure."""
    paths = save_business_threshold_recommendation(recommendations, output_dir)
    paths["profile_figure"] = plot_business_threshold_profiles(
        recommendations,
        output_path=figure_path,
    )
    return paths


def _select_high_recall_profile(
    threshold_results: Sequence[dict[str, Any]],
    minimum_recall: float,
) -> tuple[dict[str, Any], bool]:
    """Select the high-recall profile and whether the recall target was met."""
    eligible = [
        result for result in threshold_results if result["recall"] >= minimum_recall
    ]
    if eligible:
        return (
            dict(
                max(
                    eligible,
                    key=lambda result: (
                        result["precision"],
                        result["f1"],
                        result["recall"],
                        result["threshold"],
                    ),
                )
            ),
            True,
        )

    return (
        dict(
            max(
                threshold_results,
                key=lambda result: (
                    result["recall"],
                    result["precision"],
                    result["f1"],
                    result["threshold"],
                ),
            )
        ),
        False,
    )


def _select_conservative_profile(
    threshold_results: Sequence[dict[str, Any]],
    minimum_precision: float,
) -> tuple[dict[str, Any], bool]:
    """Select the conservative profile and whether the precision target was met."""
    eligible = [
        result
        for result in threshold_results
        if result["precision"] >= minimum_precision
    ]
    if eligible:
        return (
            dict(
                max(
                    eligible,
                    key=lambda result: (
                        result["recall"],
                        result["f1"],
                        result["precision"],
                        result["threshold"],
                    ),
                )
            ),
            True,
        )

    return (
        dict(
            max(
                threshold_results,
                key=lambda result: (
                    result["precision"],
                    result["recall"],
                    result["f1"],
                    result["threshold"],
                ),
            )
        ),
        False,
    )


def _build_profile(
    profile_name: str,
    row: dict[str, Any],
    business_objective: str,
    tradeoff_summary: str,
    recommended_when: str,
    limitations: str,
    recall_target_met: bool | None,
    precision_target_met: bool | None,
) -> dict[str, Any]:
    """Create one JSON-safe business operating profile."""
    actual_positive_count = row["true_positive"] + row["false_negative"]
    total_count = (
        row["true_negative"]
        + row["false_positive"]
        + row["false_negative"]
        + row["true_positive"]
    )
    return {
        "profile_name": profile_name,
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "true_negative": int(row["true_negative"]),
        "false_positive": int(row["false_positive"]),
        "false_negative": int(row["false_negative"]),
        "true_positive": int(row["true_positive"]),
        "predicted_positive_count": int(row["predicted_positive_count"]),
        "predicted_positive_rate": float(row["predicted_positive_rate"]),
        "false_positive_rate_overall": float(row["false_positive"] / total_count),
        "false_negative_rate_among_actual_positives": (
            float(row["false_negative"] / actual_positive_count)
            if actual_positive_count
            else 0.0
        ),
        "illustrative_total_cost": float(row["total_cost"]),
        "recall_target_met": recall_target_met,
        "precision_target_met": precision_target_met,
        "business_objective": business_objective,
        "tradeoff_summary": tradeoff_summary,
        "recommended_when": recommended_when,
        "limitations": limitations,
    }


def _comparison_table(profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic one-row-per-profile comparison table."""
    rows: list[dict[str, Any]] = []
    for profile_name in PROFILE_ORDER:
        profile = profiles[profile_name]
        rows.append(
            {
                "profile": profile_name,
                "threshold": profile["threshold"],
                "precision": profile["precision"],
                "recall": profile["recall"],
                "f1": profile["f1"],
                "false_positive": profile["false_positive"],
                "false_negative": profile["false_negative"],
                "predicted_positive_count": profile["predicted_positive_count"],
                "predicted_positive_rate": profile["predicted_positive_rate"],
                "illustrative_total_cost": profile["illustrative_total_cost"],
                "recall_target_met": profile["recall_target_met"],
                "precision_target_met": profile["precision_target_met"],
            }
        )
    return rows


def _shared_threshold_groups(
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic groups of profiles sharing the same threshold."""
    by_threshold: dict[float, list[str]] = {}
    for profile_name in PROFILE_ORDER:
        threshold = float(profiles[profile_name]["threshold"])
        by_threshold.setdefault(threshold, []).append(profile_name)

    return [
        {
            "threshold": threshold,
            "profiles": by_threshold[threshold],
            "explanation": (
                "Distinct selection rules converged on the same validation "
                "operating point. The profile objectives remain different."
            ),
        }
        for threshold in sorted(by_threshold)
        if len(by_threshold[threshold]) > 1
    ]


def _validate_threshold_analysis(
    threshold_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the threshold-analysis structure used by recommendations."""
    required_top_level = ("model", "split_row_counts", "threshold_results")
    missing_top_level = [
        field for field in required_top_level if field not in threshold_analysis
    ]
    if missing_top_level:
        raise ValueError(
            "threshold_analysis is missing required fields: "
            f"{', '.join(missing_top_level)}"
        )

    threshold_results = threshold_analysis["threshold_results"]
    if not isinstance(threshold_results, Sequence) or isinstance(
        threshold_results,
        str,
    ):
        raise ValueError("threshold_results must be a sequence of metric rows.")
    if not threshold_results:
        raise ValueError("threshold_results must contain at least one row.")

    validated: list[dict[str, Any]] = []
    seen_thresholds: set[float] = set()
    expected_total_rows: int | None = None
    expected_actual_positives: int | None = None
    expected_actual_negatives: int | None = None
    for index, row in enumerate(threshold_results):
        if not isinstance(row, dict):
            raise ValueError("Each threshold result must be a dictionary.")
        missing_fields = [
            field for field in _REQUIRED_THRESHOLD_FIELDS if field not in row
        ]
        if missing_fields:
            raise ValueError(
                f"Threshold result at position {index} is missing fields: "
                f"{', '.join(missing_fields)}"
            )
        threshold = float(row["threshold"])
        if threshold in seen_thresholds:
            raise ValueError(f"Duplicate threshold found: {threshold}")
        seen_thresholds.add(threshold)
        total_rows, actual_positives, actual_negatives = _validate_threshold_row(
            row,
            index,
        )
        if expected_total_rows is None:
            expected_total_rows = total_rows
            expected_actual_positives = actual_positives
            expected_actual_negatives = actual_negatives
        elif (
            total_rows != expected_total_rows
            or actual_positives != expected_actual_positives
            or actual_negatives != expected_actual_negatives
        ):
            raise ValueError(
                "Threshold result confusion-matrix totals must be consistent "
                "across rows."
            )
        validated.append(dict(row))

    return validated


def _validate_threshold_row(row: dict[str, Any], index: int) -> tuple[int, int, int]:
    """Validate numeric threshold metric fields and derived consistency."""
    for field in (
        "threshold",
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
        "total_cost",
    ):
        value = float(row[field])
        if not np.isfinite(value):
            raise ValueError(
                f"Threshold result {index} field '{field}' must be finite."
            )
        if field != "total_cost" and (value < 0.0 or value > 1.0):
            raise ValueError(
                f"Threshold result {index} field '{field}' must be within [0, 1]."
            )
        if field == "total_cost" and value < 0.0:
            raise ValueError(
                f"Threshold result {index} field 'total_cost' cannot be negative."
            )

    for field in (
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "predicted_positive_count",
    ):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Threshold result {index} field '{field}' must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"Threshold result {index} field '{field}' cannot be negative."
            )

    true_negative = row["true_negative"]
    false_positive = row["false_positive"]
    false_negative = row["false_negative"]
    true_positive = row["true_positive"]
    predicted_positive_count = row["predicted_positive_count"]
    total_rows = true_negative + false_positive + false_negative + true_positive
    actual_positives = false_negative + true_positive
    actual_negatives = true_negative + false_positive

    if total_rows <= 0:
        raise ValueError("Threshold result confusion matrix must contain rows.")
    if predicted_positive_count != false_positive + true_positive:
        raise ValueError(
            "predicted_positive_count must equal false_positive + true_positive."
        )

    expected_predicted_positive_rate = predicted_positive_count / total_rows
    if not np.isclose(
        row["predicted_positive_rate"],
        expected_predicted_positive_rate,
    ):
        raise ValueError(
            "predicted_positive_rate must equal predicted_positive_count / total rows."
        )

    expected_precision = (
        true_positive / predicted_positive_count if predicted_positive_count else 0.0
    )
    expected_recall = true_positive / actual_positives if actual_positives else 0.0
    expected_f1 = (
        2
        * expected_precision
        * expected_recall
        / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    if not np.isclose(row["precision"], expected_precision):
        raise ValueError("precision is inconsistent with confusion-matrix values.")
    if not np.isclose(row["recall"], expected_recall):
        raise ValueError("recall is inconsistent with confusion-matrix values.")
    if not np.isclose(row["f1"], expected_f1):
        raise ValueError("f1 is inconsistent with precision and recall.")

    return total_rows, actual_positives, actual_negatives


def _validate_rate(value: float, name: str) -> None:
    """Validate a rate-like input in the inclusive [0, 1] range."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1].")


def _validate_costs(false_negative_cost: float, false_positive_cost: float) -> None:
    """Validate illustrative cost assumptions before expensive workflow steps."""
    try:
        false_negative_cost = float(false_negative_cost)
        false_positive_cost = float(false_positive_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cost values must be numeric.") from exc
    if false_negative_cost <= 0.0 or false_positive_cost <= 0.0:
        raise ValueError("Cost values must be positive.")
    if not np.isfinite([false_negative_cost, false_positive_cost]).all():
        raise ValueError("Cost values must be finite.")
    if false_negative_cost <= false_positive_cost:
        raise ValueError(
            "false_negative_cost must be greater than false_positive_cost for "
            "Version 1 predictive-maintenance threshold recommendations."
        )
