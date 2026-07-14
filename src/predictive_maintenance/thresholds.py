"""Validation-set threshold analysis for binary predictive maintenance models."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import positive_class_probability
from predictive_maintenance.exceptions import (
    ModelEvaluationError,
    ThresholdAnalysisError,
)
from predictive_maintenance.models import (
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    build_model_comparison_models,
)
from predictive_maintenance.splitting import split_dataset
from predictive_maintenance.train import _class_distributions, _fit_model

DEFAULT_MINIMUM_RECALL: float = 0.80
DEFAULT_FALSE_NEGATIVE_COST: float = 10.0
DEFAULT_FALSE_POSITIVE_COST: float = 1.0
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(
    round(value, 2) for value in np.arange(0.05, 1.0, 0.05)
)

THRESHOLD_REPORT_COLUMNS: tuple[str, ...] = (
    "threshold",
    "accuracy",
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


def evaluate_thresholds(
    y_true: pd.Series | np.ndarray,
    positive_probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Calculate binary classification metrics across validation thresholds.

    Version 1 uses a deterministic default grid from 0.05 to 0.95 in 0.05
    increments. Custom thresholds may be supplied in any order; they are
    validated, deduplicated, and returned in ascending order. Thresholds must
    be strictly between 0 and 1.
    """
    target = _validate_binary_target(y_true)
    probabilities = _validate_probabilities(positive_probabilities, len(target))
    threshold_values = _validate_thresholds(
        DEFAULT_THRESHOLDS if thresholds is None else thresholds
    )

    results: list[dict[str, Any]] = []
    for threshold in threshold_values:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            target,
            predictions,
            labels=[0, 1],
        ).ravel()
        predicted_positive_count = int(predictions.sum())

        results.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(target, predictions)),
                "precision": float(
                    precision_score(
                        target,
                        predictions,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        target,
                        predictions,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(target, predictions, pos_label=1, zero_division=0)
                ),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": float(
                    predicted_positive_count / len(predictions)
                ),
            }
        )

    return results


def select_best_f1_threshold(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select the threshold with best F1, then recall, precision, and threshold.

    When metric ties remain, Version 1 prefers the higher threshold to reduce
    avoidable false positives while preserving the tied F1 and recall.
    """
    threshold_results = _validate_threshold_results(results)
    return dict(
        max(
            threshold_results,
            key=lambda result: (
                result["f1"],
                result["recall"],
                result["precision"],
                result["threshold"],
            ),
        )
    )


def select_threshold_for_minimum_recall(
    results: Sequence[dict[str, Any]],
    minimum_recall: float,
) -> dict[str, Any]:
    """Select the most precise threshold meeting a minimum validation recall."""
    _validate_rate(minimum_recall, "minimum_recall")
    threshold_results = _validate_threshold_results(results)
    eligible = [
        result for result in threshold_results if result["recall"] >= minimum_recall
    ]
    if not eligible:
        raise ThresholdAnalysisError(
            f"No threshold satisfies minimum_recall={minimum_recall:.2f}."
        )

    return dict(
        max(
            eligible,
            key=lambda result: (
                result["precision"],
                result["f1"],
                result["recall"],
                result["threshold"],
            ),
        )
    )


def calculate_threshold_costs(
    results: Sequence[dict[str, Any]],
    false_negative_cost: float,
    false_positive_cost: float,
) -> list[dict[str, Any]]:
    """Add illustrative business-cost fields to threshold metric rows.

    Version 1 intentionally requires positive costs with false-negative cost
    greater than false-positive cost to reflect the initial predictive
    maintenance assumption that missed failures are more expensive than
    unnecessary inspections.
    """
    _validate_costs(false_negative_cost, false_positive_cost)
    cost_results: list[dict[str, Any]] = []
    for result in _validate_threshold_results(results):
        row = dict(result)
        row["total_cost"] = float(
            row["false_negative"] * false_negative_cost
            + row["false_positive"] * false_positive_cost
        )
        cost_results.append(row)

    return cost_results


def select_lowest_cost_threshold(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest-cost threshold with deterministic metric tie-breaking."""
    threshold_results = _validate_threshold_results(results)
    missing_cost = [
        result for result in threshold_results if "total_cost" not in result
    ]
    if missing_cost:
        raise ThresholdAnalysisError(
            "Threshold results must include total_cost before cost-based selection."
        )

    return dict(
        min(
            threshold_results,
            key=lambda result: (
                result["total_cost"],
                -result["recall"],
                -result["precision"],
                -result["f1"],
                -result["threshold"],
            ),
        )
    )


def run_threshold_analysis(
    dataframe: pd.DataFrame | None = None,
    model_name: str = HIST_GRADIENT_BOOSTING_MODEL_NAME,
    minimum_recall: float = DEFAULT_MINIMUM_RECALL,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> dict[str, Any]:
    """Fit the selected model on train only and analyze validation thresholds."""
    _validate_rate(minimum_recall, "minimum_recall")
    _validate_costs(false_negative_cost, false_positive_cost)

    dataset = load_dataset() if dataframe is None else dataframe.copy()
    split_data = split_dataset(dataset)
    models = build_model_comparison_models()
    if model_name not in models:
        available_models = ", ".join(models)
        raise ThresholdAnalysisError(
            f"Unknown model_name '{model_name}'. Available models: {available_models}."
        )

    model = models[model_name]
    _fit_model(model_name, model, split_data.X_train, split_data.y_train)

    try:
        predicted_probabilities = model.predict_proba(split_data.X_validation)
    except Exception as exc:
        raise ThresholdAnalysisError(
            "Unable to generate validation probabilities for threshold analysis. "
            "The selected model must expose predict_proba after fitting."
        ) from exc
    try:
        validation_probabilities = positive_class_probability(
            model,
            predicted_probabilities,
        )
    except ModelEvaluationError as exc:
        raise ThresholdAnalysisError(
            "Unable to select class-1 validation probabilities for threshold "
            "analysis. Confirm the fitted model exposes classes_ and includes "
            "class 1."
        ) from exc
    threshold_results = calculate_threshold_costs(
        evaluate_thresholds(split_data.y_validation, validation_probabilities),
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    default_threshold_metrics = _find_threshold_result(threshold_results, 0.5)
    best_f1_threshold = select_best_f1_threshold(threshold_results)
    minimum_recall_threshold = select_threshold_for_minimum_recall(
        threshold_results,
        minimum_recall=minimum_recall,
    )
    lowest_cost_threshold = select_lowest_cost_threshold(threshold_results)

    return {
        "model": model_name,
        "split_row_counts": {
            "train": int(len(split_data.X_train)),
            "validation": int(len(split_data.X_validation)),
            "test": int(len(split_data.X_test)),
        },
        "validation_class_distribution": _class_distributions(split_data)["validation"],
        "assumptions": {
            "minimum_recall": float(minimum_recall),
            "false_negative_cost": float(false_negative_cost),
            "false_positive_cost": float(false_positive_cost),
            "cost_note": (
                "Costs are illustrative Version 1 assumptions for validation-set "
                "threshold comparison, not real operational cost estimates."
            ),
            "threshold_grid_note": (
                "The 0.05-step threshold grid is suitable for Version 1 "
                "comparison; final operating-threshold selection may use a "
                "finer local grid later."
            ),
        },
        "selected_thresholds": {
            "default_0_5": default_threshold_metrics,
            "best_f1": best_f1_threshold,
            "minimum_recall": minimum_recall_threshold,
            "lowest_cost": lowest_cost_threshold,
        },
        "threshold_results": threshold_results,
        "test_set_status": (
            "Final test set remains untouched; no test-set metrics were calculated."
        ),
    }


def save_threshold_analysis_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save validation-threshold analysis as strict JSON and deterministic CSV."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "threshold_analysis"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "threshold_analysis.json"
    csv_path = report_dir / "threshold_metrics.csv"

    json_path.write_text(
        json.dumps(results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results["threshold_results"]).to_csv(
        csv_path,
        columns=list(THRESHOLD_REPORT_COLUMNS),
        index=False,
    )

    return {"threshold_analysis_json": json_path, "threshold_metrics_csv": csv_path}


def _validate_binary_target(y_true: pd.Series | np.ndarray) -> np.ndarray:
    """Return a one-dimensional target array with labels exactly {0, 1}."""
    target = np.asarray(y_true)
    if target.ndim != 1:
        raise ThresholdAnalysisError(
            "y_true must be a one-dimensional array or Series."
        )
    if len(target) == 0:
        raise ThresholdAnalysisError("y_true must contain at least one row.")
    if pd.isna(target).any():
        raise ThresholdAnalysisError(
            "y_true contains missing values; threshold analysis requires labels "
            "0 and 1."
        )

    observed_labels = set(target.tolist())
    if observed_labels != {0, 1}:
        labels = ", ".join(str(label) for label in sorted(observed_labels, key=str))
        raise ThresholdAnalysisError(
            "y_true labels must be exactly {0, 1} for binary threshold analysis; "
            f"observed labels: {labels}."
        )

    return target.astype(int)


def _validate_probabilities(
    positive_probabilities: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Return finite class-1 probabilities in the inclusive [0, 1] range."""
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ThresholdAnalysisError(
            "positive_probabilities must be a one-dimensional array."
        )
    if len(probabilities) != expected_length:
        raise ThresholdAnalysisError(
            "y_true and positive_probabilities must have the same length."
        )
    if not np.isfinite(probabilities).all():
        raise ThresholdAnalysisError("positive_probabilities must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ThresholdAnalysisError(
            "positive_probabilities must be within the [0, 1] range."
        )

    return probabilities


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    """Return unique finite thresholds in ascending order."""
    try:
        threshold_values = tuple(float(threshold) for threshold in thresholds)
    except (TypeError, ValueError) as exc:
        raise ThresholdAnalysisError("Thresholds must be numeric.") from exc

    if not threshold_values:
        raise ThresholdAnalysisError("At least one threshold must be provided.")
    if not np.isfinite(threshold_values).all():
        raise ThresholdAnalysisError("Thresholds must be finite.")
    invalid_thresholds = [
        threshold
        for threshold in threshold_values
        if threshold <= 0.0 or threshold >= 1.0
    ]
    if invalid_thresholds:
        raise ThresholdAnalysisError("Thresholds must be strictly between 0 and 1.")

    return tuple(sorted(set(threshold_values)))


def _validate_threshold_results(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return threshold rows or raise a clear error for empty input."""
    threshold_results = list(results)
    if not threshold_results:
        raise ThresholdAnalysisError("Threshold results must contain at least one row.")
    return threshold_results


def _validate_rate(value: float, name: str) -> None:
    """Validate a rate-like value in the inclusive [0, 1] range."""
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ThresholdAnalysisError(f"{name} must be finite and within [0, 1].")


def _validate_costs(false_negative_cost: float, false_positive_cost: float) -> None:
    """Validate illustrative cost assumptions for threshold cost analysis."""
    if false_negative_cost <= 0.0 or false_positive_cost <= 0.0:
        raise ThresholdAnalysisError("Cost values must be positive.")
    if not np.isfinite([false_negative_cost, false_positive_cost]).all():
        raise ThresholdAnalysisError("Cost values must be finite.")
    if false_negative_cost <= false_positive_cost:
        raise ThresholdAnalysisError(
            "false_negative_cost must be greater than false_positive_cost for "
            "Version 1 predictive-maintenance threshold analysis."
        )


def _find_threshold_result(
    results: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Return one threshold row by numeric value."""
    for result in results:
        if np.isclose(result["threshold"], threshold):
            return dict(result)
    raise ThresholdAnalysisError(f"Threshold {threshold} is not present in results.")
