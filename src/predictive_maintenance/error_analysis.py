"""Validation-set error analysis for selected predictive maintenance models."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import positive_class_probability
from predictive_maintenance.exceptions import ErrorAnalysisError, ModelEvaluationError
from predictive_maintenance.models import (
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    build_model_comparison_models,
)
from predictive_maintenance.splitting import split_dataset
from predictive_maintenance.thresholds import (
    DEFAULT_FALSE_NEGATIVE_COST,
    DEFAULT_FALSE_POSITIVE_COST,
    DEFAULT_MINIMUM_RECALL,
    calculate_threshold_costs,
    evaluate_thresholds,
    select_best_f1_threshold,
    select_lowest_cost_threshold,
    select_threshold_for_minimum_recall,
)
from predictive_maintenance.train import _fit_model

OUTCOME_ORDER: tuple[str, ...] = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)
OUTCOME_COLUMNS: tuple[str, ...] = (
    "actual",
    "predicted",
    "probability",
    "outcome",
)
ERROR_CASE_COLUMNS: tuple[str, ...] = (
    "threshold",
    "outcome",
    "probability",
    *config.MODEL_FEATURES,
)


def classify_prediction_outcomes(
    y_true: pd.Series | np.ndarray,
    positive_probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Classify validation predictions as TN, FP, FN, or TP."""
    target = _validate_binary_target(y_true)
    probabilities = _validate_probabilities(positive_probabilities, len(target))
    threshold_value = _validate_threshold(threshold)
    index = y_true.index.copy() if isinstance(y_true, pd.Series) else None

    predicted = (probabilities >= threshold_value).astype(int)
    outcomes = np.select(
        [
            (target == 0) & (predicted == 0),
            (target == 0) & (predicted == 1),
            (target == 1) & (predicted == 0),
            (target == 1) & (predicted == 1),
        ],
        OUTCOME_ORDER,
        default="unknown",
    )

    return pd.DataFrame(
        {
            "actual": target,
            "predicted": predicted,
            "probability": probabilities,
            "outcome": outcomes,
        },
        index=index,
        columns=OUTCOME_COLUMNS,
    )


def build_error_analysis_table(
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    positive_probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Combine validation model features with prediction outcome labels."""
    _validate_validation_alignment(X_validation, y_validation)
    features = X_validation.loc[:, config.MODEL_FEATURES].copy()
    outcomes = classify_prediction_outcomes(
        y_validation,
        positive_probabilities,
        threshold,
    )
    return pd.concat([features, outcomes], axis=1).loc[
        :,
        (*config.MODEL_FEATURES, *OUTCOME_COLUMNS),
    ]


def summarize_outcomes(error_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize validation row counts and probabilities by outcome group."""
    row_count = len(error_table)
    return {
        outcome: {
            "count": int(len(group)),
            "percentage": _percentage(len(group), row_count),
            "mean_probability": _safe_number(group["probability"].mean()),
            "median_probability": _safe_number(group["probability"].median()),
        }
        for outcome, group in _outcome_groups(error_table).items()
    }


def summarize_numerical_features_by_outcome(
    error_table: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Summarize numerical validation features by prediction outcome."""
    return {
        outcome: _summarize_numerical_features(group)
        for outcome, group in _outcome_groups(error_table).items()
    }


def summarize_categorical_features_by_outcome(
    error_table: pd.DataFrame,
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    """Summarize categorical validation features by prediction outcome."""
    return {
        outcome: {
            feature: _value_distribution(group[feature])
            for feature in config.CATEGORICAL_FEATURES
        }
        for outcome, group in _outcome_groups(error_table).items()
    }


def summarize_false_positives(
    error_table: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return focused validation-set false-positive diagnostics."""
    _validate_top_n(top_n)
    false_positives = error_table[error_table["outcome"] == "false_positive"].copy()
    true_negatives = error_table[error_table["outcome"] == "true_negative"].copy()
    return _focused_error_summary(
        false_positives,
        reference_table=true_negatives,
        validation_row_count=len(error_table),
        top_n=top_n,
        sort_ascending=False,
        top_key="top_false_positives",
    )


def summarize_false_negatives(
    error_table: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return focused validation-set false-negative diagnostics."""
    _validate_top_n(top_n)
    false_negatives = error_table[error_table["outcome"] == "false_negative"].copy()
    true_positives = error_table[error_table["outcome"] == "true_positive"].copy()
    return _focused_error_summary(
        false_negatives,
        reference_table=true_positives,
        validation_row_count=len(error_table),
        top_n=top_n,
        sort_ascending=False,
        top_key="top_false_negatives",
    )


def run_validation_error_analysis(
    dataframe: pd.DataFrame | None = None,
    model_name: str = HIST_GRADIENT_BOOSTING_MODEL_NAME,
    thresholds: Sequence[float] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Run validation-only error analysis for the selected fitted model."""
    _validate_top_n(top_n)
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    split_data = split_dataset(dataset)
    models = build_model_comparison_models()
    if model_name not in models:
        available_models = ", ".join(models)
        raise ErrorAnalysisError(
            f"Unknown model_name '{model_name}'. Available models: {available_models}."
        )

    model = models[model_name]
    try:
        _fit_model(model_name, model, split_data.X_train, split_data.y_train)
    except Exception as exc:
        raise ErrorAnalysisError(
            "Unable to fit selected model for validation error analysis."
        ) from exc
    validation_probabilities = _validation_probabilities(
        model,
        split_data.X_validation,
    )
    selected_thresholds = _selected_thresholds(
        split_data.y_validation,
        validation_probabilities,
        thresholds,
    )

    per_threshold: dict[str, Any] = {}
    error_case_tables: list[pd.DataFrame] = []
    for label, threshold in selected_thresholds.items():
        error_table = build_error_analysis_table(
            split_data.X_validation,
            split_data.y_validation,
            validation_probabilities,
            threshold,
        )
        threshold_summary = _threshold_error_summary(error_table, threshold)
        per_threshold[label] = threshold_summary
        error_case_tables.append(_error_cases_for_report(error_table, threshold))

    primary_label = _primary_threshold_label(selected_thresholds)
    primary_table = build_error_analysis_table(
        split_data.X_validation,
        split_data.y_validation,
        validation_probabilities,
        selected_thresholds[primary_label],
    )

    return {
        "model": model_name,
        "split_row_counts": {
            "train": int(len(split_data.X_train)),
            "validation": int(len(split_data.X_validation)),
            "test": int(len(split_data.X_test)),
        },
        "thresholds_analyzed": selected_thresholds,
        "primary_threshold": primary_label,
        "per_threshold": per_threshold,
        "false_positive_summary": summarize_false_positives(primary_table, top_n),
        "false_negative_summary": summarize_false_negatives(primary_table, top_n),
        "business_interpretation_notes": (
            "Lower thresholds reduce missed failures but increase maintenance "
            "workload. Higher thresholds reduce false alarms but may miss more "
            "failures. These validation-set diagnostics are not production "
            "readiness claims."
        ),
        "error_cases": _combine_error_case_tables(error_case_tables),
        "test_set_status": (
            "Final test set remains untouched; no test-set probabilities or "
            "metrics were calculated."
        ),
    }


def save_error_analysis_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save validation error analysis JSON and row-level error-case CSV."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "error_analysis"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "error_analysis.json"
    csv_path = report_dir / "error_cases.csv"
    json_results = {
        key: value for key, value in results.items() if key != "error_cases"
    }

    json_path.write_text(
        json.dumps(json_results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results["error_cases"]).to_csv(
        csv_path,
        columns=list(ERROR_CASE_COLUMNS),
        index=False,
    )

    return {"error_analysis_json": json_path, "error_cases_csv": csv_path}


def _validation_probabilities(model: Any, X_validation: pd.DataFrame) -> np.ndarray:
    """Return validation class-1 probabilities from a fitted model."""
    try:
        predicted_probabilities = model.predict_proba(X_validation)
    except Exception as exc:
        raise ErrorAnalysisError(
            "Unable to generate validation probabilities for error analysis."
        ) from exc
    try:
        return positive_class_probability(model, predicted_probabilities)
    except ModelEvaluationError as exc:
        raise ErrorAnalysisError(
            "Unable to select class-1 validation probabilities for error analysis."
        ) from exc


def _selected_thresholds(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    thresholds: Sequence[float] | None,
) -> dict[str, float]:
    """Return deterministic threshold labels and values for error analysis."""
    if thresholds is not None:
        threshold_results = evaluate_thresholds(
            y_validation,
            validation_probabilities,
            thresholds=thresholds,
        )
        return {
            f"threshold_{result['threshold']:.2f}".replace(".", "_"): result[
                "threshold"
            ]
            for result in threshold_results
        }

    threshold_results = calculate_threshold_costs(
        evaluate_thresholds(y_validation, validation_probabilities),
        false_negative_cost=DEFAULT_FALSE_NEGATIVE_COST,
        false_positive_cost=DEFAULT_FALSE_POSITIVE_COST,
    )
    return {
        "default_0_5": 0.5,
        "best_f1": select_best_f1_threshold(threshold_results)["threshold"],
        "minimum_recall": select_threshold_for_minimum_recall(
            threshold_results,
            minimum_recall=DEFAULT_MINIMUM_RECALL,
        )["threshold"],
        "lowest_cost": select_lowest_cost_threshold(threshold_results)["threshold"],
    }


def _threshold_error_summary(
    error_table: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Return threshold-level confusion and workload diagnostics."""
    actual = error_table["actual"].to_numpy()
    predicted = error_table["predicted"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    actual_positive_count = int((actual == 1).sum())
    predicted_positive_count = int((predicted == 1).sum())
    validation_row_count = len(error_table)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "predicted_positive_count": predicted_positive_count,
        "false_positive_rate": _percentage(int(fp), validation_row_count),
        "false_negative_rate_actual_positives": _percentage(
            int(fn),
            actual_positive_count,
        ),
        "maintenance_workload_interpretation": (
            f"{predicted_positive_count} of {validation_row_count} validation rows "
            "would be flagged for maintenance review at this threshold."
        ),
        "outcome_summary": summarize_outcomes(error_table),
        "numerical_feature_summary": summarize_numerical_features_by_outcome(
            error_table
        ),
        "categorical_feature_summary": summarize_categorical_features_by_outcome(
            error_table
        ),
    }


def _focused_error_summary(
    error_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    validation_row_count: int,
    top_n: int,
    sort_ascending: bool,
    top_key: str,
) -> dict[str, Any]:
    """Summarize one focused error group against its corresponding correct group."""
    top_rows = error_table.sort_values(
        by="probability",
        ascending=sort_ascending,
        kind="mergesort",
    ).head(top_n)
    return {
        "count": int(len(error_table)),
        "percentage_of_validation_rows": _percentage(
            len(error_table),
            validation_row_count,
        ),
        "probability_summary": _probability_summary(error_table["probability"]),
        "numerical_feature_summary": _summarize_numerical_features(error_table),
        "type_distribution": _value_distribution(error_table["Type"]),
        "feature_wise_mean_differences_vs_correct": _feature_wise_mean_differences(
            error_table,
            reference_table,
        ),
        top_key: _records(
            top_rows.loc[:, (*config.MODEL_FEATURES, "probability", "outcome")]
        ),
    }


def _outcome_groups(error_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return every outcome group in stable order, including empty groups."""
    return {
        outcome: error_table[error_table["outcome"] == outcome].copy()
        for outcome in OUTCOME_ORDER
    }


def _summarize_numerical_features(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Return numeric summaries for configured numerical features."""
    return {
        feature: _numeric_summary(dataframe[feature])
        for feature in config.NUMERICAL_FEATURES
    }


def _numeric_summary(series: pd.Series) -> dict[str, float | None]:
    """Return JSON-safe numerical summary statistics."""
    return {
        "mean": _safe_number(series.mean()),
        "median": _safe_number(series.median()),
        "std": _safe_number(series.std()),
        "min": _safe_number(series.min()),
        "max": _safe_number(series.max()),
        "q1": _safe_number(series.quantile(0.25)),
        "q3": _safe_number(series.quantile(0.75)),
    }


def _probability_summary(series: pd.Series) -> dict[str, float | None]:
    """Return compact JSON-safe probability summary statistics."""
    return {
        "mean": _safe_number(series.mean()),
        "median": _safe_number(series.median()),
        "min": _safe_number(series.min()),
        "max": _safe_number(series.max()),
        "q1": _safe_number(series.quantile(0.25)),
        "q3": _safe_number(series.quantile(0.75)),
    }


def _value_distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    """Return categorical counts and percentages using stable string keys."""
    total = len(series)
    counts = series.value_counts(dropna=False).sort_index()
    return {
        str(label): {
            "count": int(count),
            "percentage": _percentage(int(count), total),
        }
        for label, count in counts.items()
    }


def _feature_wise_mean_differences(
    error_table: pd.DataFrame,
    reference_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return native-unit mean differences per feature without cross-feature ranking."""
    rows: list[dict[str, Any]] = []
    for feature in config.NUMERICAL_FEATURES:
        error_mean = _safe_number(error_table[feature].mean())
        reference_mean = _safe_number(reference_table[feature].mean())
        difference = (
            None
            if error_mean is None or reference_mean is None
            else float(error_mean - reference_mean)
        )
        rows.append(
            {
                "feature": feature,
                "error_mean": error_mean,
                "correct_mean": reference_mean,
                "difference": difference,
                "unit_note": "Difference is reported in the feature's native units.",
            }
        )

    return rows


def _error_cases_for_report(
    error_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return false-positive and false-negative rows for CSV reporting."""
    error_cases = error_table[
        error_table["outcome"].isin(("false_positive", "false_negative"))
    ].copy()
    error_cases.insert(0, "threshold", float(threshold))
    return error_cases.loc[:, ERROR_CASE_COLUMNS]


def _combine_error_case_tables(error_case_tables: Sequence[pd.DataFrame]) -> list[dict]:
    """Return deterministic JSON-safe error-case records for reporting."""
    if not error_case_tables:
        return []
    combined = pd.concat(error_case_tables, axis=0, ignore_index=True)
    combined = combined.sort_values(
        by=["threshold", "outcome", "probability"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    return _records(combined.loc[:, ERROR_CASE_COLUMNS])


def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Return dataframe records with JSON-safe scalar values."""
    return [
        {column: _json_safe_value(value) for column, value in row.items()}
        for row in dataframe.to_dict(orient="records")
    ]


def _primary_threshold_label(selected_thresholds: dict[str, float]) -> str:
    """Choose the threshold used for focused FP/FN summaries."""
    if "best_f1" in selected_thresholds:
        return "best_f1"
    return next(iter(selected_thresholds))


def _validate_validation_alignment(
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """Validate validation feature and target rows are index-aligned."""
    if not X_validation.index.equals(y_validation.index):
        raise ErrorAnalysisError(
            "X_validation and y_validation indices must match exactly before "
            "building the error-analysis table."
        )


def _validate_binary_target(y_true: pd.Series | np.ndarray) -> np.ndarray:
    """Return a one-dimensional target array with labels exactly {0, 1}."""
    target = np.asarray(y_true)
    if target.ndim != 1:
        raise ErrorAnalysisError("y_true must be a one-dimensional array or Series.")
    if len(target) == 0:
        raise ErrorAnalysisError("y_true must contain at least one row.")
    if pd.isna(target).any():
        raise ErrorAnalysisError("y_true contains missing values.")

    observed_labels = set(target.tolist())
    if observed_labels != {0, 1}:
        labels = ", ".join(str(label) for label in sorted(observed_labels, key=str))
        raise ErrorAnalysisError(
            "y_true labels must be exactly {0, 1} for error analysis; "
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
        raise ErrorAnalysisError(
            "positive_probabilities must be a one-dimensional array."
        )
    if len(probabilities) != expected_length:
        raise ErrorAnalysisError(
            "y_true and positive_probabilities must have the same length."
        )
    if not np.isfinite(probabilities).all():
        raise ErrorAnalysisError("positive_probabilities must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ErrorAnalysisError(
            "positive_probabilities must be within the [0, 1] range."
        )

    return probabilities


def _validate_threshold(threshold: float) -> float:
    """Return one finite threshold strictly between 0 and 1."""
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ErrorAnalysisError("threshold must be numeric.") from exc
    if not np.isfinite(threshold_value):
        raise ErrorAnalysisError("threshold must be finite.")
    if threshold_value <= 0.0 or threshold_value >= 1.0:
        raise ErrorAnalysisError("threshold must be strictly between 0 and 1.")
    return threshold_value


def _validate_top_n(top_n: int) -> None:
    """Validate top-N row count."""
    if top_n < 1:
        raise ErrorAnalysisError("top_n must be at least 1.")


def _percentage(numerator: int, denominator: int) -> float:
    """Return a 0-100 percentage with zero denominator protection."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator * 100.0)


def _safe_number(value: Any) -> float | None:
    """Return finite floats and convert missing/non-finite values to None."""
    if pd.isna(value):
        return None
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _json_safe_value(value: Any) -> Any:
    """Return plain Python scalar values suitable for strict JSON encoding."""
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_number(value)
    return value
