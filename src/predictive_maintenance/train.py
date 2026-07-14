"""Training orchestration for baseline predictive maintenance models."""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import evaluate_models
from predictive_maintenance.models import (
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    build_baseline_models,
    build_model_comparison_models,
)
from predictive_maintenance.splitting import SplitData, split_dataset

MODEL_COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "training_seconds",
)


def run_baseline_training(dataframe: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fit baseline models on training data and evaluate validation metrics."""
    return _run_training_workflow(dataframe, build_baseline_models())


def run_model_comparison(dataframe: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fit baseline and candidate models and evaluate validation metrics."""
    return _run_training_workflow(dataframe, build_model_comparison_models())


def _run_training_workflow(
    dataframe: pd.DataFrame | None,
    models: dict[str, Pipeline],
) -> dict[str, Any]:
    """Run a train-only fit and validation-only model comparison workflow."""
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    split_data = split_dataset(dataset)
    training_durations: dict[str, float] = {}

    for model_name, model in models.items():
        training_durations[model_name] = _fit_model(
            model_name,
            model,
            split_data.X_train,
            split_data.y_train,
        )

    validation_metrics = evaluate_models(
        models,
        split_data.X_validation,
        split_data.y_validation,
    )
    for model_name, training_seconds in training_durations.items():
        validation_metrics[model_name]["training_seconds"] = training_seconds

    return {
        "split_row_counts": _split_row_counts(split_data),
        "class_distributions": _class_distributions(split_data),
        "validation_metrics": validation_metrics,
        "training_durations": training_durations,
        "best_validation_model": _best_model_name(validation_metrics),
        "test_set_status": (
            "Final test set remains untouched; no test-set metrics were calculated."
        ),
    }


def _fit_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Fit one model on training data and return elapsed seconds."""
    fit_parameters: dict[str, Any] = {}
    if model_name == HIST_GRADIENT_BOOSTING_MODEL_NAME:
        # HistGradientBoostingClassifier has no stable class_weight API across
        # supported sklearn versions, so class imbalance is handled with
        # training-only sample weights passed to the classifier step.
        fit_parameters["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

    start_time = time.monotonic()
    model.fit(X_train, y_train, **fit_parameters)
    return float(time.monotonic() - start_time)


def save_baseline_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save baseline validation metrics and comparison tables."""
    report_dir = (
        Path(output_dir) if output_dir is not None else config.REPORTS_DIR / "baseline"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "baseline_metrics.json"
    comparison_path = report_dir / "model_comparison.csv"

    metrics_path.write_text(
        json.dumps(results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    _model_comparison_dataframe(results["validation_metrics"]).to_csv(
        comparison_path,
        index=False,
    )

    return {
        "metrics_json": metrics_path,
        "model_comparison_csv": comparison_path,
    }


def save_model_comparison_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save validation metrics for baseline and candidate model comparison."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "model_comparison"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "model_comparison_metrics.json"
    comparison_path = report_dir / "model_comparison.csv"

    metrics_path.write_text(
        json.dumps(results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    _model_comparison_dataframe(results["validation_metrics"]).to_csv(
        comparison_path,
        columns=list(MODEL_COMPARISON_COLUMNS),
        index=False,
    )

    return {
        "metrics_json": metrics_path,
        "model_comparison_csv": comparison_path,
    }


def _split_row_counts(split_data: SplitData) -> dict[str, int]:
    """Return row counts for each split."""
    return {
        "train": int(len(split_data.X_train)),
        "validation": int(len(split_data.X_validation)),
        "test": int(len(split_data.X_test)),
    }


def _class_distributions(split_data: SplitData) -> dict[str, dict[str, int]]:
    """Return class counts for each split using JSON-friendly keys."""
    return {
        "train": _series_class_counts(split_data.y_train),
        "validation": _series_class_counts(split_data.y_validation),
        "test": _series_class_counts(split_data.y_test),
    }


def _series_class_counts(series: pd.Series) -> dict[str, int]:
    """Return deterministic class counts for a target series."""
    counts = series.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _best_model_name(validation_metrics: dict[str, dict[str, Any]]) -> str:
    """Select the best model by average precision, breaking ties by name."""
    return sorted(
        validation_metrics,
        key=lambda name: (-validation_metrics[name]["average_precision"], name),
    )[0]


def _model_comparison_dataframe(
    validation_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Flatten validation metrics into a compact comparison table."""
    rows: list[dict[str, Any]] = []
    for model_name, metrics in validation_metrics.items():
        row = {"model": model_name}
        row.update(
            {key: value for key, value in metrics.items() if key != "confusion_matrix"}
        )
        row.update(
            {
                "true_negative": metrics["confusion_matrix"]["true_negatives"],
                "false_positive": metrics["confusion_matrix"]["false_positives"],
                "false_negative": metrics["confusion_matrix"]["false_negatives"],
                "true_positive": metrics["confusion_matrix"]["true_positives"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
