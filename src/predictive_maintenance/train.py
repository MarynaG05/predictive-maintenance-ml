"""Training orchestration for baseline predictive maintenance models."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import evaluate_models
from predictive_maintenance.models import build_baseline_models
from predictive_maintenance.splitting import SplitData, split_dataset


def run_baseline_training(dataframe: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fit baseline models on training data and evaluate validation metrics."""
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    split_data = split_dataset(dataset)
    models = build_baseline_models()

    for model in models.values():
        model.fit(split_data.X_train, split_data.y_train)

    validation_metrics = evaluate_models(
        models,
        split_data.X_validation,
        split_data.y_validation,
    )

    return {
        "split_row_counts": _split_row_counts(split_data),
        "class_distributions": _class_distributions(split_data),
        "validation_metrics": validation_metrics,
        "best_validation_model": _best_model_name(validation_metrics),
        "test_set_status": (
            "Final test set remains untouched; no test-set metrics were calculated."
        ),
    }


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
        json.dumps(results, allow_nan=False, indent=2, sort_keys=True),
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
        row.update(metrics["confusion_matrix"])
        rows.append(row)

    return pd.DataFrame(rows)
