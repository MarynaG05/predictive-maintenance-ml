"""Final one-time test-set evaluation for the fixed predictive model."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import positive_class_probability
from predictive_maintenance.exceptions import FinalEvaluationError, ModelEvaluationError
from predictive_maintenance.models import (
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    build_model_comparison_models,
)
from predictive_maintenance.recommendations import run_business_threshold_recommendation
from predictive_maintenance.splitting import SplitData, split_dataset
from predictive_maintenance.train import _fit_model

FINAL_MODEL_NAME = HIST_GRADIENT_BOOSTING_MODEL_NAME
FINAL_OPERATING_PROFILE = "balanced"
DEFAULT_THRESHOLD = 0.5
FINAL_EVALUATION_COLUMNS: tuple[str, ...] = (
    "operating_point",
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
    "roc_auc",
    "average_precision",
)


def run_final_model_evaluation(
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the approved one-time final evaluation on the untouched test split."""
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    recommendations = run_business_threshold_recommendation(
        dataset,
        model_name=FINAL_MODEL_NAME,
    )
    selected_threshold = _selected_balanced_threshold(recommendations)

    split_data = split_dataset(dataset)
    X_development, y_development = _combine_train_validation(split_data)
    _validate_final_split(X_development, y_development, split_data)
    _validate_binary_test_target(split_data.y_test)

    model = _build_final_model()
    try:
        _fit_model(FINAL_MODEL_NAME, model, X_development, y_development)
    except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        raise FinalEvaluationError("Unable to fit the final model.") from exc

    try:
        probability_matrix = model.predict_proba(split_data.X_test)
        test_probabilities = positive_class_probability(model, probability_matrix)
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        ModelEvaluationError,
    ) as exc:
        raise FinalEvaluationError(
            "Unable to generate class-1 probabilities for the final test set."
        ) from exc
    test_probabilities = _validate_final_probabilities(
        test_probabilities,
        expected_length=len(split_data.y_test),
    )

    try:
        roc_auc = float(roc_auc_score(split_data.y_test, test_probabilities))
        average_precision = float(
            average_precision_score(split_data.y_test, test_probabilities)
        )
    except ValueError as exc:
        raise FinalEvaluationError(
            "Final test target must contain both classes for ranking metrics."
        ) from exc

    default_metrics = _evaluate_fixed_threshold(
        split_data.y_test,
        test_probabilities,
        DEFAULT_THRESHOLD,
    )
    selected_metrics = _evaluate_fixed_threshold(
        split_data.y_test,
        test_probabilities,
        selected_threshold,
    )

    return {
        "model_name": FINAL_MODEL_NAME,
        "operating_profile": FINAL_OPERATING_PROFILE,
        "selected_threshold": selected_threshold,
        "model_selection_source": (
            "Selected before final test evaluation from the validation-only model "
            "comparison workflow."
        ),
        "threshold_selection_source": (
            "Balanced operating threshold derived before final test evaluation from "
            "the validation-only business recommendation workflow."
        ),
        "development_row_count": int(len(X_development)),
        "test_row_count": int(len(split_data.X_test)),
        "development_class_distribution": _class_distribution(y_development),
        "test_class_distribution": _class_distribution(split_data.y_test),
        "test_roc_auc": roc_auc,
        "test_average_precision": average_precision,
        "default_threshold_metrics": default_metrics,
        "selected_threshold_metrics": selected_metrics,
        "final_evaluation_statement": (
            "Model and threshold were fixed before test evaluation. Test data were "
            "used once for final evaluation, and no test-driven model or threshold "
            "changes were made."
        ),
        "governance": {
            "final_configuration_frozen_before_test": True,
            "test_evaluation_count": 1,
            "test_driven_changes_allowed": False,
            "process_note": (
                "Running this function repeatedly reproduces the same deterministic "
                "result; the project process treats the first approved run as the "
                "final evaluation. Future model changes require a new untouched "
                "dataset or external validation set."
            ),
        },
        "limitations": (
            "Final test metrics are an honest holdout estimate for this synthetic "
            "AI4I dataset, not a guarantee of production performance."
        ),
        "reproducibility": {
            "random_seed": int(config.RANDOM_SEED),
            "split_strategy": "deterministic stratified 60/20/20 split",
            "development_source": "train + validation",
            "test_source": "untouched held-out test split",
        },
    }


def save_final_evaluation_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save final evaluation results as strict JSON and deterministic CSV."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "final_evaluation"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "final_evaluation.json"
    csv_path = report_dir / "final_evaluation.csv"

    json_path.write_text(
        json.dumps(results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(_final_evaluation_rows(results)).to_csv(
        csv_path,
        columns=list(FINAL_EVALUATION_COLUMNS),
        index=False,
    )
    return {"final_evaluation_json": json_path, "final_evaluation_csv": csv_path}


def plot_final_confusion_matrices(
    results: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """Plot final test confusion matrices for default and selected thresholds."""
    figure_path = (
        Path(output_path)
        if output_path is not None
        else config.FIGURES_DIR / "final_test_confusion_matrices.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    matrices = [
        ("Default threshold 0.5", results["default_threshold_metrics"]),
        (
            f"Selected {results['operating_profile']} threshold "
            f"{results['selected_threshold']}",
            results["selected_threshold_metrics"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (title, metrics) in zip(np.ravel(axes), matrices, strict=True):
        matrix = np.array(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
        )
        ax.imshow(matrix)
        ax.set_title(f"Final test: {title}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["0", "1"])
        ax.set_yticks([0, 1], labels=["0", "1"])
        for row_index in range(2):
            for column_index in range(2):
                ax.text(
                    column_index,
                    row_index,
                    str(int(matrix[row_index, column_index])),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def save_final_evaluation_outputs(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
    figure_path: Path | str | None = None,
) -> dict[str, Path]:
    """Save JSON, CSV, and confusion-matrix figure for final evaluation."""
    paths = save_final_evaluation_report(results, output_dir)
    paths["confusion_matrices_figure"] = plot_final_confusion_matrices(
        results,
        output_path=figure_path,
    )
    return paths


def _selected_balanced_threshold(recommendations: dict[str, Any]) -> float:
    """Return the fixed balanced recommendation threshold."""
    try:
        profile = recommendations["profiles"][FINAL_OPERATING_PROFILE]
        threshold = float(profile["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalEvaluationError(
            "Recommendation output must include a balanced profile threshold."
        ) from exc
    if threshold <= 0.0 or threshold >= 1.0 or not np.isfinite(threshold):
        raise FinalEvaluationError(
            "Balanced operating threshold must be finite and strictly between 0 and 1."
        )
    return threshold


def _combine_train_validation(
    split_data: SplitData,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine train and validation splits into the final development set."""
    X_development = pd.concat(
        [split_data.X_train, split_data.X_validation],
        axis=0,
    )
    y_development = pd.concat(
        [split_data.y_train, split_data.y_validation],
        axis=0,
    )
    return X_development.copy(), y_development.copy()


def _validate_final_split(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    split_data: SplitData,
) -> None:
    """Validate final split traceability through unique original indices."""
    if not X_development.index.is_unique:
        raise FinalEvaluationError("Development feature indices must be unique.")
    if not y_development.index.is_unique:
        raise FinalEvaluationError("Development label indices must be unique.")
    if not split_data.X_test.index.is_unique:
        raise FinalEvaluationError("Test feature indices must be unique.")
    if not split_data.y_test.index.is_unique:
        raise FinalEvaluationError("Test label indices must be unique.")
    if not X_development.index.equals(y_development.index):
        raise FinalEvaluationError("Development features and labels must align.")
    if not split_data.X_test.index.equals(split_data.y_test.index):
        raise FinalEvaluationError("Test features and labels must align.")
    if set(X_development.index).intersection(set(split_data.X_test.index)):
        raise FinalEvaluationError("Development and test indices must be disjoint.")


def _build_final_model() -> Any:
    """Build the fixed final model from the existing comparison factory."""
    models = build_model_comparison_models()
    if FINAL_MODEL_NAME not in models:
        raise FinalEvaluationError(
            f"Selected final model '{FINAL_MODEL_NAME}' is not available."
        )
    return models[FINAL_MODEL_NAME]


def _validate_binary_test_target(y_test: pd.Series) -> None:
    """Validate that the final test target supports binary ranking metrics."""
    observed = set(y_test.dropna().unique())
    if observed != {0, 1}:
        labels = ", ".join(str(label) for label in sorted(observed, key=str))
        raise FinalEvaluationError(
            "Final test target must contain both classes 0 and 1; "
            f"observed labels: {labels}."
        )


def _validate_final_probabilities(
    probabilities: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Return a finite class-1 probability vector aligned to final test rows."""
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1:
        raise FinalEvaluationError(
            "Final class-1 probabilities must be a one-dimensional array."
        )
    if len(probability_array) != expected_length:
        raise FinalEvaluationError(
            "Final class-1 probabilities must match the number of test rows."
        )
    if not np.isfinite(probability_array).all():
        raise FinalEvaluationError("Final class-1 probabilities must be finite.")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise FinalEvaluationError(
            "Final class-1 probabilities must be within the [0, 1] range."
        )
    return probability_array


def _evaluate_fixed_threshold(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one fixed operating threshold on the final test probabilities."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    predicted_positive_count = int(predictions.sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": float(predicted_positive_count / len(predictions)),
    }


def _final_evaluation_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic CSV rows for the two final operating points."""
    rows = []
    for operating_point, metric_key in (
        ("default_0_5", "default_threshold_metrics"),
        ("selected_balanced_threshold", "selected_threshold_metrics"),
    ):
        metrics = results[metric_key]
        row = {
            "operating_point": operating_point,
            **metrics,
            "roc_auc": results["test_roc_auc"],
            "average_precision": results["test_average_precision"],
        }
        rows.append(row)
    return rows


def _class_distribution(target: pd.Series) -> dict[str, int]:
    """Return deterministic class counts with JSON-friendly keys."""
    counts = target.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}
