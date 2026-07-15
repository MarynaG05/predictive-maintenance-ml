"""Validation-only permutation explainability for predictive maintenance models."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from predictive_maintenance import config
from predictive_maintenance.data import load_dataset
from predictive_maintenance.evaluation import positive_class_probability
from predictive_maintenance.exceptions import ExplainabilityError
from predictive_maintenance.models import (
    HIST_GRADIENT_BOOSTING_MODEL_NAME,
    build_model_comparison_models,
)
from predictive_maintenance.splitting import split_dataset
from predictive_maintenance.train import _fit_model

FEATURE_IMPORTANCE_COLUMNS: tuple[str, ...] = (
    "rank",
    "feature",
    "importance_mean",
    "importance_std",
)
SCORING_METRIC = "average_precision"
PERMUTATION_N_JOBS = 1


def calculate_permutation_importance(
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_repeats: int = 10,
    random_state: int = config.RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Calculate validation-set permutation importance on original features.

    The complete fitted pipeline receives raw validation features exactly in
    ``config.MODEL_FEATURES`` order. This gives one importance value per logical
    input feature, including one categorical ``Type`` importance.
    """
    X_validation = _validate_inputs(X_validation, y_validation, n_repeats)

    try:
        result = permutation_importance(
            model,
            X_validation,
            y_validation,
            scoring=SCORING_METRIC,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=PERMUTATION_N_JOBS,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "Unable to calculate validation permutation importance for the fitted "
            "pipeline."
        ) from exc

    if len(result.importances_mean) != len(config.MODEL_FEATURES):
        raise ExplainabilityError(
            "Permutation importance result size does not match configured model "
            "features."
        )

    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            config.MODEL_FEATURES,
            result.importances_mean,
            result.importances_std,
            strict=True,
        )
    ]
    return rank_feature_importance(rows)


def rank_feature_importance(
    feature_importance: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank logical features by descending mean importance with stable ties."""
    ranked_rows = sorted(
        (dict(row) for row in feature_importance),
        key=lambda row: (-row["importance_mean"], row["feature"]),
    )
    return [
        {
            "rank": rank,
            "feature": row["feature"],
            "importance_mean": float(row["importance_mean"]),
            "importance_std": float(row["importance_std"]),
        }
        for rank, row in enumerate(ranked_rows, start=1)
    ]


def interpret_feature_importance(
    feature_importance: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return concise business interpretation for each feature importance row."""
    return [
        {
            "feature": row["feature"],
            "higher_importance_means": (
                "Validation average precision dropped more when this feature was "
                "permuted, so this fitted model relied on it more for validation "
                "discrimination."
            ),
            "does_not_mean": (
                "This does not prove causality, operational root cause, effect "
                "direction, or that increasing or decreasing the feature changes "
                "failure risk."
            ),
            "method_note": (
                "Permutation importance is model-specific and validation-split-"
                "specific. Correlated features can share or mask importance, and "
                "low importance does not prove a feature is useless."
            ),
            "scope_note": (
                "These results are validation diagnostics for Version 1 modeling, "
                "not production-ready causal conclusions."
            ),
        }
        for row in feature_importance
    ]


def run_validation_explainability(
    dataframe: pd.DataFrame | None = None,
    model_name: str = HIST_GRADIENT_BOOSTING_MODEL_NAME,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """Fit the selected model on train only and explain validation performance."""
    dataset = load_dataset() if dataframe is None else dataframe.copy()
    split_data = split_dataset(dataset)
    models = build_model_comparison_models()
    if model_name not in models:
        available_models = ", ".join(models)
        raise ExplainabilityError(
            f"Unknown model_name '{model_name}'. Available models: {available_models}."
        )

    model = models[model_name]
    try:
        _fit_model(model_name, model, split_data.X_train, split_data.y_train)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ExplainabilityError(
            f"Unable to fit model '{model_name}' for validation explainability."
        ) from exc

    baseline_average_precision = _baseline_validation_average_precision(
        model,
        split_data.X_validation,
        split_data.y_validation,
    )
    feature_importance = calculate_permutation_importance(
        model=model,
        X_validation=split_data.X_validation,
        y_validation=split_data.y_validation,
        n_repeats=n_repeats,
    )
    return {
        "model_name": model_name,
        "scoring_metric": SCORING_METRIC,
        "n_repeats": int(n_repeats),
        "random_seed": int(config.RANDOM_SEED),
        "baseline_validation_average_precision": baseline_average_precision,
        "split_row_counts": {
            "train": int(len(split_data.X_train)),
            "validation": int(len(split_data.X_validation)),
            "test": int(len(split_data.X_test)),
        },
        "feature_importance": feature_importance,
        "interpretation": interpret_feature_importance(feature_importance),
        "test_set_status": (
            "Final test set remains untouched; no test-set predictions, "
            "probabilities, metrics, or explanations were calculated."
        ),
    }


def save_explainability_report(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save feature importance JSON and CSV reports."""
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.REPORTS_DIR / "explainability"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "feature_importance.json"
    csv_path = report_dir / "feature_importance.csv"

    json_path.write_text(
        json.dumps(results, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results["feature_importance"]).to_csv(
        csv_path,
        columns=list(FEATURE_IMPORTANCE_COLUMNS),
        index=False,
    )
    return {"feature_importance_json": json_path, "feature_importance_csv": csv_path}


def plot_feature_importance(
    feature_importance: Sequence[dict[str, Any]],
    output_path: Path | str | None = None,
) -> Path:
    """Create a horizontal bar chart for validation permutation importance."""
    figure_path = (
        Path(output_path)
        if output_path is not None
        else config.FIGURES_DIR / "permutation_importance.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(reversed(feature_importance))
    labels = [row["feature"] for row in rows]
    values = [row["importance_mean"] for row in rows]
    errors = [row["importance_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, xerr=errors, capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Validation Permutation Importance")
    ax.set_xlabel("Average precision decrease after feature permutation")
    ax.set_ylabel("Original model feature")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def save_explainability_outputs(
    results: dict[str, Any],
    output_dir: Path | str | None = None,
    figure_path: Path | str | None = None,
) -> dict[str, Path]:
    """Save JSON, CSV, and figure outputs for explainability results."""
    paths = save_explainability_report(results, output_dir)
    paths["permutation_importance_png"] = plot_feature_importance(
        results["feature_importance"],
        output_path=figure_path,
    )
    return paths


def _baseline_validation_average_precision(
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> float:
    """Calculate baseline validation average precision for the fitted pipeline."""
    try:
        probabilities = model.predict_proba(X_validation)
        positive_probabilities = positive_class_probability(model, probabilities)
        return float(average_precision_score(y_validation, positive_probabilities))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "Unable to calculate baseline validation average precision."
        ) from exc


def _validate_inputs(
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_repeats: int,
) -> pd.DataFrame:
    """Validate explainability inputs before permutation scoring."""
    if not X_validation.index.equals(y_validation.index):
        raise ExplainabilityError(
            "X_validation and y_validation indices must match exactly."
        )
    if n_repeats < 1:
        raise ExplainabilityError("n_repeats must be at least 1.")

    missing_features = tuple(
        feature
        for feature in config.MODEL_FEATURES
        if feature not in X_validation.columns
    )
    if missing_features:
        columns = ", ".join(missing_features)
        raise ExplainabilityError(
            f"Missing required validation feature columns: {columns}"
        )

    return X_validation.loc[:, config.MODEL_FEATURES].copy()
