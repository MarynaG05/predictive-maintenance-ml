"""Validation-set evaluation utilities for binary classifiers."""

from typing import Any

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

from predictive_maintenance.exceptions import ModelEvaluationError


def evaluate_binary_classifier(
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    """Evaluate a fitted binary classifier on validation data only."""
    if not hasattr(model, "predict_proba"):
        raise ModelEvaluationError(
            "Model must provide predict_proba for ROC-AUC and PR-AUC evaluation."
        )

    try:
        y_pred = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)
    except AttributeError as exc:
        raise ModelEvaluationError(
            "Model must be fitted and provide probability predictions."
        ) from exc

    y_score = positive_class_probability(model, probabilities)
    tn, fp, fn, tp = confusion_matrix(y_validation, y_pred, labels=[0, 1]).ravel()
    _validate_binary_evaluation_target(y_validation)

    try:
        roc_auc = roc_auc_score(y_validation, y_score)
        average_precision = average_precision_score(y_validation, y_score)
    except ValueError as exc:
        raise ModelEvaluationError(
            "Validation target must contain both classes to calculate ROC-AUC "
            "and average precision."
        ) from exc

    return {
        "accuracy": float(accuracy_score(y_validation, y_pred)),
        "precision": float(
            precision_score(y_validation, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_validation, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_validation, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def evaluate_models(
    models: dict[str, Any],
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Evaluate fitted models on the validation split."""
    return {
        name: evaluate_binary_classifier(model, X_validation, y_validation)
        for name, model in models.items()
    }


def positive_class_probability(model: Any, probabilities: Any) -> np.ndarray:
    """Return probability scores for class 1 using the model's class ordering."""
    classes = _model_classes(model)
    if classes is None:
        raise ModelEvaluationError(
            "Model must expose classes_ so the class-1 probability column can "
            "be selected safely."
        )

    probability_array = np.asarray(probabilities)
    if probability_array.ndim != 2 or probability_array.shape[1] != len(classes):
        raise ModelEvaluationError(
            "Model predict_proba output shape must match the number of classes."
        )

    positive_class_indices = np.flatnonzero(classes == 1)
    if len(positive_class_indices) == 0:
        raise ModelEvaluationError("Model classes_ must include class 1.")

    return probability_array[:, int(positive_class_indices[0])]


def _positive_class_probability(model: Any, probabilities: Any) -> np.ndarray:
    """Backward-compatible wrapper for class-1 probability selection."""
    return positive_class_probability(model, probabilities)


def _model_classes(model: Any) -> np.ndarray | None:
    """Return fitted class labels from an estimator or sklearn Pipeline."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    if hasattr(model, "steps") and model.steps:
        final_estimator = model.steps[-1][1]
        final_classes = getattr(final_estimator, "classes_", None)
        if final_classes is not None:
            return np.asarray(final_classes)

    return None


def _validate_binary_evaluation_target(target: pd.Series) -> None:
    """Validate that ranking metrics can be calculated for class-1 evaluation."""
    observed_labels = set(target.dropna().unique())
    if observed_labels != {0, 1}:
        labels = ", ".join(str(label) for label in sorted(observed_labels, key=str))
        raise ModelEvaluationError(
            "Validation target must contain both classes 0 and 1 to calculate "
            f"ranking metrics; observed labels: {labels}."
        )
