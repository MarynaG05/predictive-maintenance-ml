import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from predictive_maintenance.evaluation import evaluate_binary_classifier
from predictive_maintenance.exceptions import ModelEvaluationError


class ControlledProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.predictions = np.array([0, 1, 1, 0])
        self.probabilities = np.array(
            [
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions[: len(X)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.probabilities[: len(X)]


class NoPositivePredictionModel:
    classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.full(len(X), 0.9),
                np.full(len(X), 0.1),
            ]
        )


class NoProbabilityModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


class ProbabilityModelWithoutClasses:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.full(len(X), 0.9),
                np.full(len(X), 0.1),
            ]
        )


def make_validation_data() -> tuple[pd.DataFrame, pd.Series]:
    return pd.DataFrame({"feature": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1])


def test_evaluate_binary_classifier_returns_expected_metrics() -> None:
    X_validation, y_validation = make_validation_data()

    metrics = evaluate_binary_classifier(
        ControlledProbabilityModel(),
        X_validation,
        y_validation,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
    }


def test_evaluate_binary_classifier_confusion_matrix_values_are_correct() -> None:
    X_validation, y_validation = make_validation_data()

    metrics = evaluate_binary_classifier(
        ControlledProbabilityModel(),
        X_validation,
        y_validation,
    )

    assert metrics["confusion_matrix"] == {
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives": 1,
    }
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_evaluate_binary_classifier_uses_probabilities_for_ranking_metrics() -> None:
    X_validation, y_validation = make_validation_data()
    model = ControlledProbabilityModel()

    metrics = evaluate_binary_classifier(model, X_validation, y_validation)

    assert metrics["roc_auc"] == roc_auc_score(y_validation, model.probabilities[:, 1])
    assert metrics["average_precision"] == average_precision_score(
        y_validation,
        model.probabilities[:, 1],
    )


def test_evaluate_binary_classifier_uses_actual_class_one_probability_column() -> None:
    class ReversedClassModel:
        classes_ = np.array([1, 0])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 1, 1, 0])[: len(X)]

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    [0.1, 0.9],
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.4, 0.6],
                ]
            )[: len(X)]

    X_validation, y_validation = make_validation_data()
    model = ReversedClassModel()

    metrics = evaluate_binary_classifier(model, X_validation, y_validation)

    assert metrics["roc_auc"] == roc_auc_score(y_validation, [0.1, 0.7, 0.8, 0.4])
    assert metrics["roc_auc"] == 0.75


def test_evaluation_metrics_are_json_serializable() -> None:
    X_validation, y_validation = make_validation_data()
    metrics = evaluate_binary_classifier(
        ControlledProbabilityModel(),
        X_validation,
        y_validation,
    )

    json.dumps(metrics, allow_nan=False)


def test_evaluate_binary_classifier_handles_zero_division() -> None:
    X_validation, y_validation = make_validation_data()

    metrics = evaluate_binary_classifier(
        NoPositivePredictionModel(),
        X_validation,
        y_validation,
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_missing_predict_proba_raises_clear_error() -> None:
    X_validation, y_validation = make_validation_data()

    with pytest.raises(ModelEvaluationError, match="predict_proba"):
        evaluate_binary_classifier(
            NoProbabilityModel(),
            X_validation,
            y_validation,
        )


def test_classes_unavailable_raises_clear_error() -> None:
    X_validation, y_validation = make_validation_data()

    with pytest.raises(ModelEvaluationError, match="classes_"):
        evaluate_binary_classifier(
            ProbabilityModelWithoutClasses(),
            X_validation,
            y_validation,
        )


def test_class_one_missing_raises_clear_error() -> None:
    class MissingClassOneModel:
        classes_ = np.array([0, 2])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.column_stack(
                [
                    np.full(len(X), 0.9),
                    np.full(len(X), 0.1),
                ]
            )

    X_validation, y_validation = make_validation_data()

    with pytest.raises(ModelEvaluationError, match="include class 1"):
        evaluate_binary_classifier(
            MissingClassOneModel(),
            X_validation,
            y_validation,
        )


def test_single_probability_column_raises_clear_error() -> None:
    class OneColumnProbabilityModel:
        classes_ = np.array([0, 1])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> Any:
            return np.ones((len(X), 1))

    X_validation, y_validation = make_validation_data()

    with pytest.raises(ModelEvaluationError, match="shape"):
        evaluate_binary_classifier(
            OneColumnProbabilityModel(),
            X_validation,
            y_validation,
        )


def test_probability_matrix_class_count_mismatch_raises_clear_error() -> None:
    class MismatchedProbabilityModel:
        classes_ = np.array([0, 1])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.ones((len(X), 3))

    X_validation, y_validation = make_validation_data()

    with pytest.raises(ModelEvaluationError, match="shape"):
        evaluate_binary_classifier(
            MismatchedProbabilityModel(),
            X_validation,
            y_validation,
        )


def test_one_class_validation_target_raises_model_evaluation_error() -> None:
    X_validation, _ = make_validation_data()
    y_validation = pd.Series([0, 0, 0, 0])

    with pytest.raises(ModelEvaluationError, match="both classes"):
        evaluate_binary_classifier(
            ControlledProbabilityModel(),
            X_validation,
            y_validation,
        )
