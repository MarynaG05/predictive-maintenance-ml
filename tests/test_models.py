import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from predictive_maintenance import config
from predictive_maintenance.models import (
    DUMMY_PRIOR_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
    build_baseline_models,
)


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    labels = [0, 0, 0, 0, 1, 1]
    for index, target in enumerate(labels, start=1):
        rows.append(
            {
                "Type": ("L", "M", "H")[index % 3],
                "Air temperature [K]": 298.0 + index,
                "Process temperature [K]": 308.0 + index,
                "Rotational speed [rpm]": 1400 + index * 10,
                "Torque [Nm]": 40.0 + target * 8.0,
                "Tool wear [min]": index * 5,
            }
        )
    return pd.DataFrame(rows, columns=config.MODEL_FEATURES), pd.Series(labels)


def test_build_baseline_models_returns_expected_names() -> None:
    assert set(build_baseline_models()) == {
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
    }


def test_each_baseline_model_is_complete_pipeline() -> None:
    models = build_baseline_models()

    for model in models.values():
        assert isinstance(model, Pipeline)
        assert "preprocessor" in model.named_steps
        assert "classifier" in model.named_steps


def test_dummy_classifier_uses_prior_strategy() -> None:
    classifier = build_baseline_models()[DUMMY_PRIOR_MODEL_NAME].named_steps[
        "classifier"
    ]

    assert isinstance(classifier, DummyClassifier)
    assert classifier.strategy == "prior"


def test_logistic_regression_uses_balanced_class_weight() -> None:
    classifier = build_baseline_models()[LOGISTIC_REGRESSION_MODEL_NAME].named_steps[
        "classifier"
    ]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced"


def test_baseline_pipelines_can_fit_and_predict() -> None:
    features, target = make_training_data()

    for model in build_baseline_models().values():
        model.fit(features, target)
        predictions = model.predict(features)

        assert len(predictions) == len(features)


def test_baseline_pipelines_have_independent_preprocessors() -> None:
    models = build_baseline_models()
    dummy_preprocessor = models[DUMMY_PRIOR_MODEL_NAME].named_steps["preprocessor"]
    logistic_preprocessor = models[LOGISTIC_REGRESSION_MODEL_NAME].named_steps[
        "preprocessor"
    ]

    assert dummy_preprocessor is not logistic_preprocessor

    features, target = make_training_data()
    models[DUMMY_PRIOR_MODEL_NAME].fit(features, target)

    check_is_fitted(dummy_preprocessor)
    with pytest.raises(NotFittedError):
        check_is_fitted(logistic_preprocessor)
