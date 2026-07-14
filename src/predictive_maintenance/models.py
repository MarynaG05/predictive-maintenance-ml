"""Baseline model factories for honest validation benchmarking."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictive_maintenance import config
from predictive_maintenance.preprocessing import build_preprocessor

DUMMY_PRIOR_MODEL_NAME = "dummy_prior"
LOGISTIC_REGRESSION_MODEL_NAME = "logistic_regression_balanced"


def build_baseline_models() -> dict[str, Pipeline]:
    """Build unfitted baseline pipelines with independent preprocessors."""
    return {
        DUMMY_PRIOR_MODEL_NAME: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DummyClassifier(strategy="prior", random_state=config.RANDOM_SEED),
                ),
            ]
        ),
        LOGISTIC_REGRESSION_MODEL_NAME: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.RANDOM_SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
    }
