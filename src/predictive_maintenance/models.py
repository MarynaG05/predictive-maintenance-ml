"""Model factories for honest validation benchmarking."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictive_maintenance import config
from predictive_maintenance.preprocessing import build_preprocessor

DUMMY_PRIOR_MODEL_NAME = "dummy_prior"
LOGISTIC_REGRESSION_MODEL_NAME = "logistic_regression_balanced"
RANDOM_FOREST_MODEL_NAME = "random_forest_balanced"
HIST_GRADIENT_BOOSTING_MODEL_NAME = "hist_gradient_boosting_balanced"


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


def build_model_comparison_models() -> dict[str, Pipeline]:
    """Build unfitted baseline and candidate model pipelines."""
    models = build_baseline_models()
    models.update(
        {
            RANDOM_FOREST_MODEL_NAME: Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=config.RANDOM_SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            HIST_GRADIENT_BOOSTING_MODEL_NAME: Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        }
    )
    return models
