import json
from pathlib import Path

import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.models import (
    DUMMY_PRIOR_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
)
from predictive_maintenance.train import (
    _best_model_name,
    run_baseline_training,
    save_baseline_report,
)


def make_training_dataframe() -> pd.DataFrame:
    rows = []
    labels = [0] * 80 + [1] * 20
    for index, target in enumerate(labels, start=1):
        rows.append(
            {
                "UDI": index,
                "Product ID": f"ID{index:05d}",
                "Type": ("L", "M", "H")[index % 3],
                "Air temperature [K]": 298.0 + index * 0.01,
                "Process temperature [K]": 308.0 + index * 0.01,
                "Rotational speed [rpm]": 1400 + index,
                "Torque [Nm]": 40.0 + target * 5.0 + index * 0.01,
                "Tool wear [min]": index % 200,
                "Machine failure": target,
                "TWF": target,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )

    return pd.DataFrame(rows, columns=config.REQUIRED_COLUMNS)


def test_run_baseline_training_fits_and_evaluates_both_models() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert set(results["validation_metrics"]) == {
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
    }
    assert results["best_validation_model"] in results["validation_metrics"]


def test_run_baseline_training_reports_only_validation_metrics() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert "validation_metrics" in results
    assert "test_metrics" not in results
    assert "no test-set metrics were calculated" in results["test_set_status"]


def test_run_baseline_training_reports_split_sizes_and_class_distributions() -> None:
    results = run_baseline_training(make_training_dataframe())

    assert results["split_row_counts"] == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert results["class_distributions"] == {
        "train": {"0": 48, "1": 12},
        "validation": {"0": 16, "1": 4},
        "test": {"0": 16, "1": 4},
    }


def test_save_baseline_report_writes_json_and_model_comparison(
    tmp_path: Path,
) -> None:
    results = run_baseline_training(make_training_dataframe())

    output_paths = save_baseline_report(results, tmp_path)

    assert output_paths["metrics_json"] == tmp_path / "baseline_metrics.json"
    assert output_paths["model_comparison_csv"] == tmp_path / "model_comparison.csv"
    assert output_paths["metrics_json"].is_file()
    assert output_paths["model_comparison_csv"].is_file()
    json.loads(output_paths["metrics_json"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(output_paths["model_comparison_csv"])
    assert set(comparison["model"]) == {
        DUMMY_PRIOR_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
    }


def test_best_validation_model_breaks_average_precision_ties_by_name() -> None:
    validation_metrics = {
        "z_model": {"average_precision": 0.5},
        "a_model": {"average_precision": 0.5},
    }

    assert _best_model_name(validation_metrics) == "a_model"
