# Results Summary

This document records the approved Version 1 model results. Validation metrics and final test metrics are separated intentionally because they serve different purposes.

## Validation Model Comparison

Average precision was used as the primary validation selection metric because the failure class is rare.

| Model | Validation average precision |
| --- | ---: |
| `dummy_prior` | 0.0340 |
| `logistic_regression_balanced` | 0.3679 |
| `random_forest_balanced` | 0.6448 |
| `hist_gradient_boosting_balanced` | 0.7173 |

Selected model: `hist_gradient_boosting_balanced`.

## Validation Threshold Profiles

Thresholds were selected on validation data only.

| Profile | Threshold | Precision | Recall | F1 | False positives | False negatives | Predicted positives | Illustrative cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `high_recall` | 0.05 | 0.3631 | 0.8971 | 0.5169 | 107 | 7 | 168 | 177.0 |
| `balanced` | 0.80 | 0.7288 | 0.6324 | 0.6772 | 16 | 25 | 59 | 266.0 |
| `conservative` | 0.80 | 0.7288 | 0.6324 | 0.6772 | 16 | 25 | 59 | 266.0 |

The `balanced` and `conservative` profiles share threshold `0.8` because different validation selection rules converged on the same operating point.

## Validation Error Analysis

Primary threshold: best-F1 validation threshold `0.8`.

| Metric | Value |
| --- | ---: |
| Validation precision | 0.7288 |
| Validation recall | 0.6324 |
| Validation F1 | 0.6772 |
| True negatives | 1916 |
| False positives | 16 |
| False negatives | 25 |
| True positives | 43 |

False positives represent unnecessary maintenance reviews. False negatives represent missed failure cases. These results describe validation behavior and were not used as final test evidence.

## Validation Explainability

Permutation importance was computed on validation data using average-precision degradation after feature permutation.

| Rank | Feature | Importance mean |
| ---: | --- | ---: |
| 1 | `Torque [Nm]` | 0.514849 |
| 2 | `Air temperature [K]` | 0.362717 |
| 3 | `Rotational speed [rpm]` | 0.313981 |
| 4 | `Tool wear [min]` | 0.251310 |
| 5 | `Process temperature [K]` | 0.230222 |
| 6 | `Type` | 0.041294 |

Importance is model-specific and validation-split-specific. It does not imply causality or direction of effect.

## Final Held-Out Test Results

The final model and operating threshold were frozen before final test evaluation.

| Item | Value |
| --- | --- |
| Final model | `hist_gradient_boosting_balanced` |
| Operating profile | `balanced` |
| Selected threshold | `0.8` |
| Development rows | 8000 |
| Test rows | 2000 |
| Test class distribution | 1932 no failure / 68 failure |

Final test results at threshold `0.8`:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9639 |
| Average precision | 0.8351 |
| Accuracy | 0.9875 |
| Precision | 0.8644 |
| Recall | 0.7500 |
| F1 | 0.8031 |
| True negatives | 1924 |
| False positives | 8 |
| False negatives | 17 |
| True positives | 51 |
| Predicted positives | 59 |
| Predicted positive rate | 0.0295 |

For reference, the default threshold `0.5` produced:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9815 |
| Precision | 0.7013 |
| Recall | 0.7941 |
| F1 | 0.7448 |
| True negatives | 1909 |
| False positives | 23 |
| False negatives | 14 |
| True positives | 54 |

## Interpretation Boundary

Final test results are descriptive final evidence for the frozen Version 1 workflow. They are not a new optimization signal. Any model change, threshold change, feature change, or hyperparameter change after reviewing the final test set would require a new untouched test set or external validation dataset.
