# Portfolio Case Study: Predictive Maintenance ML Pipeline

This portfolio project demonstrates how I would approach a client engagement for machine-failure risk modeling using a reproducible Python machine-learning workflow.

## 1. Client-Style Problem

A maintenance team wants to identify machines that are operating under higher-risk conditions before a failure occurs. Unexpected failures can interrupt production, increase emergency repair costs, and overload maintenance teams with reactive work.

The client needs a workflow that supports inspection prioritization while making model assumptions, limitations, and trade-offs clear.

## 2. Project Objective

The objective is to build a binary classification workflow that predicts machine failure risk from operating-condition data. The workflow must be reproducible, leakage-safe, and understandable to both technical reviewers and business stakeholders.

The model is intended for decision support, not autonomous maintenance action.

## 3. Constraints and Risks

- Failure events are rare, so accuracy alone is not a reliable success metric.
- Identifier fields and failure-mode indicators can create target leakage if used as model features.
- The AI4I 2020 dataset is synthetic, so results demonstrate methodology rather than production performance.
- Business costs are illustrative because real downtime, inspection, and safety costs are not available.
- A single train-validation-test split is useful for portfolio demonstration but should be extended with external validation in a real engagement.

## 4. Solution Architecture

The project uses a staged architecture:

1. Load the local CSV file.
2. Validate required columns and duplicate source headers.
3. Profile the validated dataset.
4. Run exploratory analysis.
5. Split data into deterministic train, validation, and test sets.
6. Fit preprocessing and models inside scikit-learn pipelines.
7. Compare candidate models on validation data.
8. Select validation-derived operating thresholds.
9. Analyze validation errors and feature importance.
10. Freeze the configuration and evaluate once on the held-out test set.

This structure separates data quality, modeling, validation, business interpretation, and final evaluation.

## 5. Data-Quality Safeguards

The ingestion layer rejects missing required columns, duplicate source CSV headers, empty datasets, and header-only files. The loader returns columns in deterministic order.

Feature handling is explicit:

- `UDI` and `Product ID` are identifiers kept for traceability, not model training.
- `Machine failure` is the binary target.
- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are failure-mode indicators excluded from model features because they are target-derived.
- `Type` and the operating-condition measurements are the predictive features.

## 6. Modeling Approach

The workflow compares:

- `dummy_prior`
- `logistic_regression_balanced`
- `random_forest_balanced`
- `hist_gradient_boosting_balanced`

Average precision is the primary validation metric because the positive failure class is rare. The selected model is `hist_gradient_boosting_balanced`, which achieved validation average precision `0.7173`, the strongest result among the candidate models.

Preprocessing is handled through scikit-learn pipelines so transformations are fit only on the appropriate training data.

## 7. Threshold and Business Trade-Offs

The project evaluates operating thresholds on validation data only. Three business profiles are produced:

- `high_recall`: threshold `0.05`, intended to reduce missed failures.
- `balanced`: threshold `0.8`, selected using the best validation F1 score.
- `conservative`: threshold `0.8`, intended to reduce false alarms.

In this validation split, `balanced` and `conservative` converge to the same threshold. That means different rules selected the same operating point under the current validation results; it does not mean the business objectives are identical.

## 8. Final Results

After model and threshold selection were frozen, the final model was refit on development data, defined as train plus validation, and evaluated on the untouched test set.

Final held-out test results at threshold `0.8`:

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

These results are descriptive final evidence for this dataset and split, not a new optimization signal.

## 9. Deliverables

The repository includes:

- Professional Python package structure
- Dataset acquisition documentation
- Structural schema validation
- Dataset profiling workflow
- Executed exploratory notebook
- Baseline and candidate model comparison
- Validation threshold analysis
- Validation error analysis
- Validation permutation importance
- Business threshold recommendations
- Final held-out test evaluation
- Automated tests and GitHub Actions CI
- Portfolio-ready README and technical documentation

## 10. What I Would Do Next in a Real Client Engagement

For a real industrial client, I would next validate the workflow on real historical equipment data, confirm label quality, replace illustrative costs with business-approved costs, calibrate probabilities if needed, and evaluate performance across time, asset types, and operating regimes.

Depending on the deployment context, the next phase could include batch inference, model persistence, API integration, monitoring, drift detection, and scheduled retraining governance.
