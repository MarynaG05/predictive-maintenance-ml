# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 1.0.0 - 2026-07-15

### Added

- Central project configuration for paths, dataset schema, and reproducible splits.
- Structural validation and local CSV loading for the UCI AI4I 2020 dataset.
- Dataset profiling and executed exploratory data analysis notebook.
- Leakage-safe preprocessing and baseline binary-classification workflow.
- Candidate model comparison with deterministic validation metrics.
- Validation threshold analysis and business operating-profile recommendations.
- Validation error analysis and permutation-based explainability.
- Frozen final model evaluation on the held-out test split.
- Final model artifact persistence with metadata validation and checksum checks.
- Version 1 batch prediction interface using persisted artifacts.
- Portfolio documentation, results summary, architecture notes, and case study.
- Automated pytest suite, Ruff linting/formatting, pre-commit hooks, and GitHub Actions CI.

### Notes

- Version 1 is an offline portfolio release, not a production deployment.
- The AI4I dataset is synthetic; external validation is required before operational use.
- Final model and threshold decisions are frozen before final held-out test evaluation.
