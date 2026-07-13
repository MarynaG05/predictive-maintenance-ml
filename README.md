# Predictive Maintenance ML

End-to-end machine learning classification pipeline for predictive maintenance using Python, scikit-learn, and the public UCI AI4I 2020 Predictive Maintenance dataset.

## Project Objective

Build a reproducible, production-minded machine learning workflow that predicts machine failure from sensor and operating-condition data. The project is intended as a professional portfolio example for applied machine learning, clean project organization, and maintainable engineering practices.

## Business Problem

Unexpected equipment failures can cause downtime, missed service-level commitments, safety risks, and expensive emergency maintenance. Predictive maintenance helps teams identify machines at elevated risk of failure early enough to schedule inspections, replace components, and reduce operational disruption.

## Planned Machine Learning Workflow

1. Ingest and validate the public UCI AI4I 2020 dataset.
2. Perform exploratory data analysis and document data quality findings.
3. Engineer features for machine operating conditions and failure modes.
4. Train baseline and improved classification models.
5. Evaluate models using metrics appropriate for imbalanced failure prediction.
6. Package the best model and preprocessing pipeline for reproducible inference.
7. Document results, limitations, and next steps.

## Repository Structure

```text
predictive-maintenance-ml/
├── data/
│   ├── raw/                 # Local raw datasets, not tracked by Git
│   └── processed/           # Local processed datasets, not tracked by Git
├── docs/                    # Project documentation
├── notebooks/               # Exploratory and analysis notebooks
├── reports/
│   └── figures/             # Generated figures and report assets
├── src/
│   └── predictive_maintenance/
│       └── __init__.py      # Python package
├── tests/                   # Automated tests
├── .github/
│   └── workflows/
│       └── ci.yml           # Continuous integration workflow
├── pyproject.toml           # Project metadata and tool configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .gitignore
└── README.md
```

## Installation

This project targets Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Development Commands

```bash
ruff check .
ruff format .
pytest
pytest --cov=predictive_maintenance --cov-report=term-missing
pre-commit run --all-files
```

## Project Status

Initial project structure and development configuration are in place. Dataset download, exploratory data analysis, model training, evaluation, and model packaging are planned future steps.

## Dataset Attribution

Dataset attribution placeholder: UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset. Full citation and access details will be added when the data workflow is implemented.

## License

License placeholder: a project license will be selected before public release.
