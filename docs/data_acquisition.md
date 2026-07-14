# Data Acquisition

## Dataset

This project uses the AI4I 2020 Predictive Maintenance dataset from the UCI Machine Learning Repository.

Official source placeholder: verify the current UCI dataset page and citation before final publication.

## Expected Local File

The expected local filename is:

```text
ai4i2020.csv
```

The expected local destination is:

```text
data/raw/ai4i2020.csv
```

Raw data files are not committed to Git. The `data/raw/` directory is intended for local data only.

## Manual Download Instructions

1. Open the official UCI Machine Learning Repository page for the AI4I 2020 Predictive Maintenance dataset.
2. Download the CSV data file.
3. Rename the file to `ai4i2020.csv` if needed.
4. Place the file at `data/raw/ai4i2020.csv`.
5. Do not commit the raw data file.

## Verification Instructions

After placing the file locally, confirm that:

- The file exists at `data/raw/ai4i2020.csv`.
- The file is a regular CSV file.
- The header contains the expected columns listed below.
- The file is not staged for Git commit.

Useful local checks:

```bash
test -f data/raw/ai4i2020.csv
git status --short data/raw/ai4i2020.csv
```

## Expected Columns

The local CSV is expected to contain:

- `UDI`
- `Product ID`
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The loader rejects duplicate source headers before pandas loads the CSV. This prevents duplicate column names from being hidden by automatic header renaming during CSV parsing.

The loader returns these required columns in configured order, including identifier columns, predictive features, the target column, and failure-mode columns. Later stages need these fields for traceability, supervised learning, and leakage-safe feature selection. Extra source columns may be present in the raw CSV, but they are omitted from the returned DataFrame by the initial ingestion layer.

Completely empty files and header-only files are rejected because they do not contain usable data rows.

## Dataset Limitation

The AI4I 2020 dataset is synthetic. It is suitable for demonstrating a reproducible predictive maintenance workflow, but model results should not be presented as production-ready performance for real industrial equipment.

## Licensing And Attribution

Licensing and attribution details must be verified against the official UCI dataset page before publication or reuse outside this portfolio project. Do not make license claims until the dataset license has been confirmed.
