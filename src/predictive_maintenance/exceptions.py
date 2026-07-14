"""Custom exceptions for predictive maintenance workflows."""


class PredictiveMaintenanceError(Exception):
    """Base exception for predictive maintenance project errors."""


class DataLoadingError(PredictiveMaintenanceError):
    """Raised when a dataset cannot be loaded from disk."""


class DataValidationError(PredictiveMaintenanceError):
    """Raised when dataset structure validation fails."""


class EmptyDatasetError(DataValidationError):
    """Raised when a dataset has no rows or no columns."""


class MissingColumnsError(DataValidationError):
    """Raised when required dataset columns are missing."""

    def __init__(self, missing_columns: tuple[str, ...]) -> None:
        self.missing_columns = missing_columns
        columns = ", ".join(missing_columns)
        super().__init__(f"Missing required columns: {columns}")


class DuplicateColumnsError(DataValidationError):
    """Raised when duplicated dataset columns are found."""

    def __init__(self, duplicate_columns: tuple[str, ...]) -> None:
        self.duplicate_columns = duplicate_columns
        columns = ", ".join(duplicate_columns)
        super().__init__(f"Duplicated columns found: {columns}")
