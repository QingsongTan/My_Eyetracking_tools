"""Custom exception hierarchy."""


class GazeToolkitError(Exception):
    """Base error for the package."""


class DataValidationError(GazeToolkitError):
    """Raised when a recording is malformed."""


class UnsupportedFormatError(GazeToolkitError):
    """Raised when the requested input format is not supported."""


class OptionalDependencyError(GazeToolkitError):
    """Raised when an optional dependency is required but unavailable."""

