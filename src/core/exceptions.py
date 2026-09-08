class CurrencyAnalyzerError(Exception):
    """Base exception for the application."""
    pass

class DataLoadError(CurrencyAnalyzerError):
    """Raised when data loading fails."""
    pass

class ModelError(CurrencyAnalyzerError):
    """Raised when ML model operations fail."""
    pass

class CacheError(CurrencyAnalyzerError):
    """Raised when cache operations fail."""
    pass

class ValidationError(CurrencyAnalyzerError):
    """Raised when validation fails."""
    pass
