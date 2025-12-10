"""Custom exception hierarchy for RAGiCamp.

All exceptions inherit from RAGiCampError, making it easy to:
- Catch all RAGiCamp-specific errors
- Distinguish from third-party errors
- Add context to error messages
"""

from typing import Any, Dict, Optional


class RAGiCampError(Exception):
    """Base exception for all RAGiCamp errors.
    
    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional context
        cause: Optional original exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with details."""
        msg = self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg = f"{msg} ({detail_str})"
        if self.cause:
            msg = f"{msg} [caused by: {type(self.cause).__name__}: {self.cause}]"
        return msg


# === Configuration Errors ===

class ConfigurationError(RAGiCampError):
    """Invalid or missing configuration.
    
    Examples:
        - Missing required config field
        - Invalid config value
        - Config file not found
    """
    pass


class ValidationError(ConfigurationError):
    """Data validation failed.
    
    Examples:
        - Invalid input format
        - Type mismatch
        - Constraint violation
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)[:100]  # Truncate long values
        if expected:
            details["expected"] = expected
        super().__init__(message, details=details, **kwargs)


# === Component Errors ===

class ComponentNotFoundError(RAGiCampError):
    """Required component is not available.
    
    Examples:
        - Model not installed
        - Metric library missing
        - Retriever index not found
    """
    
    def __init__(
        self,
        component_type: str,
        component_name: str,
        available: Optional[list] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["component_type"] = component_type
        details["component_name"] = component_name
        if available:
            details["available"] = available
        
        message = f"{component_type} '{component_name}' not found"
        if available:
            message += f". Available: {', '.join(available)}"
        
        super().__init__(message, details=details, **kwargs)


class ComponentInitError(RAGiCampError):
    """Component failed to initialize.
    
    Examples:
        - Model loading failed
        - Index creation failed
        - Connection failed
    """
    
    def __init__(
        self,
        component_type: str,
        component_name: str,
        reason: str,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["component_type"] = component_type
        details["component_name"] = component_name
        
        message = f"Failed to initialize {component_type} '{component_name}': {reason}"
        super().__init__(message, details=details, **kwargs)


# === Model Errors ===

class ModelError(RAGiCampError):
    """Error in model operations.
    
    Examples:
        - Generation failed
        - Token limit exceeded
        - API error
    """
    pass


class TokenLimitError(ModelError):
    """Token limit exceeded.
    
    Examples:
        - Input too long
        - Output exceeded max_tokens
    """
    
    def __init__(
        self,
        actual: int,
        limit: int,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["actual_tokens"] = actual
        details["token_limit"] = limit
        
        message = f"Token limit exceeded: {actual} > {limit}"
        super().__init__(message, details=details, **kwargs)


class GenerationError(ModelError):
    """Text generation failed.
    
    Examples:
        - Model returned empty output
        - Generation timed out
        - Invalid response format
    """
    pass


# === Retriever Errors ===

class RetrieverError(RAGiCampError):
    """Error in retrieval operations.
    
    Examples:
        - Index not found
        - Query failed
        - Embedding error
    """
    pass


class IndexNotFoundError(RetrieverError):
    """Retriever index not found.
    
    Examples:
        - Index file missing
        - Index not built yet
    """
    
    def __init__(self, index_path: str, **kwargs):
        details = kwargs.pop("details", {})
        details["index_path"] = index_path
        
        message = f"Index not found at: {index_path}"
        super().__init__(message, details=details, **kwargs)


# === Evaluation Errors ===

class EvaluationError(RAGiCampError):
    """Error during evaluation.
    
    Examples:
        - Metric computation failed
        - Agent crashed
        - Resource exhausted
    """
    pass


class MetricError(EvaluationError):
    """Error computing a metric.
    
    Examples:
        - Invalid input format
        - Metric library error
        - Computation failed
    """
    
    def __init__(
        self,
        metric_name: str,
        reason: str,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["metric"] = metric_name
        
        message = f"Metric '{metric_name}' failed: {reason}"
        super().__init__(message, details=details, **kwargs)


class CheckpointError(EvaluationError):
    """Error with checkpointing.
    
    Examples:
        - Checkpoint corrupted
        - Checkpoint version mismatch
        - Save failed
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        operation: str,  # "load" or "save"
        reason: str,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["checkpoint_path"] = checkpoint_path
        details["operation"] = operation
        
        message = f"Checkpoint {operation} failed for {checkpoint_path}: {reason}"
        super().__init__(message, details=details, **kwargs)


# === Dataset Errors ===

class DatasetError(RAGiCampError):
    """Error with dataset operations.
    
    Examples:
        - Dataset not found
        - Download failed
        - Parse error
    """
    pass


class DatasetNotFoundError(DatasetError):
    """Dataset not found.
    
    Examples:
        - HuggingFace dataset missing
        - Local file not found
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["dataset"] = dataset_name
        if split:
            details["split"] = split
        
        message = f"Dataset '{dataset_name}' not found"
        if split:
            message += f" (split: {split})"
        
        super().__init__(message, details=details, **kwargs)


# === State Management Errors ===

class StateError(RAGiCampError):
    """Error with experiment state management.
    
    Examples:
        - State file corrupted
        - Phase already completed
        - Invalid state transition
    """
    pass


# === Utility function ===

def wrap_exception(
    exception: Exception,
    wrapper_class: type,
    message: Optional[str] = None,
) -> RAGiCampError:
    """Wrap an external exception in a RAGiCamp exception.
    
    Args:
        exception: The original exception
        wrapper_class: The RAGiCamp exception class to use
        message: Optional custom message
        
    Returns:
        Wrapped exception with cause chain preserved
        
    Example:
        >>> try:
        ...     result = external_api_call()
        ... except APIError as e:
        ...     raise wrap_exception(e, ModelError, "API call failed")
    """
    msg = message or str(exception)
    return wrapper_class(msg, cause=exception)
