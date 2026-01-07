"""
Ralph Wiggum Integration - Iterative AI loop support for Sugar

This module provides:
- CompletionCriteriaValidator: Validates tasks have clear exit conditions
- RalphWiggumProfile: Profile for iterative task execution
- RalphConfig: Configuration for Ralph Wiggum loops
- CompletionSignal: Structured completion signal representation
- CompletionType: Enum of completion signal types
- CompletionSignalDetector: Multi-pattern completion signal detector
"""

from .validator import CompletionCriteriaValidator, ValidationResult
from .profile import RalphWiggumProfile
from .config import RalphConfig
from .signals import (
    CompletionSignal,
    CompletionType,
    CompletionSignalDetector,
    detect_completion,
    has_completion_signal,
    extract_signal_text,
)

__all__ = [
    # Core validation
    "CompletionCriteriaValidator",
    "ValidationResult",
    # Profile and config
    "RalphWiggumProfile",
    "RalphConfig",
    # Completion signals
    "CompletionSignal",
    "CompletionType",
    "CompletionSignalDetector",
    # Convenience functions
    "detect_completion",
    "has_completion_signal",
    "extract_signal_text",
]
