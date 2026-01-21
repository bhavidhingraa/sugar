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

from .config import RalphConfig
from .profile import RalphWiggumProfile
from .signals import (
    CompletionSignal,
    CompletionSignalDetector,
    CompletionType,
    detect_completion,
    extract_signal_text,
    has_completion_signal,
)
from .validator import CompletionCriteriaValidator, ValidationResult

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
