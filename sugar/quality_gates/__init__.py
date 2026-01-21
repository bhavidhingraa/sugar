"""
Quality Gates - Verification and Truth Enforcement

This module provides mandatory verification layers for Sugar tasks:

Phase 1 (Implemented):
- Test execution verification
- Success criteria validation
- Truth enforcement (proof required for claims)
- Evidence collection and storage

Phase 2 (Implemented):
- Functional verification (HTTP, browser, database)
- Pre-flight checks
- Enhanced evidence collection

Phase 3 (Implemented):
- Verification failure handling and retry logic
- Git diff validation
- Task schema validation

Phase 4 (Implemented):
- Acceptance criteria templates
- Default criteria per task type

Phase 5 (Implemented - AUTO-005):
- Self-verification before task completion
- Verification gate integration
- Verification status tracking
"""

from .coordinator import QualityGateResult, QualityGatesCoordinator

# Phase 4 exports - Acceptance Criteria
from .criteria_templates import CriteriaTemplates
from .diff_validator import DiffValidationResult, DiffValidator
from .evidence import Evidence, EvidenceCollector

# Phase 3 exports
from .failure_handler import FailureReport, VerificationFailureHandler

# Phase 2 exports
from .functional_verifier import FunctionalVerificationResult, FunctionalVerifier
from .preflight_checks import PreFlightChecker, PreFlightCheckResult
from .success_criteria import SuccessCriteriaVerifier, SuccessCriterion

# Phase 1 exports
from .test_validator import TestExecutionResult, TestExecutionValidator
from .truth_enforcer import TruthEnforcer

# Phase 5 exports - Self-verification (AUTO-005)
from .verification_gate import (
    VerificationGate,
    VerificationResult,
    VerificationResults,
    VerificationStatus,
)

__all__ = [
    # Phase 1
    "TestExecutionValidator",
    "TestExecutionResult",
    "SuccessCriteriaVerifier",
    "SuccessCriterion",
    "TruthEnforcer",
    "EvidenceCollector",
    "Evidence",
    "QualityGatesCoordinator",
    "QualityGateResult",
    # Phase 2
    "FunctionalVerifier",
    "FunctionalVerificationResult",
    "PreFlightChecker",
    "PreFlightCheckResult",
    # Phase 3
    "VerificationFailureHandler",
    "FailureReport",
    "DiffValidator",
    "DiffValidationResult",
    # Phase 4
    "CriteriaTemplates",
    # Phase 5 - Self-verification (AUTO-005)
    "VerificationGate",
    "VerificationResult",
    "VerificationResults",
    "VerificationStatus",
]
