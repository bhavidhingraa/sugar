"""
Task Orchestration System for Sugar

Provides intelligent decomposition and execution of complex features through:
- Staged workflows (research, planning, implementation, review)
- Specialist agent routing
- Model routing by task complexity (AUTO-001)
- Parallel sub-task execution
- Context accumulation across stages
"""

from .agent_router import AgentRouter
from .model_router import ModelRouter, ModelSelection, ModelTier, create_model_router
from .task_orchestrator import (
    OrchestrationResult,
    OrchestrationStage,
    StageResult,
    TaskOrchestrator,
)

__all__ = [
    "TaskOrchestrator",
    "OrchestrationStage",
    "StageResult",
    "OrchestrationResult",
    "AgentRouter",
    "ModelRouter",
    "ModelTier",
    "ModelSelection",
    "create_model_router",
]
