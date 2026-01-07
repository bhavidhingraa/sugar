"""
Task Orchestration System for Sugar

Provides intelligent decomposition and execution of complex features through:
- Staged workflows (research, planning, implementation, review)
- Specialist agent routing
- Model routing by task complexity (AUTO-001)
- Parallel sub-task execution
- Context accumulation across stages
"""

from .task_orchestrator import (
    TaskOrchestrator,
    OrchestrationStage,
    StageResult,
    OrchestrationResult,
)
from .agent_router import AgentRouter
from .model_router import ModelRouter, ModelTier, ModelSelection, create_model_router

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
