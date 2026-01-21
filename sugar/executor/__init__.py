"""
Sugar Executor Module

Provides task execution implementations:
- BaseExecutor: Abstract interface for all executors
- ClaudeWrapper: Legacy subprocess-based execution (v1/v2)
- AgentSDKExecutor: Native SDK-based execution (v3+)
"""

from .agent_sdk_executor import AgentSDKExecutor
from .base import BaseExecutor, ExecutionResult
from .claude_wrapper import ClaudeWrapper
from .hooks import HookExecutor

__all__ = [
    "BaseExecutor",
    "ExecutionResult",
    "ClaudeWrapper",
    "AgentSDKExecutor",
    "HookExecutor",
]
