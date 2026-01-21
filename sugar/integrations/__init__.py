"""
Sugar Integrations

External service integrations:
- GitHub: Issue and PR management
"""

from .github import GitHubClient, GitHubComment, GitHubIssue

__all__ = [
    "GitHubClient",
    "GitHubIssue",
    "GitHubComment",
]
