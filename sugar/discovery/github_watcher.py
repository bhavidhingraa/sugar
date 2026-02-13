"""
GitHub Issue Watcher - Discover work from GitHub issues and PRs
Supports both GitHub CLI (gh) and PyGithub authentication
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..config import IssueResponderConfig
from ..storage import IssueResponseManager

# Optional PyGithub import
try:
    from github import Github, GithubException

    PYGITHUB_AVAILABLE = True
except ImportError:
    PYGITHUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GitHubWatcher:
    """Monitor GitHub repository for issues and pull requests"""

    def __init__(
        self,
        config: dict,
        issue_responder_config: Optional[IssueResponderConfig] = None,
    ):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.repo_name = config.get("repo", "")
        self.auth_method = config.get("auth_method", "auto")
        self.issue_responder_config = issue_responder_config

        # Initialize IssueResponseManager
        db_path = config.get("db_path", ".sugar/sugar.db")
        self.issue_response_manager = IssueResponseManager(db_path)

        if not self.enabled:
            return

        if not self.repo_name:
            logger.warning("GitHub repo not configured - GitHub watching disabled")
            self.enabled = False
            return

        # Check authentication methods
        self.gh_cli_available = False
        self.pygithub_available = False

        # Try GitHub CLI first if specified
        if self.auth_method in ["gh_cli", "auto"]:
            self.gh_cli_available = self._check_gh_cli()

        # Try PyGithub if CLI not available or method is token/auto
        if self.auth_method in ["token", "auto"] and not self.gh_cli_available:
            self.pygithub_available = self._init_pygithub()

        # Determine if we can proceed
        if self.auth_method == "gh_cli" and not self.gh_cli_available:
            logger.warning(
                "GitHub CLI specified but not available - GitHub watching disabled"
            )
            self.enabled = False
        elif self.auth_method == "token" and not self.pygithub_available:
            logger.warning(
                "Token auth specified but PyGithub not available - GitHub watching disabled"
            )
            self.enabled = False
        elif self.auth_method == "auto" and not (
            self.gh_cli_available or self.pygithub_available
        ):
            logger.warning(
                "Neither GitHub CLI nor PyGithub available - GitHub watching disabled"
            )
            self.enabled = False

        if self.enabled:
            method = "GitHub CLI" if self.gh_cli_available else "PyGithub"
            logger.info(
                f"✅ GitHub watcher initialized for {self.repo_name} using {method}"
            )

    def _check_gh_cli(self) -> bool:
        """Check if GitHub CLI is available and authenticated"""
        try:
            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            # Check if gh CLI is available
            result = subprocess.run(
                [gh_command, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                logger.debug("GitHub CLI not found")
                return False

            # Check if authenticated
            result = subprocess.run(
                [gh_command, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("GitHub CLI not authenticated")
                return False

            logger.info("🔑 GitHub CLI available and authenticated")
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"GitHub CLI check failed: {e}")
            return False

    def _init_pygithub(self) -> bool:
        """Initialize PyGithub if available"""
        if not PYGITHUB_AVAILABLE:
            logger.debug("PyGithub not available")
            return False

        token = self.config.get("token", "") or os.getenv("GITHUB_TOKEN", "")
        if not token:
            logger.debug("No GitHub token configured")
            return False

        try:
            self.github = Github(token)
            # Test the connection
            repo = self.github.get_repo(self.repo_name)
            logger.info("🔑 PyGithub initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyGithub: {e}")
            return False

    async def discover(self) -> List[Dict[str, Any]]:
        """Discover work items from GitHub issues and PRs"""
        if not self.enabled:
            return []

        work_items = []

        try:
            if self.gh_cli_available:
                # Use GitHub CLI
                issues_work = await self._discover_issues_gh_cli()
                work_items.extend(issues_work)
            elif self.pygithub_available:
                # Use PyGithub
                issues_work = await self._discover_issues_pygithub()
                work_items.extend(issues_work)

            # Discover PR review comments if enabled
            if self.config.get("review_comments", {}).get("enabled", True):
                review_comments_work = await self._discover_pr_review_comments()
                work_items.extend(review_comments_work)

        except Exception as e:
            logger.error(f"Error discovering GitHub work: {e}")

        logger.debug(f"🔍 GitHubWatcher discovered {len(work_items)} work items")
        return work_items

    async def _discover_issues_gh_cli(self) -> List[Dict[str, Any]]:
        """Discover work from GitHub issues using GitHub CLI"""
        work_items = []

        try:
            # Initialize issue response manager
            await self.issue_response_manager.initialize()

            # Load issue responder config if not provided in constructor
            responder_config = self.issue_responder_config
            if responder_config is None:
                try:
                    responder_config = IssueResponderConfig.load_from_file(
                        ".sugar/config.yaml"
                    )
                except (FileNotFoundError, KeyError):
                    # If config doesn't exist, use default (disabled)
                    responder_config = IssueResponderConfig()

            gh_command = self.config.get("gh_cli", {}).get("command", "gh")
            issue_labels = self.config.get("issue_labels", ["bug", "enhancement"])

            # Log the label filtering mode being used
            self._log_label_filtering_mode(issue_labels)

            # Get all open issues first (we'll filter by labels after)
            # Include author field for bot detection
            cmd = [
                gh_command,
                "issue",
                "list",
                "--repo",
                self.repo_name,
                "--state",
                "open",
                "--limit",
                "50",  # Get more issues to filter from
                "--json",
                "number,title,body,labels,assignees,comments,createdAt,updatedAt,url,author",
            ]

            # Note: We don't use --label flag here because it uses AND logic
            # Instead we'll filter by labels after getting the results

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"GitHub CLI issue command failed: {result.stderr}")
                return []

            issues = json.loads(result.stdout)

            # Filter issues by labels with flexible filtering modes
            filtered_issues = []
            for issue in issues:
                issue_label_names = [
                    label["name"].lower() for label in issue.get("labels", [])
                ]
                config_labels = [label.lower() for label in issue_labels]

                # Determine if this issue should be included based on label filtering mode
                should_include = self._should_include_issue_by_labels(
                    issue_label_names, config_labels, issue_labels
                )

                if should_include:
                    filtered_issues.append(issue)

            # Limit to 10 issues after filtering
            filtered_issues = filtered_issues[:10]

            logger.debug(
                f"Found {len(issues)} total issues, {len(filtered_issues)} match filtering criteria"
            )

            for issue in filtered_issues:
                # Check if issue responder is enabled and should create response work item
                if responder_config.enabled and self._should_respond_to_issue(
                    issue, responder_config
                ):
                    # Check if we've already responded to this issue
                    has_responded = await self.issue_response_manager.has_responded(
                        self.repo_name, issue["number"], "initial"
                    )

                    if not has_responded:
                        # Create issue response work item
                        response_work_item = {
                            "type": "issue_response",
                            "title": f"Respond to issue #{issue['number']}: {issue['title']}",
                            "source_file": f"github://issues/{issue['number']}",
                            "priority": 2,  # Medium priority
                            "context": {
                                "github_issue": {
                                    "number": issue["number"],
                                    "title": issue["title"],
                                    "body": issue.get("body", ""),
                                    "user": {
                                        "login": issue.get("author", {}).get(
                                            "login", "unknown"
                                        )
                                    },
                                    "labels": [
                                        label["name"]
                                        for label in issue.get("labels", [])
                                    ],
                                    "created_at": issue["createdAt"],
                                },
                                "repo": self.repo_name,
                                "response_type": "initial",
                            },
                        }
                        work_items.append(response_work_item)
                        logger.debug(
                            f"Created issue_response work item for #{issue['number']}"
                        )
                        continue

                # Create standard work item if not creating a response item
                work_item = self._create_work_item_from_issue_data(issue)
                if work_item:
                    work_items.append(work_item)

        except Exception as e:
            logger.error(f"Error getting issues via GitHub CLI: {e}")

        return work_items

    async def _discover_issues_pygithub(self) -> List[Dict[str, Any]]:
        """Discover work from GitHub issues using PyGithub"""
        work_items = []

        try:
            # Initialize issue response manager
            await self.issue_response_manager.initialize()

            # Load issue responder config if not provided in constructor
            responder_config = self.issue_responder_config
            if responder_config is None:
                try:
                    responder_config = IssueResponderConfig.load_from_file(
                        ".sugar/config.yaml"
                    )
                except (FileNotFoundError, KeyError):
                    # If config doesn't exist, use default (disabled)
                    responder_config = IssueResponderConfig()

            # Get recent issues
            since = datetime.now(timezone.utc) - timedelta(days=7)
            repo = self.github.get_repo(self.repo_name)
            issues = repo.get_issues(state="open", since=since, sort="created")

            issue_labels = self.config.get("issue_labels", ["bug", "enhancement"])

            # Log the label filtering mode being used
            self._log_label_filtering_mode(issue_labels)

            processed_count = 0
            for issue in issues:
                # Skip pull requests (they show up in issues)
                if issue.pull_request:
                    continue

                # Apply label filtering
                issue_label_names = [label.name.lower() for label in issue.labels]
                config_labels = [label.lower() for label in issue_labels]

                if not self._should_include_issue_by_labels(
                    issue_label_names, config_labels, issue_labels
                ):
                    continue

                # Convert PyGithub issue to dict format
                issue_data = {
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "labels": [{"name": label.name} for label in issue.labels],
                    "assignees": (
                        [{"login": issue.assignee.login}] if issue.assignee else []
                    ),
                    "comments": issue.comments,
                    "createdAt": issue.created_at.isoformat(),
                    "updatedAt": issue.updated_at.isoformat(),
                    "url": issue.html_url,
                    "author": {"login": issue.user.login},
                }

                # Check if issue responder is enabled and should create response work item
                if responder_config.enabled and self._should_respond_to_issue(
                    issue_data, responder_config
                ):
                    # Check if we've already responded to this issue
                    has_responded = await self.issue_response_manager.has_responded(
                        self.repo_name, issue.number, "initial"
                    )

                    if not has_responded:
                        # Create issue response work item
                        response_work_item = {
                            "type": "issue_response",
                            "title": f"Respond to issue #{issue.number}: {issue.title}",
                            "source_file": f"github://issues/{issue.number}",
                            "priority": 2,  # Medium priority
                            "context": {
                                "github_issue": {
                                    "number": issue.number,
                                    "title": issue.title,
                                    "body": issue.body or "",
                                    "user": {"login": issue.user.login},
                                    "labels": [label.name for label in issue.labels],
                                    "created_at": issue.created_at.isoformat(),
                                },
                                "repo": self.repo_name,
                                "response_type": "initial",
                            },
                        }
                        work_items.append(response_work_item)
                        logger.debug(
                            f"Created issue_response work item for #{issue.number}"
                        )
                        processed_count += 1

                        # Limit to 10 issues after filtering
                        if processed_count >= 10:
                            break
                        continue

                # Create standard work item if not creating a response item
                work_item = self._create_work_item_from_issue_data(issue_data)
                if work_item:
                    work_items.append(work_item)
                    processed_count += 1

                # Limit to 10 issues after filtering
                if processed_count >= 10:
                    break

        except Exception as e:
            logger.error(f"GitHub API error getting issues: {e}")

        return work_items

    def _create_work_item_from_issue_data(
        self, issue: dict
    ) -> Optional[Dict[str, Any]]:
        """Create work item from GitHub issue data (works with both CLI and PyGithub)"""

        # Determine work type from labels
        work_type = "feature"  # default
        priority = 3  # default

        labels = [label["name"].lower() for label in issue.get("labels", [])]

        if any(label in labels for label in ["bug", "error", "critical"]):
            work_type = "bug_fix"
            priority = 4
        elif any(label in labels for label in ["enhancement", "feature"]):
            work_type = "feature"
            priority = 3
        elif any(label in labels for label in ["documentation", "docs"]):
            work_type = "documentation"
            priority = 2
        elif any(label in labels for label in ["test", "testing"]):
            work_type = "test"
            priority = 3

        # Increase priority for urgent labels
        if any(label in labels for label in ["urgent", "high priority", "critical"]):
            priority = min(5, priority + 1)

        # Skip if assigned to someone else (optional)
        assignees = issue.get("assignees", [])
        if self.config.get("only_unassigned", False) and assignees:
            return None

        work_item = {
            "type": work_type,
            "title": f"Address GitHub issue: {issue['title']}",
            "description": self._format_issue_description(issue),
            "priority": priority,
            "source": "github_watcher",
            "source_file": f"github://issues/{issue['number']}",
            "context": {
                "github_issue": {
                    "number": issue["number"],
                    "url": issue["url"],
                    "labels": labels,
                    "assignees": [a.get("login") for a in issue.get("assignees", [])],
                    "comments": issue.get("comments", 0),
                    "created_at": issue["createdAt"],
                    "updated_at": issue["updatedAt"],
                },
                "discovered_at": datetime.now(timezone.utc).isoformat(),
                "source_type": "github_issue",
            },
        }

        return work_item

    def _format_issue_description(self, issue: dict) -> str:
        """Format GitHub issue into work description"""
        description_parts = [
            f"**GitHub Issue #{issue['number']}**",
            f"URL: {issue['url']}",
            f"Created: {issue['createdAt']}",
            f"Comments: {issue.get('comments', 0)}",
            "",
        ]

        if issue.get("labels"):
            label_names = [label["name"] for label in issue["labels"]]
            description_parts.append(f"Labels: {', '.join(label_names)}")
            description_parts.append("")

        assignees = issue.get("assignees", [])
        if assignees:
            assignee_names = [a.get("login", "unknown") for a in assignees]
            description_parts.append(f"Assigned to: {', '.join(assignee_names)}")
            description_parts.append("")

        description_parts.append("**Issue Description:**")
        description_parts.append(issue.get("body") or "No description provided.")

        return "\n".join(description_parts)

    async def health_check(self) -> dict:
        """Return health status of GitHub watcher"""
        if not self.enabled:
            return {
                "enabled": False,
                "reason": "GitHub integration disabled or not configured",
            }

        method = "GitHub CLI" if self.gh_cli_available else "PyGithub"

        try:
            if self.gh_cli_available:
                # Test GitHub CLI
                gh_command = self.config.get("gh_cli", {}).get("command", "gh")
                result = subprocess.run(
                    [gh_command, "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                auth_ok = result.returncode == 0

                return {
                    "enabled": True,
                    "method": method,
                    "repository": self.repo_name,
                    "authenticated": auth_ok,
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }
            elif self.pygithub_available:
                # Test PyGithub API access
                rate_limit = self.github.get_rate_limit()

                return {
                    "enabled": True,
                    "method": method,
                    "repository": self.repo_name,
                    "api_rate_limit": {
                        "remaining": rate_limit.core.remaining,
                        "limit": rate_limit.core.limit,
                        "reset": rate_limit.core.reset.isoformat(),
                    },
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            return {
                "enabled": True,
                "method": method,
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat(),
            }

    async def comment_on_issue(self, issue_number: int, comment_body: str) -> bool:
        """Add a comment to a GitHub issue"""
        if not self.enabled:
            return False

        try:
            if self.gh_cli_available:
                return await self._comment_via_gh_cli(issue_number, comment_body)
            elif self.pygithub_available:
                return await self._comment_via_pygithub(issue_number, comment_body)
            else:
                logger.warning(
                    "No GitHub authentication method available for commenting"
                )
                return False

        except Exception as e:
            logger.error(f"Error commenting on GitHub issue #{issue_number}: {e}")
            return False

    async def assign_issue(self, issue_number: int) -> bool:
        """Assign a GitHub issue to the authenticated user"""
        if not self.enabled:
            return False

        try:
            if self.gh_cli_available:
                return await self._assign_via_gh_cli(issue_number)
            elif self.pygithub_available:
                return await self._assign_via_pygithub(issue_number)
            else:
                logger.warning(
                    "No GitHub authentication method available for assignment"
                )
                return False

        except Exception as e:
            logger.error(f"Error assigning GitHub issue #{issue_number}: {e}")
            return False

    async def _comment_via_gh_cli(self, issue_number: int, comment_body: str) -> bool:
        """Add comment using GitHub CLI"""
        try:
            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            cmd = [
                gh_command,
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                self.repo_name,
                "--body",
                comment_body,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"✅ Added comment to GitHub issue #{issue_number}")
                return True
            else:
                logger.error(f"GitHub CLI comment failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error using GitHub CLI to comment: {e}")
            return False

    async def _comment_via_pygithub(self, issue_number: int, comment_body: str) -> bool:
        """Add comment using PyGithub"""
        try:
            repo = self.github.get_repo(self.repo_name)
            issue = repo.get_issue(issue_number)
            issue.create_comment(comment_body)
            logger.info(f"✅ Added comment to GitHub issue #{issue_number}")
            return True

        except Exception as e:
            logger.error(f"Error using PyGithub to comment: {e}")
            return False

    async def _assign_via_gh_cli(self, issue_number: int) -> bool:
        """Assign issue using GitHub CLI"""
        try:
            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            cmd = [
                gh_command,
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                self.repo_name,
                "--add-assignee",
                "@me",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(
                    f"✅ Assigned GitHub issue #{issue_number} to authenticated user"
                )
                return True
            else:
                logger.error(f"GitHub CLI assignment failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error using GitHub CLI to assign: {e}")
            return False

    async def _assign_via_pygithub(self, issue_number: int) -> bool:
        """Assign issue using PyGithub"""
        try:
            repo = self.github.get_repo(self.repo_name)
            issue = repo.get_issue(issue_number)

            # Get current user
            user = self.github.get_user()

            # Add current user to assignees (preserving existing ones)
            current_assignees = [assignee.login for assignee in issue.assignees]
            if user.login not in current_assignees:
                current_assignees.append(user.login)
                issue.edit(assignees=current_assignees)
                logger.info(f"✅ Assigned GitHub issue #{issue_number} to {user.login}")
            else:
                logger.debug(
                    f"GitHub issue #{issue_number} already assigned to {user.login}"
                )

            return True

        except Exception as e:
            logger.error(f"Error using PyGithub to assign: {e}")
            return False

    def _should_respond_to_issue(
        self, issue: dict, config: IssueResponderConfig
    ) -> bool:
        """Check if we should create a response work item for this issue"""
        # Check bot issues
        if config.skip_bot_issues:
            author = issue.get("author", {}).get("login", "")
            if author.endswith("[bot]") or "[bot]" in author:
                return False

        # Check skip labels
        issue_labels = [l.get("name", "") for l in issue.get("labels", [])]
        for skip_label in config.skip_labels:
            if skip_label in issue_labels:
                return False

        # Check respond_to_labels (if not empty, issue must have one of these)
        if config.respond_to_labels:
            has_matching_label = any(
                label in issue_labels for label in config.respond_to_labels
            )
            if not has_matching_label:
                return False

        return True

    def _should_include_issue_by_labels(
        self, issue_labels: list, config_labels: list, original_config: list
    ) -> bool:
        """Determine if an issue should be included based on label filtering configuration"""

        # Mode 1: Empty list [] - No filtering, include ALL issues
        if not original_config:
            return True

        # Mode 2: Wildcard ["*"] - Include issues that have ANY labels (exclude unlabeled)
        if len(original_config) == 1 and original_config[0] == "*":
            return len(issue_labels) > 0

        # Mode 3: Unlabeled ["unlabeled"] - Include only issues WITHOUT labels
        if len(original_config) == 1 and original_config[0].lower() == "unlabeled":
            return len(issue_labels) == 0

        # Mode 4: Specific labels - Include issues that have at least one matching label
        return any(label in issue_labels for label in config_labels)

    def _log_label_filtering_mode(self, issue_labels: list):
        """Log what label filtering mode is being used"""
        if not issue_labels:
            logger.debug("🏷️ Label filtering: ALL open issues (no label restrictions)")
        elif len(issue_labels) == 1 and issue_labels[0] == "*":
            logger.debug(
                "🏷️ Label filtering: Issues with ANY labels (excluding unlabeled)"
            )
        elif len(issue_labels) == 1 and issue_labels[0].lower() == "unlabeled":
            logger.debug("🏷️ Label filtering: Only UNLABELED issues")
        else:
            logger.debug(f"🏷️ Label filtering: Issues with labels: {issue_labels}")

    async def close_issue(
        self, issue_number: int, completion_comment: str = None
    ) -> bool:
        """Close a GitHub issue after successful completion"""
        if not self.enabled:
            return False

        try:
            if self.gh_cli_available:
                return await self._close_issue_via_gh_cli(
                    issue_number, completion_comment
                )
            elif self.pygithub_available:
                return await self._close_issue_via_pygithub(
                    issue_number, completion_comment
                )
            else:
                logger.warning(
                    "No GitHub authentication method available for closing issues"
                )
                return False

        except Exception as e:
            logger.error(f"Error closing GitHub issue #{issue_number}: {e}")
            return False

    async def create_pull_request(
        self, branch_name: str, title: str, body: str, base_branch: str = "main"
    ) -> Optional[str]:
        """Create a pull request from a branch"""
        if not self.enabled:
            return None

        try:
            if self.gh_cli_available:
                return await self._create_pr_via_gh_cli(
                    branch_name, title, body, base_branch
                )
            elif self.pygithub_available:
                return await self._create_pr_via_pygithub(
                    branch_name, title, body, base_branch
                )
            else:
                logger.warning(
                    "No GitHub authentication method available for creating PRs"
                )
                return None

        except Exception as e:
            logger.error(f"Error creating pull request: {e}")
            return None

    async def _close_issue_via_gh_cli(
        self, issue_number: int, completion_comment: str = None
    ) -> bool:
        """Close issue using GitHub CLI"""
        try:
            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            # Add final comment if provided
            if completion_comment:
                comment_success = await self._comment_via_gh_cli(
                    issue_number, completion_comment
                )
                if not comment_success:
                    logger.warning(
                        f"Could not add final comment to issue #{issue_number}"
                    )

            # Close the issue
            cmd = [
                gh_command,
                "issue",
                "close",
                str(issue_number),
                "--repo",
                self.repo_name,
                "--comment",
                "Completed by Sugar AI - closing issue.",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"🔒 Closed GitHub issue #{issue_number}")
                return True
            else:
                logger.error(f"GitHub CLI close failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error using GitHub CLI to close issue: {e}")
            return False

    async def _close_issue_via_pygithub(
        self, issue_number: int, completion_comment: str = None
    ) -> bool:
        """Close issue using PyGithub"""
        try:
            repo = self.github.get_repo(self.repo_name)
            issue = repo.get_issue(issue_number)

            # Add final comment if provided
            if completion_comment:
                try:
                    issue.create_comment(completion_comment)
                except Exception as e:
                    logger.warning(
                        f"Could not add final comment to issue #{issue_number}: {e}"
                    )

            # Close the issue
            issue.edit(state="closed")
            logger.info(f"🔒 Closed GitHub issue #{issue_number}")
            return True

        except Exception as e:
            logger.error(f"Error using PyGithub to close issue: {e}")
            return False

    async def _create_pr_via_gh_cli(
        self, branch_name: str, title: str, body: str, base_branch: str = "main"
    ) -> Optional[str]:
        """Create PR using GitHub CLI"""
        try:
            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            cmd = [
                gh_command,
                "pr",
                "create",
                "--repo",
                self.repo_name,
                "--title",
                title,
                "--body",
                body,
                "--base",
                base_branch,
                "--head",
                branch_name,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                pr_url = result.stdout.strip()
                logger.info(f"🔀 Created pull request: {pr_url}")
                return pr_url
            else:
                logger.error(f"GitHub CLI PR creation failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error using GitHub CLI to create PR: {e}")
            return None

    async def _create_pr_via_pygithub(
        self, branch_name: str, title: str, body: str, base_branch: str = "main"
    ) -> Optional[str]:
        """Create PR using PyGithub"""
        try:
            repo = self.github.get_repo(self.repo_name)

            pr = repo.create_pull(
                title=title, body=body, head=branch_name, base=base_branch
            )

            logger.info(f"🔀 Created pull request #{pr.number}: {pr.html_url}")
            return pr.html_url

        except Exception as e:
            logger.error(f"Error using PyGithub to create PR: {e}")
            return None

    async def _discover_pr_review_comments(self) -> List[Dict[str, Any]]:
        """Discover unresolved PR review comments and create work items"""
        work_items = []

        if not self.enabled:
            return work_items

        try:
            await self.issue_response_manager.initialize()

            gh_command = self.config.get("gh_cli", {}).get("command", "gh")

            # Get open PRs
            cmd = [
                gh_command,
                "pr",
                "list",
                "--repo",
                self.repo_name,
                "--state",
                "open",
                "--limit",
                "50",
                "--json",
                "number,title,headRefName,baseRefName,createdAt,updatedAt",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"Failed to list PRs: {result.stderr}")
                return work_items

            prs_data = json.loads(result.stdout)

            # Import GitHubClient to use its GraphQL methods
            from ..integrations.github import GitHubClient

            gh_client = GitHubClient(repo=self.repo_name)
            all_comments = []

            # Get review comments for each PR, storing branch info with each comment
            for pr_data in prs_data:
                pr_number = pr_data.get("number")
                pr_branch = pr_data.get("headRefName", "")
                pr_base_branch = pr_data.get("baseRefName", "main")
                comments = gh_client.get_pr_review_comments(pr_number)
                # Store branch info with each comment
                for comment in comments:
                    comment.branch = pr_branch
                    comment.base_branch = pr_base_branch
                all_comments.extend(comments)

            # Sort by creation time (oldest first)
            all_comments.sort(key=lambda c: c.created_at)

            # Create work items for each unresolved comment
            for comment in all_comments:
                # Skip comments from bots (including our own)
                if comment.user.type == "Bot" or comment.user.login.endswith("[bot]"):
                    continue

                # Use thread_id as the unique identifier for deduplication
                # Hash the thread_id to get a numeric value for the database
                thread_id_hash = int(hashlib.md5(comment.thread_id.encode()).hexdigest()[:8], 16)

                has_responded = await self.issue_response_manager.has_responded(
                    self.repo_name, thread_id_hash, "pr_review_comment"
                )

                if not has_responded:
                    work_item = {
                        "type": "pr_review_comment",
                        "title": f"Address PR #{comment.pr_number} review comment",
                        "description": self._format_review_comment_description(comment),
                        "priority": 4,  # High priority for review comments
                        "source": "github_watcher",
                        "source_file": f"github://pr/{comment.pr_number}/comment/{thread_id_hash}",
                        "context": {
                            "github_pr_comment": {
                                "thread_id": comment.thread_id,
                                "comment_id": str(comment.id),
                                "pr_number": comment.pr_number,
                                "body": comment.body,
                                "path": comment.path,
                                "line": comment.line,
                                "created_at": comment.created_at,
                                "user": {"login": comment.user.login},
                            },
                            "repo": self.repo_name,
                            "pr_number": comment.pr_number,
                            "branch": getattr(comment, "branch", ""),
                            "base_branch": getattr(comment, "base_branch", "main"),
                            "thread_id": thread_id_hash,
                            "comment_body": comment.body,
                            "file_path": comment.path,
                            "line": comment.line,
                        },
                    }
                    work_items.append(work_item)
                    logger.info(
                        f"Created pr_review_comment work item for PR #{comment.pr_number} "
                        f"(thread: {thread_id_hash})"
                    )

            logger.debug(f"Found {len(all_comments)} unresolved review comments, created {len(work_items)} work items")

        except Exception as e:
            logger.error(f"Error discovering PR review comments: {e}")

        return work_items

    def _format_review_comment_description(self, comment) -> str:
        """Format review comment into work description"""
        description_parts = [
            f"**Review Comment on PR #{comment.pr_number}**",
            f"File: {comment.path}",
            f"Line: {comment.line}",
            "",
            "**Comment:**",
            comment.body,
            "",
            "**Task:** Address this review comment and push changes to the PR branch.",
        ]
        return "\n".join(description_parts)

    async def reply_to_review_thread(
        self,
        pr_number: int,
        thread_id: str,
        body: str,
    ) -> bool:
        """Reply to a review comment thread"""
        if not self.enabled:
            return False

        try:
            from ..integrations.github import GitHubClient

            gh_client = GitHubClient(repo=self.repo_name)
            return gh_client.reply_to_review_thread(pr_number, thread_id, body)

        except Exception as e:
            logger.error(f"Error replying to review thread: {e}")
            return False

    async def resolve_conversation_graphql(
        self,
        thread_id: str,
    ) -> bool:
        """Resolve a review conversation using GraphQL"""
        if not self.enabled:
            return False

        try:
            from ..integrations.github import GitHubClient

            gh_client = GitHubClient(repo=self.repo_name)
            return gh_client.resolve_conversation_graphql(thread_id)

        except Exception as e:
            logger.error(f"Error resolving conversation: {e}")
            return False

    async def add_pr_comment(
        self,
        pr_number: int,
        body: str,
    ) -> bool:
        """Add a comment to a PR"""
        if not self.enabled:
            return False

        try:
            from ..integrations.github import GitHubClient

            gh_client = GitHubClient(repo=self.repo_name)
            return gh_client.add_pr_comment(pr_number, body)

        except Exception as e:
            logger.error(f"Error adding PR comment: {e}")
            return False
