"""
GitHub Integration - Issue and PR management

Provides a clean interface for GitHub API operations:
- Reading issues and comments
- Posting comments
- Managing labels
- Searching similar issues
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GitHubUser:
    """GitHub user information"""

    login: str
    id: int = 0
    type: str = "User"  # User, Bot, Organization

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubUser":
        return cls(
            login=data.get("login", "unknown"),
            id=data.get("id", 0),
            type=data.get("type", "User"),
        )


@dataclass
class GitHubLabel:
    """GitHub label"""

    name: str
    color: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubLabel":
        return cls(
            name=data.get("name", ""),
            color=data.get("color", ""),
            description=data.get("description", ""),
        )


@dataclass
class GitHubComment:
    """GitHub issue/PR comment"""

    id: int
    body: str
    user: GitHubUser
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubComment":
        return cls(
            id=data.get("id", 0),
            body=data.get("body", ""),
            user=GitHubUser.from_dict(data.get("user", {})),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class GitHubReviewComment:
    """GitHub PR review comment"""

    id: str  # GraphQL ID is a string (e.g., "PRRC_kwDOPnDFGc6lz2iF")
    thread_id: str
    pr_number: int
    body: str
    path: str
    line: int
    commit_id: str
    user: GitHubUser
    created_at: str
    state: str  # "resolved", "outdated", "active"
    diff_hunk: str = ""
    start_line: Optional[int] = None
    original_commit_id: Optional[str] = None
    branch: Optional[str] = None  # PR head branch name
    base_branch: Optional[str] = None  # PR base branch name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubReviewComment":
        return cls(
            id=str(data.get("id", "")),
            thread_id=data.get("thread_id", ""),
            pr_number=data.get("pr_number", 0),
            body=data.get("body", ""),
            path=data.get("path", ""),
            line=data.get("line", 0),
            commit_id=data.get("commit_id", ""),
            user=GitHubUser.from_dict(data.get("user", {})),
            created_at=data.get("created_at", ""),
            state=data.get("state", "active"),
            diff_hunk=data.get("diff_hunk", ""),
            start_line=data.get("start_line"),
            original_commit_id=data.get("original_commit_id"),
            branch=data.get("branch"),
            base_branch=data.get("base_branch"),
        )


@dataclass
class GitHubPullRequest:
    """GitHub Pull Request"""

    number: int
    title: str
    body: str
    state: str
    user: GitHubUser
    head_ref: str
    base_ref: str
    html_url: str
    created_at: str
    updated_at: str
    labels: List[GitHubLabel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubPullRequest":
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body", ""),
            state=data.get("state", "open"),
            user=GitHubUser.from_dict(data.get("user", {})),
            head_ref=data.get("head_ref", data.get("headRefName", "")),
            base_ref=data.get("base_ref", data.get("baseRefName", "")),
            html_url=data.get("html_url", data.get("url", "")),
            created_at=data.get("created_at", data.get("createdAt", "")),
            updated_at=data.get("updated_at", data.get("updatedAt", "")),
            labels=[GitHubLabel.from_dict(l) for l in data.get("labels", [])],
        )


@dataclass
class GitHubIssue:
    """GitHub issue"""

    number: int
    title: str
    body: str
    state: str
    user: GitHubUser
    labels: List[GitHubLabel]
    created_at: str
    updated_at: str
    comments_count: int = 0
    comments: List[GitHubComment] = field(default_factory=list)
    is_pull_request: bool = False
    html_url: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubIssue":
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body", "") or "",
            state=data.get("state", "open"),
            user=GitHubUser.from_dict(data.get("user", {})),
            labels=[GitHubLabel.from_dict(l) for l in data.get("labels", [])],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            comments_count=data.get("comments", 0),
            is_pull_request="pull_request" in data,
            html_url=data.get("html_url", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing"""
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "user": {"login": self.user.login, "type": self.user.type},
            "labels": [l.name for l in self.labels],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "comments_count": self.comments_count,
            "is_pull_request": self.is_pull_request,
            "html_url": self.html_url,
        }


class GitHubClient:
    """
    GitHub API client using the gh CLI.

    Uses the gh CLI for authentication and API access, which:
    - Handles authentication automatically
    - Works in GitHub Actions with GITHUB_TOKEN
    - Supports both user and app authentication
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize the GitHub client.

        Args:
            repo: Repository in owner/repo format (optional, uses current repo)
            token: GitHub token (optional, uses gh auth or GITHUB_TOKEN)
        """
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")

    def _run_gh(
        self,
        args: List[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a gh CLI command"""
        cmd = ["gh"] + args

        if self.repo:
            cmd.extend(["-R", self.repo])

        env = os.environ.copy()
        if self.token:
            env["GH_TOKEN"] = self.token

        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        if check and result.returncode != 0:
            logger.error(f"gh command failed: {result.stderr}")
            raise RuntimeError(f"gh command failed: {result.stderr}")

        return result

    def _gh_api(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a GitHub API request via gh (without -R flag)"""
        cmd = ["gh", "api", endpoint, "-X", method]

        if data:
            cmd.extend(["-f", json.dumps(data)])

        env = os.environ.copy()
        if self.token:
            env["GH_TOKEN"] = self.token

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        return None

    def _gh_graphql(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a GraphQL API request via gh CLI"""
        # Build command: -f for query (string), -F for typed variables
        cmd = ["gh", "api", "graphql", "-f", f"query={query}"]

        # Add variables with -F flag for typed literals (Integer, Boolean, etc.)
        if variables:
            for key, value in variables.items():
                if value is not None:  # Skip None values
                    cmd.extend(["-F", f"{key}={value}"])

        env = os.environ.copy()
        if self.token:
            env["GH_TOKEN"] = self.token

        logger.debug(f"Running GraphQL with {len(variables) if variables else 0} variables")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"GraphQL command failed: {result.stderr}")
            return None

        if result.stdout:
            try:
                parsed = json.loads(result.stdout)
                # Log any errors in the GraphQL response
                if "errors" in parsed:
                    logger.debug(f"GraphQL response contains errors: {parsed['errors']}")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GraphQL response: {e}")
                logger.debug(f"Raw response: {result.stdout[:500]}")
                return None
        return None

    async def _gh_graphql_async(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a GraphQL API request via gh CLI using asyncio"""
        # Build command: -f for query (string), -F for typed variables
        cmd = ["gh", "api", "graphql", "-f", f"query={query}"]

        # Add variables with -F flag for typed literals (Integer, Boolean, etc.)
        if variables:
            for key, value in variables.items():
                if value is not None:  # Skip None values
                    cmd.extend(["-F", f"{key}={value}"])

        env = os.environ.copy()
        if self.token:
            env["GH_TOKEN"] = self.token

        logger.debug(f"Running async GraphQL with {len(variables) if variables else 0} variables")

        process = await asyncio.create_subprocess_exec(
            *[asyncio.subprocess.PIPE] * 3,  # stdin, stdout, stderr
            cmd,
            env=env,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr.decode()
            logger.error(f"Async GraphQL command failed: {stderr_text}")
            return None

        if stdout:
            try:
                parsed = json.loads(stdout.decode())
                # Log any errors in the GraphQL response
                if "errors" in parsed:
                    logger.debug(f"GraphQL response contains errors: {parsed['errors']}")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GraphQL response: {e}")
                logger.debug(f"Raw response: {stdout.decode()[:500]}")
                return None
        return None

    def get_issue(self, issue_number: int) -> GitHubIssue:
        """Get a single issue by number"""
        result = self._run_gh(
            [
                "issue",
                "view",
                str(issue_number),
                "--json",
                "number,title,body,state,author,labels,createdAt,updatedAt,comments,url",
            ]
        )

        data = json.loads(result.stdout)

        # Map gh output format to our format
        issue_data = {
            "number": data.get("number"),
            "title": data.get("title"),
            "body": data.get("body"),
            "state": data.get("state"),
            "user": {"login": data.get("author", {}).get("login", "unknown")},
            "labels": data.get("labels", []),
            "created_at": data.get("createdAt"),
            "updated_at": data.get("updatedAt"),
            "comments": len(data.get("comments", [])),
            "html_url": data.get("url"),
        }

        issue = GitHubIssue.from_dict(issue_data)

        # Add comments
        for comment_data in data.get("comments", []):
            comment = GitHubComment(
                id=comment_data.get("id", 0),
                body=comment_data.get("body", ""),
                user=GitHubUser(
                    login=comment_data.get("author", {}).get("login", "unknown")
                ),
                created_at=comment_data.get("createdAt", ""),
                updated_at=comment_data.get("updatedAt", ""),
            )
            issue.comments.append(comment)

        return issue

    def list_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        limit: int = 30,
    ) -> List[GitHubIssue]:
        """List issues with optional filters"""
        args = [
            "issue",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,author,labels,createdAt,updatedAt,comments,url",
        ]

        if labels:
            args.extend(["--label", ",".join(labels)])

        result = self._run_gh(args)
        issues_data = json.loads(result.stdout)

        issues = []
        for data in issues_data:
            issue_data = {
                "number": data.get("number"),
                "title": data.get("title"),
                "body": data.get("body"),
                "state": data.get("state"),
                "user": {"login": data.get("author", {}).get("login", "unknown")},
                "labels": data.get("labels", []),
                "created_at": data.get("createdAt"),
                "updated_at": data.get("updatedAt"),
                "comments": len(data.get("comments", [])),
                "html_url": data.get("url"),
            }
            issues.append(GitHubIssue.from_dict(issue_data))

        return issues

    def post_comment(self, issue_number: int, body: str) -> GitHubComment:
        """Post a comment on an issue"""
        result = self._run_gh(
            [
                "issue",
                "comment",
                str(issue_number),
                "--body",
                body,
            ]
        )

        logger.info(f"Posted comment on issue #{issue_number}")

        # Return a minimal comment object (gh doesn't return the created comment)
        return GitHubComment(
            id=0,
            body=body,
            user=GitHubUser(login="sugar[bot]"),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def add_labels(self, issue_number: int, labels: List[str]) -> None:
        """Add labels to an issue"""
        if not labels:
            return

        self._run_gh(
            [
                "issue",
                "edit",
                str(issue_number),
                "--add-label",
                ",".join(labels),
            ]
        )

        logger.info(f"Added labels {labels} to issue #{issue_number}")

    def remove_labels(self, issue_number: int, labels: List[str]) -> None:
        """Remove labels from an issue"""
        if not labels:
            return

        self._run_gh(
            [
                "issue",
                "edit",
                str(issue_number),
                "--remove-label",
                ",".join(labels),
            ]
        )

        logger.info(f"Removed labels {labels} from issue #{issue_number}")

    def search_issues(
        self,
        query: str,
        limit: int = 10,
    ) -> List[GitHubIssue]:
        """Search for issues matching a query"""
        # Build search query
        search_query = query
        if self.repo:
            search_query = f"repo:{self.repo} {query}"

        args = [
            "search",
            "issues",
            search_query,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,author,labels,createdAt,updatedAt,url",
        ]

        result = self._run_gh(args)
        issues_data = json.loads(result.stdout)

        issues = []
        for data in issues_data:
            issue_data = {
                "number": data.get("number"),
                "title": data.get("title"),
                "body": data.get("body"),
                "state": data.get("state"),
                "user": {"login": data.get("author", {}).get("login", "unknown")},
                "labels": data.get("labels", []),
                "created_at": data.get("createdAt"),
                "updated_at": data.get("updatedAt"),
                "html_url": data.get("url"),
            }
            issues.append(GitHubIssue.from_dict(issue_data))

        return issues

    def find_similar_issues(
        self,
        issue: GitHubIssue,
        limit: int = 5,
    ) -> List[GitHubIssue]:
        """Find issues similar to the given issue"""
        import re

        # Build search query from issue title keywords
        # Remove special characters that break GitHub search
        clean_title = re.sub(r"[\[\]\(\)\{\}:\"'`]", " ", issue.title)
        keywords = [w for w in clean_title.split() if len(w) > 2][:5]

        if not keywords:
            return []

        query = " ".join(keywords)

        similar = self.search_issues(f"{query} is:issue", limit=limit + 1)

        # Filter out the current issue
        return [i for i in similar if i.number != issue.number][:limit]

    def has_maintainer_response(self, issue: GitHubIssue) -> bool:
        """Check if a maintainer has already responded"""
        # This is a heuristic - would need CODEOWNERS or team info for accuracy
        issue_author = issue.user.login

        for comment in issue.comments:
            if comment.user.login != issue_author:
                # Someone other than the author commented
                # Could be a maintainer
                return True

        return False

    def is_bot_author(self, issue: GitHubIssue) -> bool:
        """Check if the issue was created by a bot"""
        return issue.user.type == "Bot" or issue.user.login.endswith("[bot]")

    def list_open_prs(
        self,
        state: str = "open",
        limit: int = 50,
    ) -> List[GitHubPullRequest]:
        """List open pull requests"""
        args = [
            "pr",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,body,state,headRefName,baseRefName,createdAt,updatedAt,url,author,labels",
        ]

        result = self._run_gh(args)
        prs_data = json.loads(result.stdout)

        prs = []
        for data in prs_data:
            pr_data = {
                "number": data.get("number"),
                "title": data.get("title"),
                "body": data.get("body"),
                "state": data.get("state"),
                "user": {"login": data.get("author", {}).get("login", "unknown")},
                "headRefName": data.get("headRefName"),
                "baseRefName": data.get("baseRefName"),
                "url": data.get("url"),
                "createdAt": data.get("createdAt"),
                "updatedAt": data.get("updatedAt"),
                "labels": data.get("labels", []),
            }
            prs.append(GitHubPullRequest.from_dict(pr_data))

        return prs

    def get_pr_review_comments(
        self,
        pr_number: int,
    ) -> List[GitHubReviewComment]:
        """Get all unresolved review comments for a PR using GraphQL"""
        if not self.repo:
            logger.warning("No repo configured for GraphQL queries")
            return []

        # Validate repo format
        if not self.repo or "/" not in self.repo:
            logger.error(f"Invalid repo format: '{self.repo}'. Expected 'owner/repo'.")
            return []

        owner, repo_name = self.repo.split("/", 1)
        logger.debug(f"Querying PR #{pr_number} for repo: {self.repo} (owner={owner}, repo={repo_name})")

        query = """
        query($owner: String!, $repo: String!, $pr_number: Int!, $after: String) {
            repository(owner: $owner, name: $repo) {
                pullRequest(number: $pr_number) {
                    reviewThreads(first: 50, after: $after) {
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                        nodes {
                            id
                            isResolved
                            comments(first: 100) {
                                nodes {
                                    id
                                    body
                                    author { login }
                                    createdAt
                                    originalCommit { oid }
                                    path
                                    line
                                    startLine
                                    diffHunk
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        all_comments = []
        has_next_page = True
        after_cursor = None

        while has_next_page:
            # Only include 'after' if we have a cursor
            variables = {
                "owner": owner,
                "repo": repo_name,
                "pr_number": pr_number,
            }
            if after_cursor:
                variables["after"] = after_cursor

            result = self._gh_graphql(query, variables)

            if not result:
                logger.error(f"GraphQL query returned no result for PR #{pr_number}")
                break

            # Check for errors in response
            if "errors" in result:
                errors = result.get("errors", [])
                error_msgs = [e.get("message", str(e)) for e in errors]
                logger.error(f"GraphQL errors for PR #{pr_number}: {error_msgs}")
                break

            if "data" not in result:
                logger.error(f"GraphQL response missing 'data' for PR #{pr_number}: {result}")
                break

            # Check if data is None or repository is missing
            data = result.get("data")
            if not data:
                logger.error(f"GraphQL response has null 'data' for PR #{pr_number}")
                break

            if "repository" not in data:
                logger.error(f"GraphQL response missing 'repository' for PR #{pr_number}. Data keys: {list(data.keys())}")
                break

            if data.get("repository") is None:
                logger.error(f"GraphQL response has null 'repository' for PR #{pr_number}")
                break

            try:
                pr_data = data["repository"].get("pullRequest")
                if not pr_data:
                    logger.error(f"GraphQL response has null 'pullRequest' for PR #{pr_number}")
                    break

                threads_data = pr_data.get("reviewThreads")
                if not threads_data:
                    logger.error(f"GraphQL response missing 'reviewThreads' for PR #{pr_number}")
                    break

                threads = threads_data.get("nodes", [])
                page_info = threads_data.get("pageInfo", {})

                for thread in threads:
                    if not thread.get("isResolved", False):  # Only unresolved threads
                        comments = thread.get("comments", {}).get("nodes", [])
                        if comments:
                            # Get the original (first) comment in the thread
                            original_comment = comments[0]
                            author_login = original_comment["author"]["login"]
                            # Determine if bot by checking login suffix
                            author_type = "Bot" if author_login.endswith("[bot]") else "User"

                            comment = GitHubReviewComment(
                                id=original_comment["id"],
                                thread_id=thread["id"],
                                pr_number=pr_number,
                                body=original_comment.get("body", ""),
                                path=original_comment.get("path", ""),
                                line=original_comment.get("line", 0),
                                commit_id=original_comment.get("originalCommit", {}).get("oid", ""),
                                user=GitHubUser(
                                    login=author_login,
                                    type=author_type,
                                ),
                                created_at=original_comment["createdAt"],
                                state="active",
                                diff_hunk=original_comment.get("diffHunk", ""),
                                start_line=original_comment.get("startLine"),
                            )
                            all_comments.append(comment)

                has_next_page = page_info.get("hasNextPage", False)
                if has_next_page:
                    after_cursor = page_info.get("endCursor")

            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing GraphQL response for PR #{pr_number}: {e}")
                break

        return all_comments

    async def get_pr_review_comments_async(
        self,
        pr_number: int,
    ) -> List[GitHubReviewComment]:
        """Get all unresolved review comments for a PR using GraphQL (async)"""
        if not self.repo:
            logger.warning("No repo configured for GraphQL queries")
            return []

        # Validate repo format
        if not self.repo or "/" not in self.repo:
            logger.error(f"Invalid repo format: '{self.repo}'. Expected 'owner/repo'.")
            return []

        owner, repo_name = self.repo.split("/", 1)
        logger.debug(f"Querying PR #{pr_number} for repo: {self.repo} (owner={owner}, repo={repo_name})")

        query = """
        query($owner: String!, $repo: String!, $pr_number: Int!, $after: String) {
            repository(owner: $owner, name: $repo) {
                pullRequest(number: $pr_number) {
                    reviewThreads(first: 50, after: $after) {
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                        nodes {
                            id
                            isResolved
                            comments(first: 100) {
                                nodes {
                                    id
                                    body
                                    author { login }
                                    createdAt
                                    originalCommit { oid }
                                    path
                                    line
                                    startLine
                                    diffHunk
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        all_comments = []
        has_next_page = True
        after_cursor = None

        while has_next_page:
            # Only include 'after' if we have a cursor
            variables = {
                "owner": owner,
                "repo": repo_name,
                "pr_number": pr_number,
            }
            if after_cursor:
                variables["after"] = after_cursor

            result = await self._gh_graphql_async(query, variables)

            if not result:
                logger.error(f"GraphQL query returned no result for PR #{pr_number}")
                break

            # Check for errors in response
            if "errors" in result:
                errors = result.get("errors", [])
                error_msgs = [e.get("message", str(e)) for e in errors]
                logger.error(f"GraphQL errors for PR #{pr_number}: {error_msgs}")
                break

            if "data" not in result:
                logger.error(f"GraphQL response missing 'data' for PR #{pr_number}: {result}")
                break

            # Check if data is None or repository is missing
            data = result.get("data")
            if not data:
                logger.error(f"GraphQL response has null 'data' for PR #{pr_number}")
                break

            if "repository" not in data:
                logger.error(f"GraphQL response missing 'repository' for PR #{pr_number}. Data keys: {list(data.keys())}")
                break

            if data.get("repository") is None:
                logger.error(f"GraphQL response has null 'repository' for PR #{pr_number}")
                break

            try:
                pr_data = data["repository"].get("pullRequest")
                if not pr_data:
                    logger.error(f"GraphQL response has null 'pullRequest' for PR #{pr_number}")
                    break

                threads_data = pr_data.get("reviewThreads")
                if not threads_data:
                    logger.error(f"GraphQL response missing 'reviewThreads' for PR #{pr_number}")
                    break

                threads = threads_data.get("nodes", [])
                page_info = threads_data.get("pageInfo", {})

                for thread in threads:
                    if not thread.get("isResolved", False):  # Only unresolved threads
                        comments = thread.get("comments", {}).get("nodes", [])
                        if comments:
                            # Get the original (first) comment in the thread
                            original_comment = comments[0]
                            author_login = original_comment["author"]["login"]
                            # Determine if bot by checking login suffix
                            author_type = "Bot" if author_login.endswith("[bot]") else "User"

                            comment = GitHubReviewComment(
                                id=original_comment["id"],
                                thread_id=thread["id"],
                                pr_number=pr_number,
                                body=original_comment.get("body", ""),
                                path=original_comment.get("path", ""),
                                line=original_comment.get("line", 0),
                                commit_id=original_comment.get("originalCommit", {}).get("oid", ""),
                                user=GitHubUser(
                                    login=author_login,
                                    type=author_type,
                                ),
                                created_at=original_comment["createdAt"],
                                state="active",
                                diff_hunk=original_comment.get("diffHunk", ""),
                                start_line=original_comment.get("startLine"),
                            )
                            all_comments.append(comment)

                has_next_page = page_info.get("hasNextPage", False)
                if has_next_page:
                    after_cursor = page_info.get("endCursor")

            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing GraphQL response for PR #{pr_number}: {e}")
                break

        return all_comments

    def get_all_unresolved_comments(
        self,
    ) -> List[GitHubReviewComment]:
        """Get all unresolved review comments across all open PRs, sorted by creation time"""
        all_comments = []

        for pr in self.list_open_prs():
            comments = self.get_pr_review_comments(pr.number)
            all_comments.extend(comments)

        # Sort by created_at to get oldest first
        all_comments.sort(key=lambda c: c.created_at)

        return all_comments

    def reply_to_review_thread(
        self,
        pr_number: int,
        thread_id: str,
        body: str,
    ) -> bool:
        """Reply to a review comment thread using GraphQL"""
        mutation = """
        mutation($body: String!, $thread_id: ID!) {
            addPullRequestReviewThreadReply(input: {
                body: $body
                pullRequestReviewThreadId: $thread_id
            }) {
                comment {
                    id
                    body
                }
            }
        }
        """

        variables = {
            "body": body,
            "thread_id": thread_id,
        }

        result = self._gh_graphql(mutation, variables)

        if result and "data" in result:
            logger.info(f"Replied to review thread on PR #{pr_number}")
            return True
        else:
            logger.error(f"Failed to reply to review thread on PR #{pr_number}")
            if result and "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
            return False

    def resolve_conversation_graphql(
        self,
        thread_id: str,
    ) -> bool:
        """Resolve a review conversation using GraphQL mutation"""
        # The threadId is a global ID that already encodes the repository
        # We don't need repositoryId in the mutation
        mutation = """
        mutation($thread_id: ID!) {
            resolveReviewThread(input: {
                threadId: $thread_id
            }) {
                thread {
                    isResolved
                }
            }
        }
        """

        variables = {
            "thread_id": thread_id,
        }

        result = self._gh_graphql(mutation, variables)

        if result and "data" in result:
            logger.info(f"Resolved review conversation: {thread_id}")
            return True
        else:
            logger.error(f"Failed to resolve review conversation: {thread_id}")
            if result and "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
            return False

    def add_pr_comment(
        self,
        pr_number: int,
        body: str,
    ) -> bool:
        """Add a comment to a PR (for triggering review with /gemini review)"""
        result = self._run_gh(
            [
                "pr",
                "comment",
                str(pr_number),
                "--body",
                body,
            ]
        )

        if result.returncode == 0:
            logger.info(f"Added comment to PR #{pr_number}")
            return True
        else:
            logger.error(f"Failed to add comment to PR #{pr_number}: {result.stderr}")
            return False
