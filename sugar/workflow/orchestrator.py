"""
Workflow Orchestrator - Apply consistent git/GitHub workflows to all Sugar work
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowProfile(Enum):
    SOLO = "solo"
    BALANCED = "balanced"
    ENTERPRISE = "enterprise"


class WorkflowType(Enum):
    DIRECT_COMMIT = "direct_commit"
    PULL_REQUEST = "pull_request"


class WorkflowOrchestrator:
    """Manages consistent workflows for all Sugar work items"""

    def __init__(self, config: Dict[str, Any], git_ops=None, work_queue=None, github_watcher=None):
        self.config = config
        self.git_ops = git_ops
        self.work_queue = work_queue
        self.github_watcher = github_watcher
        self.workflow_config = self._load_workflow_config()

    def _get_nested_config(self, *keys, default=None):
        """Helper to safely access nested config values.

        Args:
            *keys: Sequence of keys to traverse the config dict
            default: Default value if any key in path is missing

        Example:
            self._get_nested_config("sugar", "discovery", "github", "review_comments")
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value if value is not None else default

        # Initialize quality gates coordinator if enabled
        self.quality_gates = None
        if config.get("quality_gates", {}).get("enabled", False):
            from ..quality_gates import QualityGatesCoordinator

            self.quality_gates = QualityGatesCoordinator(config)
            logger.info("🔒 Quality Gates enabled for workflow validation")

    def _load_workflow_config(self) -> Dict[str, Any]:
        """Load and validate workflow configuration"""
        workflow_config = self.config.get("sugar", {}).get("workflow", {})

        # Set defaults based on profile
        profile = WorkflowProfile(workflow_config.get("profile", "solo"))

        if profile == WorkflowProfile.SOLO:
            defaults = {
                "git": {
                    "workflow_type": "direct_commit",
                    "commit_style": "conventional",
                    "auto_commit": True,
                },
                "github": {
                    "auto_create_issues": False,
                    "update_existing_issues": True,  # Still update if work comes from GitHub
                },
                "discovery": {"handle_internally": True},  # No external issue creation
            }
        elif profile == WorkflowProfile.BALANCED:
            defaults = {
                "git": {
                    "workflow_type": "pull_request",
                    "commit_style": "conventional",
                    "auto_commit": True,
                },
                "github": {
                    "auto_create_issues": True,
                    "selective_creation": True,
                    "min_priority": 3,
                },
                "discovery": {"handle_internally": False},
            }
        else:  # ENTERPRISE
            defaults = {
                "git": {
                    "workflow_type": "pull_request",
                    "commit_style": "conventional",
                    "auto_commit": False,
                    "require_review": True,
                },
                "github": {
                    "auto_create_issues": True,
                    "selective_creation": False,
                    "issue_templates": True,
                },
                "discovery": {"handle_internally": False},
            }

        # Merge user config with defaults
        merged = {**defaults, **workflow_config}
        merged["profile"] = profile

        logger.debug(f"🔧 Loaded workflow config for {profile.value} profile")
        return merged

    def get_workflow_for_work_item(self, work_item: Dict[str, Any]) -> Dict[str, Any]:
        """Determine appropriate workflow for a work item"""
        source = work_item.get("source", work_item.get("source_type", "unknown"))
        work_type = work_item.get("type", work_item.get("work_type", "unknown"))
        priority = work_item.get("priority", 3)

        workflow = {
            "git_workflow": WorkflowType(self.workflow_config["git"]["workflow_type"]),
            "commit_style": self.workflow_config["git"]["commit_style"],
            "auto_commit": self.workflow_config["git"].get("auto_commit", True),
            "create_github_issue": False,  # Default to internal handling
            "update_github_issue": False,
            "branch_name": None,
            "commit_message_template": self._get_commit_template(work_type),
        }

        # Handle PR review comments specially - use existing PR branch
        if work_type == "pr_review_comment":
            context = work_item.get("context", {})
            workflow = {
                "git_workflow": WorkflowType.DIRECT_COMMIT,
                "commit_style": "conventional",
                "auto_commit": True,
                "create_github_issue": False,
                "update_github_issue": False,
                "branch_name": context.get("branch"),  # Use existing PR branch
                "checkout_existing_branch": True,  # Flag to checkout existing branch
                "commit_message_template": "fix: address review comment on {file_path}:{line}",
                "push_after_commit": True,
                "resolve_conversation": True,
            }
            logger.debug(
                f"🔄 Determined workflow for pr_review_comment: checkout existing branch {workflow['branch_name']}"
            )
            return workflow

        # Handle GitHub-sourced work differently
        if source == "github_watcher":
            workflow["update_github_issue"] = True
            # Use existing GitHub workflow settings
            github_config = (
                self.config.get("sugar", {}).get("discovery", {}).get("github", {})
            )
            git_workflow = github_config.get("workflow", {}).get(
                "git_workflow", "direct_commit"
            )
            workflow["git_workflow"] = WorkflowType(git_workflow)

        # Apply source-specific overrides for solo profile
        elif self.workflow_config["profile"] == WorkflowProfile.SOLO:
            if source in ["error_logs"] and priority >= 4:
                # High priority errors might need different handling
                workflow["commit_message_template"] = "fix: {title}"

        logger.debug(
            f"🔄 Determined workflow for {source}/{work_type}: {workflow['git_workflow'].value}"
        )
        return workflow

    def _get_commit_template(self, work_type: str) -> str:
        """Get conventional commit message template based on work type"""
        templates = {
            "bug_fix": "fix: {title}",
            "feature": "feat: {title}",
            "test": "test: {title}",
            "refactor": "refactor: {title}",
            "documentation": "docs: {title}",
            "code_quality": "refactor: {title}",
            "test_coverage": "test: {title}",
        }

        return templates.get(work_type, "chore: {title}")

    def format_commit_message(
        self, work_item: Dict[str, Any], workflow: Dict[str, Any]
    ) -> str:
        """Format commit message according to workflow style"""
        template = workflow["commit_message_template"]
        title = work_item.get("title", "Unknown work")
        work_id = work_item.get("id", "unknown")

        # Get context for PR review comments
        context = work_item.get("context", {})

        if workflow["commit_style"] == "conventional":
            # For PR review comments, include file_path and line from context
            if work_item.get("type") == "pr_review_comment":
                file_path = context.get("github_pr_comment", {}).get("path", "unknown")
                line = context.get("github_pr_comment", {}).get("line", 0)
                message = template.format(title=title, file_path=file_path, line=line)
            else:
                # Use the template as-is (already conventional format)
                message = template.format(title=title)
        else:
            # Simple format
            message = title

        # Add work item ID for traceability
        message += f"\n\nWork ID: {work_id}"

        # Add Sugar attribution
        from ..__version__ import get_version_info

        message += f"\nGenerated with {get_version_info()}"

        return message

    async def prepare_work_execution(self, work_item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare work item for execution with proper workflow"""
        workflow = self.get_workflow_for_work_item(work_item)

        # Check dry_run mode
        dry_run = self.config.get("sugar", {}).get("dry_run", False)

        # Handle PR review comments - checkout existing PR branch
        if workflow.get("checkout_existing_branch") and workflow.get("branch_name") and self.git_ops:
            branch_name = workflow["branch_name"]
            if not dry_run:
                try:
                    success = await self.git_ops.checkout_existing_branch(branch_name)
                    if success:
                        logger.info(f"🌿 Checked out existing PR branch: {branch_name}")
                    else:
                        logger.warning(
                            f"⚠️ Failed to checkout branch {branch_name}, using current branch"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Branch checkout failed: {e}"
                    )
            else:
                logger.info(f"🧪 DRY RUN: Would checkout existing branch: {branch_name}")
            return workflow

        # Create branch if using PR workflow (skip in dry_run)
        if not dry_run and workflow["git_workflow"] == WorkflowType.PULL_REQUEST and self.git_ops:
            branch_name = self._generate_branch_name(work_item)
            workflow["branch_name"] = branch_name

            try:
                success = await self.git_ops.create_branch(branch_name)
                if success:
                    logger.info(f"🌿 Created workflow branch: {branch_name}")
                else:
                    logger.warning(
                        f"⚠️ Failed to create branch {branch_name}, using current branch"
                    )
                    workflow["git_workflow"] = WorkflowType.DIRECT_COMMIT
            except Exception as e:
                logger.warning(
                    f"⚠️ Branch creation failed, falling back to direct commit: {e}"
                )
                workflow["git_workflow"] = WorkflowType.DIRECT_COMMIT
        elif dry_run and workflow["git_workflow"] == WorkflowType.PULL_REQUEST:
            branch_name = self._generate_branch_name(work_item)
            workflow["branch_name"] = branch_name
            logger.info(f"🧪 DRY RUN: Would create branch: {branch_name}")

        return workflow

    async def complete_work_execution(
        self,
        work_item: Dict[str, Any],
        workflow: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> bool:
        """Complete workflow after work execution"""

        # Check dry_run mode - skip all git operations
        dry_run = self.config.get("sugar", {}).get("dry_run", False)
        if dry_run:
            branch_name = workflow.get("branch_name")
            if branch_name:
                logger.info(f"🧪 DRY RUN: Would commit, push, and create PR for branch: {branch_name}")
            else:
                logger.info("🧪 DRY RUN: Would commit changes")
            return True

        # Run self-verification before any completion steps (AUTO-005)
        if self.quality_gates and self.quality_gates.is_enabled():
            verification_passed = await self._run_self_verification(
                work_item, execution_result
            )
            if not verification_passed:
                logger.error("🚫 Self-verification failed - blocking work completion")
                return False

        if not workflow.get("auto_commit", True):
            logger.info("🔧 Auto-commit disabled, skipping git operations")
            return True

        if not self.git_ops:
            logger.warning("⚠️ No git operations available")
            return False

        try:
            # Check if there are changes to commit
            has_changes = await self.git_ops.has_uncommitted_changes()
            if not has_changes:
                logger.info("📝 No changes to commit")

                # For PR review comments, still reply to the thread explaining why
                if workflow.get("resolve_conversation") and self.github_watcher:
                    context = work_item.get("context", {})
                    pr_number = context.get("github_pr_comment", {}).get("pr_number")
                    thread_id_hash = context.get("thread_id_hash")
                    thread_id = context.get("github_pr_comment", {}).get("thread_id")

                    if pr_number and thread_id_hash and thread_id:
                        # Get the summary or reason from execution result
                        summary = execution_result.get("summary", "")
                        result = execution_result.get("result", "")

                        # Build a reply explaining why no changes were made
                        if summary or result:
                            reply_body = f"""I've reviewed this comment but no changes were needed.

**Analysis:**
{summary[:1000] if summary else result[:1000]}

**Reason:** The code appears to already meet the requirements or the issue may be addressed differently."""
                        else:
                            reply_body = """I've reviewed this comment but no changes were made.

**Reason:** Unable to make changes - this could be because:
- The code already meets the requirements
- The issue requires more context or clarification
- Changes would be outside the scope of this task

Please provide more details if you'd like me to revisit this."""

                        await self.github_watcher.reply_to_review_thread(
                            pr_number, thread_id, reply_body
                        )
                        logger.info(f"Replied to review thread for PR #{pr_number} (no changes made)")

                        # Resolve the conversation even though no changes were made
                        await self.github_watcher.resolve_conversation_graphql(thread_id)
                        logger.info(f"Resolved review conversation for PR #{pr_number} (no changes)")

                        # Record the response even though no changes were made
                        await self.github_watcher.issue_response_manager.initialize()
                        await self.github_watcher.issue_response_manager.record_response(
                            repo=context.get("repo"),
                            issue_number=thread_id_hash,
                            response_type="pr_review_comment",
                            confidence=1.0,
                            response_content=reply_body,
                            was_auto_posted=True,
                            work_item_id=work_item.get("id"),
                        )

                return True

            # Run quality gate validation before committing
            if self.quality_gates and self.quality_gates.is_enabled():
                logger.info("🔒 Running quality gate validation before commit")

                # Get list of changed files
                changed_files = await self._get_changed_files()

                # Extract claims from execution result if available
                claims = self._extract_claims_from_result(execution_result)

                # Validate with quality gates
                can_commit, gate_result = (
                    await self.quality_gates.validate_before_commit(
                        task=work_item, changed_files=changed_files, claims=claims
                    )
                )

                if not can_commit:
                    logger.error(
                        f"❌ Quality gate validation failed: {gate_result.reason}"
                    )
                    logger.error("🚫 Blocking commit - quality requirements not met")

                    # Store failure information in work item
                    if self.work_queue:
                        await self.work_queue.update_work(
                            work_item["id"],
                            {
                                "quality_gate_status": "failed",
                                "quality_gate_reason": gate_result.reason,
                                "quality_gate_details": gate_result.to_dict(),
                            },
                        )

                    return False

                logger.info(f"✅ Quality gates passed: {gate_result.reason}")

                # Add quality gate evidence to commit message
                quality_footer = self.quality_gates.get_commit_message_footer(
                    gate_result
                )
            else:
                quality_footer = ""

            # Format commit message
            commit_message = self.format_commit_message(work_item, workflow)

            # Append quality gate evidence if available
            if quality_footer:
                commit_message += quality_footer

            # Commit changes
            success = await self.git_ops.commit_changes(commit_message)
            if not success:
                logger.error("❌ Failed to commit changes")
                return False

            # Capture commit SHA and store in database for traceability
            if self.work_queue:
                commit_sha = await self.git_ops.get_latest_commit_sha()
                if commit_sha:
                    work_id = work_item.get("id")
                    if work_id:
                        await self.work_queue.update_commit_sha(work_id, commit_sha)
                        logger.debug(
                            f"🔗 Linked commit {commit_sha[:8]} to work item {work_id}"
                        )

            # Handle PR workflow
            if workflow["git_workflow"] == WorkflowType.PULL_REQUEST:
                branch_name = workflow.get("branch_name")
                if branch_name:
                    # Push branch
                    push_success = await self.git_ops.push_branch(branch_name)
                    if push_success:
                        logger.info(f"📤 Pushed branch {branch_name}")
                        # Create PR if configured and GitHub watcher is available
                        await self._create_pull_request_if_enabled(
                            work_item, workflow, branch_name
                        )
                    else:
                        logger.error(f"❌ Failed to push branch {branch_name}")
                        return False

            # Handle PR review comment workflow - push and respond to review
            if workflow.get("resolve_conversation") and self.github_watcher:
                branch_name = workflow.get("branch_name")
                if branch_name:
                    # Push branch
                    push_success = await self.git_ops.push_branch(branch_name)
                    if push_success:
                        logger.info(f"📤 Pushed branch {branch_name} for PR review comment")

                        # Get context from work item
                        context = work_item.get("context", {})
                        pr_number = context.get("github_pr_comment", {}).get("pr_number")
                        thread_id_hash = context.get("thread_id_hash")
                        thread_id = context.get("github_pr_comment", {}).get("thread_id")

                        # Handle GitHub interactions
                        if pr_number and thread_id_hash:
                            # Step 1: Reply to review thread with summary of changes
                            summary = execution_result.get("summary", "Changes have been made to address the review comment.")
                            reply_body = f"""I've addressed this review comment and pushed the changes.

**Changes made:**
{summary[:1000]}

Please review the updated commit."""
                            await self.github_watcher.reply_to_review_thread(
                                pr_number, thread_id, reply_body
                            )

                            # Step 2: Resolve the conversation
                            await self.github_watcher.resolve_conversation_graphql(thread_id)
                            logger.info(f"Resolved review conversation for PR #{pr_number}")

                            # Step 3: Trigger re-review with configured command
                            review_config = self._get_nested_config(
                                "sugar", "discovery", "github", "review_comments", default={}
                            )
                            re_review_command = review_config.get("re_review_command", "/gemini review")
                            await self.github_watcher.add_pr_comment(pr_number, re_review_command)
                            logger.info(f"Triggered re-review for PR #{pr_number}")

                            # Record the response
                            await self.github_watcher.issue_response_manager.initialize()
                            await self.github_watcher.issue_response_manager.record_response(
                                repo=context.get("repo"),
                                issue_number=thread_id_hash,  # Using thread_id_hash as unique identifier
                                response_type="pr_review_comment",
                                confidence=1.0,
                                response_content=reply_body,
                                was_auto_posted=True,
                                work_item_id=work_item.get("id"),
                            )
                    else:
                        logger.error(f"❌ Failed to push branch {branch_name}")
                        return False

            logger.info(f"✅ Completed {workflow['git_workflow'].value} workflow")
            return True

        except Exception as e:
            logger.error(f"❌ Workflow completion failed: {e}")
            return False

    async def _create_pull_request_if_enabled(
        self, work_item: Dict[str, Any], workflow: Dict[str, Any], branch_name: str
    ):
        """Create pull request if enabled in config and GitHub watcher is available"""
        if not self.github_watcher:
            logger.debug("No GitHub watcher available for PR creation")
            return

        # Check if PR creation is enabled for GitHub-sourced work
        github_config = (
            self.config.get("sugar", {}).get("discovery", {}).get("github", {})
        )
        pr_config = github_config.get("workflow", {}).get("pull_request", {})

        if not pr_config.get("auto_create", True):
            logger.debug("PR auto-creation is disabled in config")
            return

        try:
            # Get PR details
            issue_number = work_item.get("context", {}).get("github_issue", {}).get("number")
            work_title = work_item.get("title", "")
            source = work_item.get("source", work_item.get("source_type", "unknown"))

            if issue_number:
                # GitHub-sourced work: use issue-based PR
                issue_title = work_title.replace("Address GitHub issue: ", "")

                # Format PR title
                title_pattern = pr_config.get("title_pattern", "Fix #{issue_number}: {issue_title}")
                variables = {"issue_number": issue_number, "issue_title": issue_title}
                pr_title = self.git_ops.format_pr_title(title_pattern, variables)

                # Create PR body
                pr_body = f"Fixes #{issue_number}\n\n"
                if pr_config.get("include_work_summary", True):
                    work_summary = work_item.get("description", issue_title)
                    pr_body += f"## Summary\n{work_summary}\n\n"

                pr_body += "\n---\n*This PR was automatically created by Sugar AI*"
            else:
                # Non-GitHub work: use task-based PR
                # Generate a clean title from the work item
                pr_title = work_title.split("\n")[0][:80]  # First line, max 80 chars
                if len(pr_title) < len(work_title):
                    pr_title = pr_title.rstrip(".")

                # Prefix with source if not CLI
                if source != "cli":
                    pr_title = f"{source.capitalize()}: {pr_title}"

                # Create PR body
                pr_body = f"## Task\n{work_title}\n\n"
                if work_item.get("description"):
                    pr_body += f"## Description\n{work_item['description']}\n\n"

                # Add task metadata
                task_type = work_item.get("type", "unknown")
                priority = work_item.get("priority", 3)
                pr_body += f"\n---\n*Type: {task_type} | Priority: {priority} | Source: {source}*\n"
                pr_body += "*This PR was automatically created by Sugar AI*"

            # Get base branch
            base_branch = github_config.get("workflow", {}).get("branch", {}).get("base_branch", "main")

            # Create PR
            pr_url = await self.github_watcher.create_pull_request(
                branch_name, pr_title, pr_body, base_branch
            )

            if pr_url:
                logger.info(f"🔀 Created pull request: {pr_url}")

                # For GitHub-sourced work, comment on issue and close it
                if issue_number:
                    # Comment on issue with PR link
                    completion_comment = f"✅ Work completed. Pull request created: {pr_url}"
                    await self.github_watcher.comment_on_issue(issue_number, completion_comment)

                    # Close issue if configured
                    if not pr_config.get("auto_merge", False) and github_config.get(
                        "workflow", {}
                    ).get("auto_close_issues", True):
                        await self.github_watcher.close_issue(issue_number)
                        logger.info(f"🔒 Closed GitHub issue #{issue_number}")
            else:
                logger.warning("PR creation returned no URL")

        except Exception as e:
            logger.error(f"Error creating pull request: {e}")

    def _generate_branch_name(self, work_item: Dict[str, Any]) -> str:
        """Generate branch name for work item"""
        # For GitHub-sourced work, use the configured branch pattern
        source = work_item.get("source", work_item.get("source_type", "unknown"))
        if source == "github_watcher":
            github_config = (
                self.config.get("sugar", {}).get("discovery", {}).get("github", {})
            )
            branch_config = github_config.get("workflow", {}).get("branch", {})
            name_pattern = branch_config.get("name_pattern", "sugar/issue-{issue_number}")

            # Get issue number from context
            issue_number = work_item.get("context", {}).get("github_issue", {}).get("number", "unknown")
            return name_pattern.format(issue_number=issue_number)

        # Default pattern for non-GitHub work
        work_id = work_item.get("id", "unknown")[:8]  # Short ID
        work_type = work_item.get("type", "work")

        # Clean title for branch name
        title = work_item.get("title", "unknown")
        clean_title = "".join(c for c in title.lower() if c.isalnum() or c in "-_")[:30]

        return f"sugar/{source}/{work_type}-{clean_title}-{work_id}"

    async def _get_changed_files(self) -> List[str]:
        """Get list of changed files for quality gate validation"""
        if not self.git_ops:
            return []

        try:
            changed_files = await self.git_ops.get_changed_files()
            return changed_files if changed_files else []
        except Exception as e:
            logger.warning(f"Could not get changed files: {e}")
            return []

    def _extract_claims_from_result(
        self, execution_result: Dict[str, Any]
    ) -> List[str]:
        """Extract claims from execution result for truth enforcement"""
        claims = []

        # Look for explicit claims in result
        if "claims" in execution_result:
            claims.extend(execution_result["claims"])

        # Extract implicit claims from summary/actions
        summary = execution_result.get("summary", "").lower()
        actions = execution_result.get("actions_taken", [])

        # Common claim patterns
        claim_patterns = {
            "all tests pass": ["tests pass", "all tests passed", "tests successful"],
            "functionality verified": ["verified", "tested", "confirmed working"],
            "no errors": ["no errors", "error-free", "without errors"],
            "implementation complete": ["complete", "implemented", "finished"],
        }

        for claim, patterns in claim_patterns.items():
            if any(pattern in summary for pattern in patterns):
                claims.append(claim)
            for action in actions:
                if any(pattern in str(action).lower() for pattern in patterns):
                    if claim not in claims:
                        claims.append(claim)
                    break

        return claims

    async def _run_self_verification(
        self,
        work_item: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> bool:
        """
        Run self-verification before allowing task completion (AUTO-005).

        This method integrates the QualityGatesCoordinator's verification
        gate to ensure tasks self-verify before being marked as complete.

        Args:
            work_item: The work item being completed
            execution_result: The result from task execution

        Returns:
            True if verification passed, False otherwise
        """
        if not self.quality_gates:
            return True

        try:
            task_id = work_item.get("id", "unknown")
            logger.info(f"🔍 Running self-verification for task {task_id}")

            # Run the verification gate
            can_complete, gate_result = (
                await self.quality_gates.validate_before_completion(
                    work_item=work_item,
                    execution_result=execution_result,
                )
            )

            # Store verification results in work item
            if self.work_queue and gate_result.verification_results:
                verification_updates = {
                    "verification_status": gate_result.verification_results.status.value,
                    "verification_results": gate_result.verification_results.to_dict(),
                }
                await self.work_queue.update_work(work_item["id"], verification_updates)

            if can_complete:
                logger.info(f"✅ Self-verification passed for task {task_id}")
                return True
            else:
                logger.warning(
                    f"❌ Self-verification failed for task {task_id}: {gate_result.reason}"
                )

                # Store failure information
                if self.work_queue:
                    await self.work_queue.update_work(
                        work_item["id"],
                        {
                            "verification_status": "failed",
                            "quality_gate_status": "failed",
                            "quality_gate_reason": gate_result.reason,
                            "quality_gate_details": gate_result.to_dict(),
                        },
                    )

                return False

        except Exception as e:
            logger.error(f"Error during self-verification: {e}")
            # On error, allow completion but log the issue
            return True
