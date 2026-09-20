"""Controlled task execution with clean workspaces and isolated evaluation."""

from __future__ import annotations

import asyncio
import json
import platform
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from contextlab.agent.models import AgentState
from contextlab.agent.runtime import CodingAgent
from contextlab.benchmarking.config import ContextBudgetVariant, ExperimentConfig
from contextlab.config import AgentConfig, BudgetConfig, PolicyConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.factory import create_policy
from contextlab.evaluation import EvaluationResult, evaluate_task
from contextlab.inference.client import InferenceClient, OpenAIClient
from contextlab.inference.mock import ScriptedInference
from contextlab.instrumentation.events import EventLogger
from contextlab.tasks import TaskDefinition, load_task
from contextlab.tools.execution import MutationTools
from contextlab.tools.repository import RepositoryTools
from contextlab.tools.workspace import RepositoryWorkspace


class RunRecord(BaseModel):
    experiment_id: str
    run_id: str
    task_id: str
    policy_id: str
    budget_name: str
    seed: int
    trial: int
    execution_index: int
    baseline_revision: str
    model: str
    tokenizer: str
    hardware: dict[str, str]
    inference_server: dict[str, Any]
    official_evaluation: EvaluationResult
    completed: bool
    failure: str | None
    files_modified: list[str]
    wall_seconds: float
    inference: dict[str, Any]
    tool_calls: int
    trace: str
    patch: str


InferenceFactory = Callable[[TaskDefinition, str], InferenceClient]


def mock_inference_for(task: TaskDefinition, policy: str) -> ScriptedInference:
    del policy
    if task.id != "localized-divide-zero":
        raise ValueError(f"no deterministic mock solution is registered for {task.id}")
    decisions = [
        {"action": "tool", "tool": {"name": "list_files", "arguments": {}}},
        {"action": "tool", "tool": {"name": "read_file", "arguments": {"path": "calculator.py"}}},
        {
            "action": "tool",
            "tool": {
                "name": "write_file",
                "arguments": {
                    "path": "calculator.py",
                    "content": (
                        "def divide(a: float, b: float) -> float:\n"
                        '    """Divide a by b and reject a zero denominator."""\n'
                        "    if b == 0:\n"
                        '        raise ZeroDivisionError("cannot divide by zero")\n'
                        "    return a / b\n"
                    ),
                },
            },
        },
        {
            "action": "tool",
            "tool": {"name": "run_command", "arguments": {"argv": ["pytest", "-q"]}},
        },
        {"action": "finish", "summary": "fixed denominator guard and validated visible tests"},
    ]
    return ScriptedInference([json.dumps(item) for item in decisions])


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        inference_factory: InferenceFactory | None = None,
    ) -> None:
        self.config = config
        self.inference_factory = inference_factory or self._default_inference

    def _default_inference(self, task: TaskDefinition, policy: str) -> InferenceClient:
        if self.config.mock:
            return mock_inference_for(task, policy)
        return OpenAIClient(self.config.model)

    async def run(self) -> Path:
        output = self.config.output_directory / self.config.experiment_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="contextlab-baselines-") as temporary:
            baseline_root = Path(temporary)
            for task_path in self.config.tasks:
                task = load_task(task_path)
                baseline, revision = self._prepare_baseline(task, baseline_root / task.id)
                for trial in range(self.config.trials):
                    seed = self.config.seeds[trial]
                    for budget in self.config.budgets:
                        for execution_index, policy in enumerate(
                            self.ordered_policies(trial=trial, seed=seed)
                        ):
                            record = await self._run_one(
                                task,
                                baseline,
                                revision,
                                policy,
                                budget,
                                trial,
                                seed,
                                execution_index,
                                output,
                            )
                            run_path = output / "runs" / f"{record.run_id}.json"
                            run_path.parent.mkdir(parents=True, exist_ok=True)
                            run_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return output

    async def _run_one(
        self,
        task: TaskDefinition,
        baseline: Path,
        revision: str,
        policy_name: str,
        budget_variant: ContextBudgetVariant,
        trial: int,
        seed: int,
        execution_index: int,
        output: Path,
    ) -> RunRecord:
        run_id = f"{task.id}-{budget_variant.name}-{policy_name}-t{trial}-{uuid4().hex[:8]}"
        started = time.perf_counter()
        with RepositoryWorkspace.from_revision(baseline, revision) as workspace:
            inference = self.inference_factory(task, policy_name)
            agent_config = AgentConfig(
                model=self.config.model,
                budget=BudgetConfig(
                    context_window=budget_variant.context_window,
                    reserved_output_tokens=budget_variant.reserved_output_tokens,
                ),
                policy=PolicyConfig(
                    name=policy_name,
                    deterministic_compaction=self.config.mock,
                ),
                allowed_commands=["pytest", "python"],
            )
            policy = create_policy(
                agent_config.policy, repository_root=workspace.root, inference=inference
            )
            agent = CodingAgent(
                config=agent_config,
                inference=inference,
                policy=policy,
                repository_tools=RepositoryTools(workspace),
                mutation_tools=MutationTools(
                    workspace,
                    allowed_commands=agent_config.allowed_commands,
                    command_timeout=agent_config.limits.command_timeout_seconds,
                ),
                budget=TokenBudget(
                    agent_config.budget,
                    model=self.config.model.model,
                ),
            )
            state = AgentState(
                experiment_id=self.config.experiment_id,
                run_id=run_id,
                task_id=task.id,
                policy_id=policy_name,
                system_instruction=(
                    "You are an autonomous coding agent in an isolated repository. Use only the "
                    "provided tools, make the smallest correct change, and run visible tests."
                ),
                task_instruction=task.description,
                objective=task.success_criteria,
            )
            final = await agent.run(state, seed=seed)
            official = await asyncio.to_thread(evaluate_task, task, workspace.root)
            trace_path = output / "traces" / f"{run_id}.jsonl"
            EventLogger(trace_path).write_all(final.events)
            patch = subprocess.run(
                ["git", "diff", "--no-ext-diff"],
                cwd=workspace.root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            totals = agent.inference_totals
            return RunRecord(
                experiment_id=self.config.experiment_id,
                run_id=run_id,
                task_id=task.id,
                policy_id=policy_name,
                budget_name=budget_variant.name,
                seed=seed,
                trial=trial,
                execution_index=execution_index,
                baseline_revision=revision,
                model=self.config.model.model,
                tokenizer=self.config.model.tokenizer or self.config.model.model,
                hardware={"platform": platform.platform(), "machine": platform.machine()},
                inference_server={
                    "endpoint": self.config.model.endpoint,
                    "temperature": self.config.model.temperature,
                    "mock": self.config.mock,
                },
                official_evaluation=official,
                completed=final.completed,
                failure=final.failure,
                files_modified=sorted(final.files_modified),
                wall_seconds=time.perf_counter() - started,
                inference=totals.model_dump() | {"total_tokens": totals.total_tokens},
                tool_calls=agent.tool_calls,
                trace=str(trace_path.relative_to(output)),
                patch=patch,
            )

    def ordered_policies(self, *, trial: int, seed: int) -> list[str]:
        policies: list[str] = list(self.config.policies)
        if self.config.order == "rotate" and policies:
            offset = trial % len(policies)
            return [*policies[offset:], *policies[:offset]]
        if self.config.order == "randomize":
            random.Random(seed).shuffle(policies)
        return policies

    @staticmethod
    def _prepare_baseline(task: TaskDefinition, target: Path) -> tuple[Path, str]:
        shutil.copytree(task.repository, target)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=target, check=True)
        subprocess.run(
            ["git", "config", "user.email", "benchmark@contextlab.local"], cwd=target, check=True
        )
        subprocess.run(["git", "config", "user.name", "ContextLab"], cwd=target, check=True)
        subprocess.run(["git", "add", "."], cwd=target, check=True)
        env = {
            **__import__("os").environ,
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        }
        subprocess.run(
            ["git", "commit", "-qm", "benchmark baseline"], cwd=target, env=env, check=True
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=target, capture_output=True, text=True, check=True
        ).stdout.strip()
        return target, revision
