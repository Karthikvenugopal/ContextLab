import json
from pathlib import Path

import pytest

from contextlab.agent.models import AgentState
from contextlab.agent.runtime import CodingAgent
from contextlab.config import AgentConfig, BudgetConfig, PolicyConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.factory import create_policy
from contextlab.evaluation import evaluate_task
from contextlab.inference.mock import ScriptedInference
from contextlab.tasks.models import TaskDefinition
from contextlab.tools.execution import MutationTools
from contextlab.tools.repository import RepositoryTools
from contextlab.tools.workspace import RepositoryWorkspace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy_name", ["full-history", "bounded-tool-output", "retrieval", "compaction"]
)
async def test_agent_completes_real_task_under_each_policy(
    tmp_path: Path, policy_name: str
) -> None:
    workspace_path = tmp_path / "repo"
    workspace_path.mkdir()
    (workspace_path / "mathlib.py").write_text("def add(a, b):\n    return a - b\n")
    (workspace_path / "test_mathlib.py").write_text(
        "from mathlib import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_official.py").write_text(
        "from mathlib import add\n\ndef test_negative():\n    assert add(-2, 3) == 1\n"
    )
    corrected = "def add(a, b):\n    return a + b\n"
    responses = [
        json.dumps({"action": "tool", "tool": {"name": "list_files", "arguments": {}}}),
        json.dumps(
            {"action": "tool", "tool": {"name": "read_file", "arguments": {"path": "mathlib.py"}}}
        ),
        json.dumps(
            {
                "action": "tool",
                "tool": {
                    "name": "write_file",
                    "arguments": {"path": "mathlib.py", "content": corrected},
                },
            }
        ),
        json.dumps(
            {
                "action": "tool",
                "tool": {"name": "run_command", "arguments": {"argv": ["pytest", "-q"]}},
            }
        ),
        json.dumps({"action": "finish", "summary": "fixed addition and ran tests"}),
    ]
    config = AgentConfig(
        budget=BudgetConfig(context_window=4096, reserved_output_tokens=256),
        policy=PolicyConfig(
            name=policy_name,  # type: ignore[arg-type]
            deterministic_compaction=True,
            compaction_every_steps=1,
        ),
        allowed_commands=["pytest"],
    )
    inference = ScriptedInference(responses)
    workspace = RepositoryWorkspace(workspace_path)
    policy = create_policy(config.policy, repository_root=workspace_path, inference=inference)
    agent = CodingAgent(
        config=config,
        inference=inference,
        policy=policy,
        repository_tools=RepositoryTools(workspace),
        mutation_tools=MutationTools(workspace, allowed_commands=["pytest"]),
        budget=TokenBudget(config.budget, model="mock"),
    )
    state = AgentState(
        experiment_id="integration",
        run_id=f"run-{policy_name}",
        task_id="fix-add",
        policy_id=policy_name,
        system_instruction="Act as a coding agent. Inspect, edit, and validate the repository.",
        task_instruction="Fix add so visible and hidden arithmetic cases pass.",
        objective="all tests pass",
    )
    result = await agent.run(state)
    task = TaskDefinition(
        id="fix-add",
        category="localized_bug",
        repository=workspace_path,
        revision="baseline",
        description="fix add",
        evaluation_tests=hidden,
        success_criteria="hidden test passes",
    )
    official = evaluate_task(task, workspace_path)
    assert result.completed
    assert result.files_modified == {"mathlib.py"}
    assert official.success
    assert inference.requests
