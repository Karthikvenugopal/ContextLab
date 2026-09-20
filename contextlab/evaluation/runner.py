"""Execute evaluation-only tests outside the agent-visible workspace."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel

from contextlab.tasks.models import TaskDefinition


class EvaluationResult(BaseModel):
    success: bool
    passed: int
    failed: int
    errors: int
    skipped: int
    pass_rate: float
    exit_code: int
    duration_seconds: float
    output: str


def evaluate_task(
    task: TaskDefinition, workspace: Path, *, timeout_seconds: float = 180
) -> EvaluationResult:
    """Run hidden tests by path; they are never copied into the agent workspace."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="contextlab-evaluation-") as temporary:
        report = Path(temporary) / "junit.xml"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(workspace.resolve()),
            "CONTEXTLAB_EVALUATION": "1",
        }
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(task.evaluation_tests),
                    f"--junitxml={report}",
                ],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            passed = failed = errors = skipped = 0
            if report.exists():
                root = ET.parse(report).getroot()
                suite = root if root.tag == "testsuite" else root.find("testsuite")
                if suite is not None:
                    total = int(suite.attrib.get("tests", 0))
                    failed = int(suite.attrib.get("failures", 0))
                    errors = int(suite.attrib.get("errors", 0))
                    skipped = int(suite.attrib.get("skipped", 0))
                    passed = max(0, total - failed - errors - skipped)
            total_run = passed + failed + errors
            return EvaluationResult(
                success=completed.returncode == 0 and total_run > 0,
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                pass_rate=passed / total_run if total_run else 0.0,
                exit_code=completed.returncode,
                duration_seconds=time.perf_counter() - started,
                output=(completed.stdout + completed.stderr)[-20_000:],
            )
        except subprocess.TimeoutExpired as error:
            return EvaluationResult(
                success=False,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                pass_rate=0,
                exit_code=124,
                duration_seconds=time.perf_counter() - started,
                output=f"evaluation timeout: {error}",
            )
