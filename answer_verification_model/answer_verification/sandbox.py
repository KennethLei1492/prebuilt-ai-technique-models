"""Subprocess sandbox for the code execution verifier.

Runs candidate code plus caller-supplied tests in a fresh interpreter with an
isolated environment, a working directory in a temporary folder, no inherited
environment variables and a hard timeout. This is containment against
accidents, not a security boundary: production deployments should run it
inside a container or microVM with no network.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

_FENCE = re.compile(r"```(?P<lang>[\w+-]*)\s*\n(?P<code>.*?)```", re.S)
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    language: str = "python"


def extract_code(answer: str, *, language_hint: str | None = None) -> tuple[str, str]:
    """Return ``(code, language)``: the first fenced block (preferring the hinted language) or the whole text."""
    blocks = list(_FENCE.finditer(answer))
    if blocks:
        if language_hint:
            for m in blocks:
                if m.group("lang").lower() == language_hint.lower():
                    return m.group("code"), m.group("lang").lower()
        m = blocks[0]
        return m.group("code"), (m.group("lang").lower() or language_hint or "python")
    return answer, language_hint or "python"


def fenced_languages(answer: str) -> list[str]:
    """Lower-cased language tags of the fenced blocks in ``answer`` (untagged fences are omitted)."""
    return [m.group("lang").lower() for m in _FENCE.finditer(answer) if m.group("lang")]


def parse_test_summary(output: str) -> tuple[Optional[int], Optional[int]]:
    """Read ``N passed`` / ``M failed`` counters from pytest-like output."""
    p = _PASSED_RE.search(output)
    f = _FAILED_RE.search(output)
    return (int(p.group(1)) if p else None, int(f.group(1)) if f else None)


def run_python(code: str, tests: str, *, timeout_s: float = 30.0, max_output_chars: int = 4000) -> ExecutionResult:
    """Execute ``code`` followed by ``tests`` in an isolated interpreter."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="av_sandbox_") as tmp:
        path = Path(tmp) / "candidate_with_tests.py"
        path.write_text(code.rstrip() + "\n\n# ---- tests ----\n" + tests.strip() + "\n", encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            # Windows needs SYSTEMROOT for the interpreter to start.
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                passed=False,
                exit_code=None,
                stdout=(exc.stdout or "")[:max_output_chars] if isinstance(exc.stdout, str) else "",
                stderr=f"timed out after {timeout_s}s",
                timed_out=True,
                duration_s=time.perf_counter() - t0,
            )
    out, err = proc.stdout[-max_output_chars:], proc.stderr[-max_output_chars:]
    tp, tf = parse_test_summary(out + err)
    return ExecutionResult(
        passed=proc.returncode == 0 and not tf,
        exit_code=proc.returncode,
        stdout=out,
        stderr=err,
        duration_s=time.perf_counter() - t0,
        tests_passed=tp,
        tests_failed=tf,
    )
