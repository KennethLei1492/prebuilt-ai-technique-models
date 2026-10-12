"""Repository quality gate.

Runs, from the repository root and fully offline:

1. the complete pytest suite;
2. every model package's CLI in ``--offline`` mode with a representative task;
3. every example script in each model folder and the enterprise pipeline example;
4. static checks from docs/ARCHITECTURE.md section 6: no ``temperature``/``top_p``/
   ``top_k``/``budget_tokens`` in model code, no date-suffixed Claude model IDs,
   and no ``datetime.now``/``uuid4`` inside prompt modules.

Exit status is non-zero if anything fails. Use it in CI and before releases::

    python tools/quality_gate.py            # everything
    python tools/quality_gate.py --fast     # tests + static checks only
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_FOLDERS = {
    "hybrid_retrieval_model": "hybrid_retrieval",
    "task_decomposition_model": "task_decomposition",
    "answer_verification_model": "answer_verification",
    "parallelization_model": "parallelization",
    "caching_model": "caching",
    "pipeline_rag_model": "pipeline_rag",
}

CLI_TASKS = {
    "hybrid_retrieval": "Find the clauses that specify the pump impeller material and its corrosion allowance",
    "task_decomposition": "Review the pump station design: check hydraulic sizing, verify the motor rating against the duty point, and list open risks",
    "answer_verification": "Verify this answer: the required NPSH margin is 1.5 m and the selected pump provides 2.1 m at the duty point",
    "parallelization": "Summarise each of the following five test logs independently and then combine the findings",
    "caching": "Explain the difference between BM25 and dense retrieval",
    "pipeline_rag": "What corrosion allowance does the specification require for the pump casing?",
}

# Sampling parameters are only a problem when they are sent to the model. The models never call the
# SDK directly (everything goes through pipeline_core), so flag them only on lines that build an
# LLM request; "top_k" is also a legitimate retrieval/semantic-cache parameter.
LLM_CALL_CONTEXT = re.compile(r"anthropic|\.create\(|kwargs|extra_body|LLMRequest|output_config|messages=")
FORBIDDEN_PATTERNS = {
    r"\btemperature\s*[=:]": "sampling parameter rejected by current Claude models",
    r"\btop_p\s*[=:]": "sampling parameter rejected by current Claude models",
    r"\btop_k\s*[=:]": "sampling parameter rejected by current Claude models",
    r"budget_tokens": "removed thinking parameter",
    r"claude-[a-z]+-\d(?:-\d)?-20\d{6}": "date-suffixed model id",
}
CONTEXT_REQUIRED = {r"\btemperature\s*[=:]", r"\btop_p\s*[=:]", r"\btop_k\s*[=:]"}
PROMPT_FILE_PATTERNS = {
    r"datetime\.now\(": "timestamp in prompt module breaks prompt-cache prefix stability",
    r"uuid4\(": "random id in prompt module breaks prompt-cache prefix stability",
}


def pythonpath() -> str:
    parts = ["common", *MODEL_FOLDERS.keys()]
    return os.pathsep.join(str(ROOT / p) for p in parts)


def run(cmd: list[str], *, cwd: Path = ROOT, env_extra: dict[str, str] | None = None, timeout: int = 900) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PIPELINE_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def check_static() -> list[str]:
    problems: list[str] = []
    for folder, pkg in MODEL_FOLDERS.items():
        pkg_dir = ROOT / folder / pkg
        if not pkg_dir.exists():
            problems.append(f"{folder}: package directory {pkg} missing")
            continue
        for py in pkg_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            for pat, why in FORBIDDEN_PATTERNS.items():
                for m in re.finditer(pat, text):
                    line = text.count("\n", 0, m.start()) + 1
                    snippet = lines[line - 1].strip()
                    # allow mentions inside comments/docstrings that explain the rule
                    if snippet.startswith("#") or "rejected" in snippet or "never" in snippet.lower() or "not supported" in snippet.lower():
                        continue
                    if pat in CONTEXT_REQUIRED:
                        window = "\n".join(lines[max(0, line - 4) : line + 3])
                        if not LLM_CALL_CONTEXT.search(window):
                            continue  # retrieval top_k, cache top_k, thermal temperature, ...
                    problems.append(f"{py.relative_to(ROOT)}:{line}: {why} ({snippet[:80]})")
            if py.name == "prompts.py":
                for pat, why in PROMPT_FILE_PATTERNS.items():
                    for m in re.finditer(pat, text):
                        line = text.count("\n", 0, m.start()) + 1
                        problems.append(f"{py.relative_to(ROOT)}:{line}: {why}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip CLI and example runs")
    args = ap.parse_args()
    failures: list[str] = []

    print("== pytest")
    code, out = run([sys.executable, "-m", "pytest", "-q"])
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    if code != 0:
        failures.append("pytest failed")
        print(out[-4000:])

    print("== static checks")
    problems = check_static()
    for p in problems:
        print("  ", p)
    if problems:
        failures.append(f"{len(problems)} static check problem(s)")
    else:
        print("   ok")

    if not args.fast:
        print("== CLIs (offline)")
        for folder, pkg in MODEL_FOLDERS.items():
            if not (ROOT / folder / pkg).exists():
                print(f"   {pkg}: missing")
                failures.append(f"{pkg} package missing")
                continue
            code, out = run([sys.executable, "-m", pkg, "--offline", CLI_TASKS[pkg]])
            status = "ok" if code == 0 else f"exit {code}"
            print(f"   {pkg}: {status}")
            if code != 0:
                failures.append(f"CLI {pkg} failed")
                print(out[-2000:])

        print("== examples (offline)")
        flat = sorted(p for f in MODEL_FOLDERS for p in (ROOT / f / "examples").glob("*.py"))
        ep = ROOT / "examples" / "enterprise_pipeline.py"
        if ep.exists():
            flat.append(ep)
        for ex in flat:
            extra = ["--offline"] if ex.name == "enterprise_pipeline.py" else []
            code, out = run([sys.executable, str(ex), *extra])
            print(f"   {ex.relative_to(ROOT)}: {'ok' if code == 0 else f'exit {code}'}")
            if code != 0:
                failures.append(f"example {ex.relative_to(ROOT)} failed")
                print(out[-2000:])

    print("== result")
    if failures:
        for f in failures:
            print("   FAIL:", f)
        return 1
    print("   all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
