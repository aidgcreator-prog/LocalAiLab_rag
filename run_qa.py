"""Run the local QA flow for the repository.

Default flow:
1. Ruff lint
2. Pytest suite
3. Live RAG evaluation harness

Use --skip-rag-eval if the local RAG runtime is not available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _run_step(label: str, command: list[str]) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lint, tests, and RAG evaluation")
    parser.add_argument("--skip-lint", action="store_true", help="Skip ruff")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest")
    parser.add_argument(
        "--skip-rag-eval",
        action="store_true",
        help="Skip live RAG evaluation",
    )
    parser.add_argument(
        "--rag-fail-under",
        type=float,
        default=1.0,
        help="Required pass_rate for the live RAG eval step",
    )
    args = parser.parse_args()

    failures: list[str] = []

    if not args.skip_lint:
        if _run_step("Ruff", [sys.executable, "-m", "ruff", "check", "."]) != 0:
            failures.append("ruff")

    if not args.skip_tests:
        if _run_step("Pytest", [sys.executable, "-m", "pytest"]) != 0:
            failures.append("pytest")

    if not args.skip_rag_eval:
        if _run_step(
            "RAG Eval",
            [
                sys.executable,
                "evals/run_rag_eval.py",
                "--fail-under",
                str(args.rag_fail_under),
            ],
        ) != 0:
            failures.append("rag-eval")

    if failures:
        print(f"\n[ERROR] QA failed: {', '.join(failures)}")
        return 1

    print("\n[OK] QA passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
