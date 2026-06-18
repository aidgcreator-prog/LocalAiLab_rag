"""Run the live RAG evaluation harness.

Usage:
    python evals/run_rag_eval.py
    python evals/run_rag_eval.py --fail-under 1.0
    python evals/run_rag_eval.py --mode "Top-K Globally" --top-k 6 --fetch-k 60
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv  # noqa: E402

from evals.rag_eval import (  # noqa: E402
    DATASET_PATH,
    format_rag_eval_sweep_summary,
    format_rag_eval_summary,
    run_rag_eval_sweep,
    run_live_rag_eval,
)

load_dotenv(PROJECT_DIR / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the live RAG evaluation harness"
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the RAG eval dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k for every case",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="Override fetch_k for every case",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override retrieval mode for every case",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit non-zero if pass_rate falls below this value",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="Temporarily override RAG_MIN_RERANK_SCORE for this run",
    )
    parser.add_argument(
        "--sweep-defaults",
        action="store_true",
        help="Run the built-in retrieval sweep and report the best config",
    )
    args = parser.parse_args()

    previous_score = os.environ.get("RAG_MIN_RERANK_SCORE")
    if args.min_rerank_score is not None:
        os.environ["RAG_MIN_RERANK_SCORE"] = str(args.min_rerank_score)

    try:
        if args.sweep_defaults:
            report = run_rag_eval_sweep(dataset_path=Path(args.dataset))
            print(format_rag_eval_sweep_summary(report))
            return 0

        report = run_live_rag_eval(
            dataset_path=Path(args.dataset),
            top_k_override=args.top_k,
            mode_override=args.mode,
            fetch_k_override=args.fetch_k,
        )
    except Exception as exc:
        print(f"[ERROR] RAG evaluation failed: {exc}")
        return 1
    finally:
        if args.min_rerank_score is None:
            pass
        elif previous_score is None:
            os.environ.pop("RAG_MIN_RERANK_SCORE", None)
        else:
            os.environ["RAG_MIN_RERANK_SCORE"] = previous_score

    print(format_rag_eval_summary(report))
    if report["summary"]["pass_rate"] < args.fail_under:
        print(
            (
                f"[ERROR] pass_rate {report['summary']['pass_rate']:.3f} "
                f"is below fail-under {args.fail_under:.3f}"
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
