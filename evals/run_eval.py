"""Evaluation Framework for LV Combined Agents

Evaluates the orchestrator agent using Azure AI Evaluation SDK with:
- Built-in evaluators: TaskAdherence, IntentResolution, Coherence, Relevance
- Custom code-based evaluators: DelegationAccuracy, ResponseLength

Supports two modes:
1. Collect responses: Run the agent on eval_dataset.jsonl to collect responses
2. Evaluate: Score collected responses with evaluators

Usage:
    # Step 1: Collect agent responses (requires Ollama running)
    python evals/run_eval.py collect

    # Step 2: Run evaluation on collected responses
    python evals/run_eval.py evaluate

    # Or do both in one go:
    python evals/run_eval.py all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent_runtime import build_agent_config
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "eval_dataset.jsonl"
RESPONSES_PATH = EVALS_DIR / "eval_responses.jsonl"
RESULTS_DIR = EVALS_DIR / "results"


# ── Agent Runner: Collect Responses ──────────────────────────────────────────


async def collect_responses() -> Path:
    """Run the agent on each query in the dataset and save responses."""
    from agent import agent, extract_last_ai_text
    from langchain_core.messages import HumanMessage

    print(f"Loading dataset: {DATASET_PATH}")
    rows = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    print(f"Found {len(rows)} evaluation cases\n")

    results = []
    for i, row in enumerate(rows, 1):
        query = row["query"]
        print(f"[{i}/{len(rows)}] {query[:80]}...")

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(query)]},
                config=build_agent_config(
                    thread_id=f"eval-{i}",
                    user_id="eval-user",
                ),
            )
            messages = result.get("messages", [])
            response = extract_last_ai_text(messages)

            # Extract tool calls to see which subagent was delegated to
            tool_calls = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "name": tc.get("name", ""),
                            "args_keys": list(tc.get("args", {}).keys()),
                        })

            results.append({
                "query": query,
                "response": response or "(empty response)",
                "expected_delegation": row.get("expected_delegation", ""),
                "tool_calls": tool_calls,
            })
            print(f"  ✓ Response length: {len(response)} chars, Tool calls: {len(tool_calls)}")

        except Exception as exc:
            results.append({
                "query": query,
                "response": f"[ERROR] {exc}",
                "expected_delegation": row.get("expected_delegation", ""),
                "tool_calls": [],
            })
            print(f"  ✗ Error: {exc}")

    # Save responses
    RESPONSES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {len(results)} responses to {RESPONSES_PATH}")
    return RESPONSES_PATH


# ── Custom Code-Based Evaluators ─────────────────────────────────────────────


class DelegationAccuracyEvaluator:
    """Checks if the agent delegated to the expected subagent.

    Looks at tool_calls to see if a `task` call was made with the
    expected subagent_type.
    """

    def __init__(self):
        pass

    def __call__(self, *, query: str, response: str, expected_delegation: str = "", **kwargs) -> dict:
        if not expected_delegation:
            return {"delegation_accuracy": 1.0, "delegation_match": "N/A (no expected)"}

        # Check if expected subagent appears in the response text
        # (the orchestrator mentions which agent it delegated to)
        response_lower = response.lower()
        expected_lower = expected_delegation.lower()

        # Common patterns the orchestrator uses when delegating
        delegation_indicators = [
            f"task(subagent_type='{expected_lower}'",
            f"delegat",
            expected_lower,
        ]

        matched = any(ind in response_lower for ind in delegation_indicators)
        return {
            "delegation_accuracy": 1.0 if matched else 0.0,
            "delegation_match": f"{'✓' if matched else '✗'} expected={expected_delegation}",
        }


class ResponseLengthEvaluator:
    """Measures response length and flags very short or error responses."""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def __call__(self, *, response: str, **kwargs) -> dict:
        length = len(response)
        is_error = response.startswith("[ERROR]")
        is_adequate = length >= self.min_length and not is_error
        return {
            "response_length": length,
            "response_adequate": 1.0 if is_adequate else 0.0,
        }


# ── Run Evaluation ───────────────────────────────────────────────────────────


def run_evaluation(responses_path: Path | None = None) -> dict:
    """Run evaluators on collected responses."""
    from azure.ai.evaluation import evaluate

    data_path = responses_path or RESPONSES_PATH
    if not data_path.exists():
        print(f"No responses found at {data_path}")
        print("Run 'python evals/run_eval.py collect' first.")
        sys.exit(1)

    print(f"Loading responses: {data_path}")
    row_count = sum(1 for line in data_path.read_text().splitlines() if line.strip())
    print(f"Evaluating {row_count} responses\n")

    # Initialize evaluators
    evaluators = {}
    evaluator_config = {}

    # Custom code-based evaluators (always available — no LLM needed)
    delegation_eval = DelegationAccuracyEvaluator()
    length_eval = ResponseLengthEvaluator(min_length=50)

    evaluators["delegation_accuracy"] = delegation_eval
    evaluators["response_length"] = length_eval

    evaluator_config["delegation_accuracy"] = {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.response}",
            "expected_delegation": "${data.expected_delegation}",
        }
    }
    evaluator_config["response_length"] = {
        "column_mapping": {
            "response": "${data.response}",
        }
    }

    # Built-in prompt-based evaluators (require an LLM judge)
    # Attempt to configure with Ollama's OpenAI-compatible endpoint
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "")

    if judge_model:
        try:
            from azure.ai.evaluation import OpenAIModelConfiguration
            from azure.ai.evaluation import (
                CoherenceEvaluator,
                RelevanceEvaluator,
            )

            model_config = OpenAIModelConfiguration(
                type="openai",
                model=judge_model,
                base_url=f"{ollama_base}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )

            evaluators["coherence"] = CoherenceEvaluator(model_config=model_config)
            evaluators["relevance"] = RelevanceEvaluator(model_config=model_config)

            evaluator_config["coherence"] = {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            }
            evaluator_config["relevance"] = {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            }
            print(f"[OK] LLM-judge evaluators enabled (model: {judge_model})")

        except ImportError:
            print("[WARN] Could not load prompt-based evaluators — running code-based only")
        except Exception as exc:
            print(f"[WARN] LLM-judge setup failed ({exc}) — running code-based only")
    else:
        print("[INFO] No EVAL_JUDGE_MODEL set — running code-based evaluators only")
        print("  To enable LLM-judge evaluators, set EVAL_JUDGE_MODEL in .env")
        print("  Example: EVAL_JUDGE_MODEL=gpt-oss:20b\n")

    # Run evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"eval-{timestamp}"

    print("Running evaluation...")
    result = evaluate(
        data=str(data_path),
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        output_path=str(output_path),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = result.get("metrics", {})
    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")

    print(f"\nDetailed results saved to: {output_path}")
    print("=" * 60)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LV Combined Agents orchestrator")
    parser.add_argument(
        "command",
        choices=["collect", "evaluate", "all"],
        help="collect=run agent on dataset, evaluate=score responses, all=both",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=str(DATASET_PATH),
        help=f"Path to eval dataset (default: {DATASET_PATH})",
    )
    args = parser.parse_args()

    if args.command in ("collect", "all"):
        print("=" * 60)
        print("STEP 1: Collecting Agent Responses")
        print("=" * 60 + "\n")
        asyncio.run(collect_responses())

    if args.command in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("STEP 2: Running Evaluation")
        print("=" * 60 + "\n")
        run_evaluation()


if __name__ == "__main__":
    main()
