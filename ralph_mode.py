"""Ralph Mode — Autonomous agent loop.

Inspired by the DeepAgents ralph_mode example, this module lets the
orchestrator run autonomously in a loop: receive a goal, work toward
it across multiple iterations, and optionally check back in with the
user when it needs input.

Usage:
    python ralph_mode.py "Build a FastAPI CRUD app for a todo list"
    python ralph_mode.py --iterations 5 "Refactor the data pipeline"
    python ralph_mode.py --verbose "Research LangGraph best practices and write a summary"

The loop:
    1. Send the goal (or follow-up) to the orchestrator
    2. Print the response
    3. If the agent says DONE or max iterations reached → stop
    4. Otherwise generate a continuation prompt and loop
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from agent_runtime import build_agent_config

load_dotenv(Path(__file__).parent / ".env")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MAX_ITERATIONS = 10
DONE_MARKERS = {"[DONE]", "DONE", "[COMPLETE]", "COMPLETE"}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _looks_done(text: str) -> bool:
    """Heuristic: does the agent think it's finished?"""
    upper = text.strip().upper()
    for marker in DONE_MARKERS:
        if marker in upper:
            return True
    # Also treat very short "I'm done" style messages as complete
    if len(text.strip()) < 60 and any(w in upper for w in ("FINISHED", "NOTHING LEFT", "ALL DONE")):
        return True
    return False


def _continuation_prompt(iteration: int, previous_response: str) -> str:
    """Generate the follow-up prompt for the next loop iteration."""
    return (
        f"Continue working on the goal. This is iteration {iteration}. "
        "If you have completed all tasks, respond with [DONE]. "
        "Otherwise, pick up where you left off and make progress."
    )


# ── Main loop ─────────────────────────────────────────────────────────────────


async def run_ralph_mode(
    goal: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
) -> list[str]:
    """Run the orchestrator in autonomous loop mode.

    Returns the list of agent responses across all iterations.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    def _print_header(text: str) -> None:
        if use_rich:
            console.print(Panel(text, style="bold cyan"))
        else:
            print(f"\n{'='*60}\n{text}\n{'='*60}")

    def _print_response(text: str, iteration: int) -> None:
        if use_rich:
            console.print(Panel(text, title=f"Iteration {iteration}", border_style="green"))
        else:
            print(f"\n--- Iteration {iteration} ---\n{text}\n")

    # Lazy-load agent to avoid heavy init on import
    from agent import agent
    from langchain_core.messages import HumanMessage

    _print_header(f"Ralph Mode — Goal: {goal}")
    if verbose:
        print(f"Max iterations: {max_iterations}")

    responses: list[str] = []
    current_prompt = goal

    for i in range(1, max_iterations + 1):
        if verbose:
            print(f"\n[ralph] Iteration {i}/{max_iterations} — sending prompt...")

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(current_prompt)]},
                config=build_agent_config(
                    thread_id="ralph-mode-thread",
                    user_id="ralph-mode-user",
                ),
            )
            last = result["messages"][-1]
            text = last.content if isinstance(last.content, str) else str(last.content)
        except Exception as exc:
            text = f"[ERROR] Agent raised: {exc}"

        responses.append(text)
        _print_response(text, i)

        if _looks_done(text):
            _print_header("Agent signalled DONE — stopping.")
            break

        if i == max_iterations:
            _print_header(f"Reached max iterations ({max_iterations}) — stopping.")
            break

        current_prompt = _continuation_prompt(i + 1, text)

    return responses


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the orchestrator in autonomous Ralph Mode",
    )
    parser.add_argument("goal", help="The high-level goal for the agent to pursue")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra debug info",
    )
    args = parser.parse_args()

    asyncio.run(run_ralph_mode(args.goal, max_iterations=args.iterations, verbose=args.verbose))


if __name__ == "__main__":
    main()
