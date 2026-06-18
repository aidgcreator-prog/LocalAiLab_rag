"""Live RAG evaluation helpers and report generation.

This module evaluates real retrieval output from the local RAG stack using a
small fixture corpus. It is intentionally lightweight so the harness can be run
from local QA without introducing a second evaluation framework.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "rag_eval_dataset.jsonl"
FIXTURE_DIR = EVALS_DIR / "rag_fixtures"
RESULTS_DIR = EVALS_DIR / "results"

PRIMARY_PROJECT = "RAG_EVAL_PRIMARY"
NOISE_PROJECT = "RAG_EVAL_NOISE"
PRIMARY_THEME = "qa-primary"
NOISE_THEME = "qa-noise"

DEFAULT_SWEEP_CONFIGS: list[dict[str, Any]] = [
    {
        "label": "hybrid-default",
        "mode": "Hybrid",
        "top_k": 5,
        "fetch_k": 40,
        "env": {"RAG_MIN_RERANK_SCORE": "0.0"},
    },
    {
        "label": "hybrid-threshold-0.1",
        "mode": "Hybrid",
        "top_k": 5,
        "fetch_k": 60,
        "env": {"RAG_MIN_RERANK_SCORE": "0.1"},
    },
    {
        "label": "top-k-per-file",
        "mode": "Top-K Per File",
        "top_k": 6,
        "fetch_k": 80,
        "env": {"RAG_MIN_RERANK_SCORE": "0.0"},
    },
    {
        "label": "mmr-diverse",
        "mode": "MMR",
        "top_k": 6,
        "fetch_k": 80,
        "env": {
            "RAG_MIN_RERANK_SCORE": "0.0",
            "RAG_MMR_LAMBDA": "0.45",
        },
    },
]


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def load_rag_eval_cases(dataset_path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["expected_sources"] = _normalize_list(row.get("expected_sources"))
        row["forbidden_sources"] = _normalize_list(row.get("forbidden_sources"))
        row["themes"] = row.get("themes", "")
        row["top_k"] = int(row.get("top_k", 5))
        row["fetch_k"] = int(row.get("fetch_k", 40))
        row["max_files"] = int(row.get("max_files", 5))
        row["min_references"] = int(row.get("min_references", 1))
        row["mode"] = row.get("mode", "Hybrid")
        rows.append(row)
    return rows


def parse_rag_output(output: str) -> dict[str, Any]:
    lines = output.splitlines()
    lowered = output.lower()
    is_warning = lowered.startswith("[warn]")
    is_error = lowered.startswith("[error]")

    retrieved: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    in_references = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "References:":
            in_references = True
            continue
        if line == "---":
            continue

        match = re.match(r"^\[(R\d+)\]\s+(.+)$", line)
        if not match:
            continue

        ref_id = match.group(1)
        remainder = match.group(2).strip()
        if in_references:
            source = remainder.split("  (", 1)[0].strip()
            references.append({"ref_id": ref_id, "source": source, "raw": remainder})
        else:
            source = remainder.split(" p.", 1)[0].split(" § ", 1)[0].strip()
            retrieved.append({"ref_id": ref_id, "source": source, "raw": remainder})

    return {
        "is_warning": is_warning,
        "is_error": is_error,
        "retrieved": retrieved,
        "references": references,
        "retrieved_sources": [item["source"] for item in retrieved],
        "reference_sources": [item["source"] for item in references],
        "raw_output": output,
    }


def evaluate_rag_case(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    retrieved_sources = parsed["retrieved_sources"]
    expected_sources = set(case.get("expected_sources", []))
    forbidden_sources = set(case.get("forbidden_sources", []))

    first_relevant_rank = None
    for index, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            first_relevant_rank = index
            break

    expected_hit = (
        bool(expected_sources.intersection(retrieved_sources))
        if expected_sources
        else True
    )
    forbidden_hit = any(source in forbidden_sources for source in retrieved_sources)
    reference_count = len(parsed["references"])
    references_ok = reference_count >= int(case.get("min_references", 1))
    passed = (
        expected_hit
        and not forbidden_hit
        and references_ok
        and not parsed["is_error"]
    )

    return {
        "id": case["id"],
        "query": case["query"],
        "project": case.get("project", ""),
        "mode": case.get("mode", "Hybrid"),
        "passed": passed,
        "expected_hit": expected_hit,
        "forbidden_hit": forbidden_hit,
        "references_ok": references_ok,
        "first_relevant_rank": first_relevant_rank,
        "reference_count": reference_count,
        "retrieved_sources": retrieved_sources,
        "expected_sources": sorted(expected_sources),
        "forbidden_sources": sorted(forbidden_sources),
        "raw_output": parsed["raw_output"],
    }


def summarize_rag_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for case in case_results if case["passed"])
    expected_hits = sum(1 for case in case_results if case["expected_hit"])
    isolation_passes = sum(1 for case in case_results if not case["forbidden_hit"])
    reference_passes = sum(1 for case in case_results if case["references_ok"])
    relevant_ranks = [
        case["first_relevant_rank"]
        for case in case_results
        if case["first_relevant_rank"] is not None
    ]

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": passed / total if total else 0.0,
        "expected_hit_rate": expected_hits / total if total else 0.0,
        "project_isolation_rate": isolation_passes / total if total else 0.0,
        "citation_rate": reference_passes / total if total else 0.0,
        "avg_first_relevant_rank": (
            (sum(relevant_ranks) / len(relevant_ranks))
            if relevant_ranks
            else None
        ),
    }


def _seed_fixture_corpus() -> list[dict[str, Any]]:
    from ragsub_agent.tools import ingest_rag_paths

    primary_paths = sorted((FIXTURE_DIR / "primary").glob("*.txt"))
    noise_paths = sorted((FIXTURE_DIR / "noise").glob("*.txt"))
    if not primary_paths or not noise_paths:
        raise RuntimeError("RAG evaluation fixtures are missing.")

    return [
        ingest_rag_paths(
            paths=primary_paths,
            project=PRIMARY_PROJECT,
            theme=PRIMARY_THEME,
            chunking_method="recursive",
            chunk_size=500,
            chunk_overlap=50,
        ),
        ingest_rag_paths(
            paths=noise_paths,
            project=NOISE_PROJECT,
            theme=NOISE_THEME,
            chunking_method="recursive",
            chunk_size=500,
            chunk_overlap=50,
        ),
    ]


def _current_rag_settings() -> dict[str, Any]:
    return {
        "embed_model": os.getenv("RAG_EMBED_MODEL", "qwen3-embedding:8b"),
        "cross_encoder_model": os.getenv(
            "RAG_CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        "min_rerank_score": os.getenv("RAG_MIN_RERANK_SCORE", "0.0"),
        "mmr_lambda": os.getenv("RAG_MMR_LAMBDA", "0.5"),
        "max_context_tokens": os.getenv("RAG_MAX_CONTEXT_TOKENS", "6000"),
        "ollama_base_url": os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
    }


def _report_rank_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    avg_rank = summary.get("avg_first_relevant_rank")
    avg_rank_score = 0.0 if avg_rank is None else -float(avg_rank)
    return (
        float(summary.get("pass_rate", 0.0)),
        float(summary.get("expected_hit_rate", 0.0)),
        float(summary.get("citation_rate", 0.0)),
        avg_rank_score,
    )


def run_live_rag_eval(
    dataset_path: Path = DATASET_PATH,
    *,
    top_k_override: int | None = None,
    mode_override: str | None = None,
    fetch_k_override: int | None = None,
) -> dict[str, Any]:
    from ragsub_agent.tools import get_last_rag_query_diagnostics, rag_retrieve

    cases = load_rag_eval_cases(dataset_path)
    ingest_results = _seed_fixture_corpus()

    case_results: list[dict[str, Any]] = []
    for case in cases:
        tool_input = {
            "query": case["query"],
            "project": case.get("project", ""),
            "themes": case.get("themes", ""),
            "top_k": top_k_override if top_k_override is not None else case["top_k"],
            "mode": mode_override or case["mode"],
            "fetch_k": (
                fetch_k_override
                if fetch_k_override is not None
                else case["fetch_k"]
            ),
            "max_files": case["max_files"],
        }
        output = rag_retrieve.invoke(tool_input)
        parsed = parse_rag_output(output)
        evaluated = evaluate_rag_case(case, parsed)
        evaluated["diagnostics"] = get_last_rag_query_diagnostics()
        case_results.append(evaluated)

    summary = summarize_rag_results(case_results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "fixture_projects": {
            "primary": PRIMARY_PROJECT,
            "noise": NOISE_PROJECT,
        },
        "settings": _current_rag_settings(),
        "ingest_results": ingest_results,
        "summary": summary,
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"rag-eval-{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def run_rag_eval_sweep(
    dataset_path: Path = DATASET_PATH,
    sweep_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    experiments = []
    configs = sweep_configs or DEFAULT_SWEEP_CONFIGS

    for config in configs:
        env_overrides = config.get("env", {})
        previous_env = {key: os.environ.get(key) for key in env_overrides}
        try:
            for key, value in env_overrides.items():
                os.environ[key] = str(value)
            report = run_live_rag_eval(
                dataset_path=dataset_path,
                top_k_override=config.get("top_k"),
                mode_override=config.get("mode"),
                fetch_k_override=config.get("fetch_k"),
            )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        experiments.append(
            {
                "label": config["label"],
                "mode": config.get("mode"),
                "top_k": config.get("top_k"),
                "fetch_k": config.get("fetch_k"),
                "env": env_overrides,
                "summary": report["summary"],
                "report_path": report["output_path"],
                "settings": report["settings"],
            }
        )

    best = max(experiments, key=lambda item: _report_rank_key(item["summary"]))
    sweep_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "experiments": experiments,
        "best": best,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"rag-eval-sweep-{stamp}.json"
    output_path.write_text(json.dumps(sweep_report, indent=2), encoding="utf-8")
    sweep_report["output_path"] = str(output_path)
    return sweep_report


def format_rag_eval_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    settings = report["settings"]
    return "\n".join([
        "RAG evaluation complete",
        (
            f"  pass_rate: {summary['pass_rate']:.3f} "
            f"({summary['passed_cases']}/{summary['total_cases']})"
        ),
        f"  expected_hit_rate: {summary['expected_hit_rate']:.3f}",
        f"  project_isolation_rate: {summary['project_isolation_rate']:.3f}",
        f"  citation_rate: {summary['citation_rate']:.3f}",
        f"  avg_first_relevant_rank: {summary['avg_first_relevant_rank']}",
        f"  embed_model: {settings['embed_model']}",
        f"  min_rerank_score: {settings['min_rerank_score']}",
        f"  report: {report['output_path']}",
    ])


def format_rag_eval_sweep_summary(report: dict[str, Any]) -> str:
    lines = ["RAG evaluation sweep complete"]
    for experiment in report["experiments"]:
        summary = experiment["summary"]
        lines.append(
            (
                f"  {experiment['label']}: pass_rate={summary['pass_rate']:.3f} "
                f"expected_hit_rate={summary['expected_hit_rate']:.3f} "
                f"citation_rate={summary['citation_rate']:.3f} "
                f"avg_first_relevant_rank={summary['avg_first_relevant_rank']}"
            )
        )
    best = report["best"]
    lines.append(
        (
            f"  best: {best['label']} mode={best['mode']} top_k={best['top_k']} "
            f"fetch_k={best['fetch_k']} env={best['env']}"
        )
    )
    lines.append(f"  report: {report['output_path']}")
    return "\n".join(lines)
