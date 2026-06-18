# RAG Remediation Plan

This plan turns the current RAG review into a staged remediation track with concrete implementation steps, acceptance criteria, and validation hooks.

## P0: Correctness And Isolation

Goal: eliminate cross-project leakage, stale-chunk collisions, and reporting errors before tuning retrieval quality.

### 1. Make project a required boundary for reporting and admin paths

Files:
- `literature_review/tools.py`
- `ragsub_agent/tools.py`
- `streamlit_app.py`

Implementation steps:
1. Pass `project` explicitly when reading collections for report generation.
2. Add project-aware index summary helpers so UI and maintenance actions do not aggregate unrelated collections by default.
3. Add project-aware deletion helpers instead of global source/theme deletion.
4. Surface the active project in the UI and in any destructive action confirmation text.

Acceptance criteria:
- A report generated for a non-default project only uses chunks from that project.
- Listing indexed files for `project=A` never includes files from `project=B`.
- Deleting a source in `project=A` does not remove the same source from `project=B`.

### 2. Fix chunk identity and re-ingestion semantics

Files:
- `ragsub_agent/tools.py`

Implementation steps:
1. Change chunk IDs to include project, theme, source fingerprint, and chunk fingerprint.
2. Replace `delete(where={"source": path.name})` with a narrower stale-chunk cleanup strategy keyed by project plus source plus theme.
3. Add tests for same filename across themes and across projects.

Acceptance criteria:
- Re-ingesting the same file into a different theme does not erase the first copy.
- Re-ingesting an updated file replaces only its own stale chunks.

### 3. Add regression tests for project isolation

Files:
- `test_rag_output_contract.py`
- `test_rag_eval_harness.py`

Implementation steps:
1. Lock down the `rag_retrieve` output format used by the evaluation harness.
2. Add unit coverage for project-filtered case scoring and forbidden-source detection.

Acceptance criteria:
- Output parsing remains stable across future prompt or formatting changes.
- Project-isolation regressions fail fast in local QA.

## P1: Retrieval Quality And Empirical Tuning

Goal: improve answer quality once boundaries are safe.

### 4. Replace file selection by chunk count with file selection by relevance

Files:
- `ragsub_agent/tools.py`

Implementation steps:
1. In `Top-K Per File` and `MMR` modes, compute a per-file relevance score from the best rerank score or fused dense+lexical score.
2. Select files by relevance, then rerank within those files.
3. Keep `max_files` as a diversity cap rather than a chunk-count proxy.

Acceptance criteria:
- Longer documents no longer dominate retrieval solely because they produce more chunks.
- RAG eval reports improved first relevant rank on synthesis-oriented cases.

### 5. Improve token-budget handling

Files:
- `ragsub_agent/tools.py`

Implementation steps:
1. Change token-budget truncation to skip oversized chunks instead of breaking the loop.
2. Record skipped chunk IDs and sizes in debug telemetry.
3. Add a small test for the truncation policy.

Acceptance criteria:
- One oversized chunk does not suppress later relevant chunks that fit in the budget.

### 6. Tune thresholds and retrieval modes with the new live eval harness

Files:
- `evals/rag_eval_dataset.jsonl`
- `evals/run_rag_eval.py`
- `evals/rag_eval.py`

Implementation steps:
1. Run the live fixture corpus with current settings.
2. Sweep `RAG_MIN_RERANK_SCORE`, `top_k`, `fetch_k`, and `mode` across a few runs.
3. Compare pass rate, hit rate, citation rate, and first relevant rank.
4. Promote the best configuration into `.env.example` and docs.

Acceptance criteria:
- Retrieval settings are chosen from observed harness output, not intuition.
- A baseline report is saved under `evals/results/` for future comparison.

## P2: Observability And Workflow Alignment

Goal: make failures diagnosable and keep prompts aligned with actual tooling.

### 7. Add query-time telemetry for retrieval diagnostics

Files:
- `ragsub_agent/tools.py`
- optional UI surfaces in `streamlit_app.py`

Implementation steps:
1. Log candidate count, rerank count, filtered count, skipped-for-budget count, and final chunk IDs.
2. Include tuning parameters in eval reports.
3. Optionally surface the latest retrieval diagnostics in the Streamlit RAG section.

Acceptance criteria:
- When a query misses expected evidence, you can tell whether the loss happened at retrieval, rerank, thresholding, or truncation.

### 8. Align the prompt with the real standalone RAG tool surface

Files:
- `ragsub_agent/prompts.py`
- `ragsub_agent/agent.py`

Implementation steps:
1. Remove `write_todos` from the standalone RAG prompt, or add the tool if you truly want that behavior.
2. Keep the standalone RAG agent prompt focused on tools it actually owns.

Acceptance criteria:
- The standalone RAG agent never asks for a missing tool.

## Suggested Execution Order

1. P0.1 project scoping
2. P0.2 chunk identity
3. P0.3 regression tests
4. P1.4 relevance-based file selection
5. P1.5 token-budget fix
6. P1.6 empirical tuning with the harness
7. P2 telemetry and prompt alignment

## Validation Checklist

- `python run_qa.py`
- `python evals/run_rag_eval.py --fail-under 1.0`
- manual spot-check in Streamlit for project-scoped retrieval and deletion
