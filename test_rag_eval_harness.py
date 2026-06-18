from evals.rag_eval import evaluate_rag_case, parse_rag_output, summarize_rag_results


def test_parse_rag_output_extracts_sources_and_references():
    output = """Retrieved 2 chunk(s) for: _example query_

[R1] asyncio_primitives.txt p.2 § Concurrency
Chunk body here.

[R2] citation_rules.txt § Evidence
More chunk text.

---
References:
  [R1] Example APA reference.  (Smith, 2024, p. 2)
  [R2] Another APA reference.  (Jones, 2023)
"""

    parsed = parse_rag_output(output)

    assert parsed["retrieved_sources"] == [
        "asyncio_primitives.txt",
        "citation_rules.txt",
    ]
    assert parsed["reference_sources"] == [
        "Example APA reference.",
        "Another APA reference.",
    ]


def test_evaluate_rag_case_detects_forbidden_sources_and_missing_hits():
    case = {
        "id": "case-1",
        "query": "example",
        "project": "RAG_EVAL_PRIMARY",
        "expected_sources": ["asyncio_primitives.txt"],
        "forbidden_sources": ["vector_search_noise.txt"],
        "min_references": 1,
        "mode": "Hybrid",
    }
    parsed = {
        "retrieved_sources": ["vector_search_noise.txt"],
        "references": [{"ref_id": "R1", "source": "noise"}],
        "raw_output": "raw",
        "is_error": False,
    }

    result = evaluate_rag_case(case, parsed)

    assert result["passed"] is False
    assert result["expected_hit"] is False
    assert result["forbidden_hit"] is True


def test_summarize_rag_results_computes_rates():
    summary = summarize_rag_results([
        {
            "passed": True,
            "expected_hit": True,
            "forbidden_hit": False,
            "references_ok": True,
            "first_relevant_rank": 1,
        },
        {
            "passed": False,
            "expected_hit": False,
            "forbidden_hit": True,
            "references_ok": False,
            "first_relevant_rank": None,
        },
    ])

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["expected_hit_rate"] == 0.5
    assert summary["project_isolation_rate"] == 0.5
    assert summary["citation_rate"] == 0.5
    assert summary["avg_first_relevant_rank"] == 1.0
