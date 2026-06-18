"""LangChain @tool wrappers for the literature review pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@tool(parse_docstring=True)
def run_literature_review(
    query: str,
    project: str = "Literature Review",
    theme: str = "Literature Review",
    max_papers: int = 20,
) -> str:
    """Run a fully automated academic literature review pipeline.

    Searches Semantic Scholar + OpenAlex, downloads legal open-access PDFs,
    falls back to abstract text for paywalled papers, and ingests everything
    into the RAG store under ``project``/``theme`` for subsequent retrieval
    and synthesis.

    Args:
        query: Research question or topic to search for
            (e.g., 'deep learning for drug discovery').
        project: Project name used as the RAG partition for all ingested papers.
            Use the same name later when calling rag_retrieve to scope results.
        theme: Sub-category theme label within the project
            (default 'Literature Review').
        max_papers: Maximum number of papers to process. Range: 5-50.

    Returns:
        Pipeline summary report including paper counts,
        acquisition methods, and ingestion stats.
    """
    from literature_review.pipeline import run_literature_search

    max_papers = max(5, min(50, max_papers))

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nested event loop — use a thread to run async work
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, run_literature_search(
                    query=query,
                    project=project,
                    theme=theme,
                    max_papers=max_papers,
                ))
                result = future.result(timeout=300)
        else:
            result = loop.run_until_complete(
                run_literature_search(
                    query=query,
                    project=project,
                    theme=theme,
                    max_papers=max_papers,
                )
            )
    except Exception as e:
        return f"[ERROR] Literature review pipeline failed: {e}"

    lines = [
        "=== Literature Review Pipeline Complete ===",
        f"Query:              {result['query']}",
        f"Project:            {result['project']}",
        f"Theme:              {result['theme']}",
        "",
        f"Papers found:       {result['papers_found']}",
        f"Full-text PDFs:     {result['papers_downloaded']}",
        f"Abstract-only:      {result['papers_abstract_only']}",
        f"Skipped (no data):  {result['papers_skipped']}",
        f"RAG chunks added:   {result['ingested_chunks']}",
    ]
    if result["failures"]:
        lines.append("")
        lines.append(f"Failures ({len(result['failures'])}):")
        lines.extend([f"  - {f}" for f in result["failures"][:10]])

    lines += [
        "",
        "Next steps:",
        (
            f"  1. Use rag_retrieve with project='{result['project']}' "
            "to query the ingested papers."
        ),
        "  2. Use generate_literature_report to produce a .docx report.",
    ]
    return "\n".join(lines)


@tool(parse_docstring=True)
def generate_literature_report(
    project: str,
    query: str,
    synthesis_text: str,
    output_filename: str = "",
) -> str:
    """Generate a formatted Word (.docx) literature review report.

    Should be called after run_literature_review has completed ingestion and
    the RAG pipeline has been used to produce a synthesis (passed as synthesis_text).

    Args:
        project: Project name (must match the one used in run_literature_review).
        query: Original search query (used in the report header and abstract).
        synthesis_text: Agent-generated synthesis text in Markdown format with
            section headings (## Findings, ## Consensus, ## Debates,
            ## Gaps, ## Conclusion).
        output_filename: Output .docx filename (without path). If empty, auto-generated
            from project name.

    Returns:
        Path to the generated .docx file and a brief summary.
    """
    from literature_review.report_generator import generate_docx_report
    from ragsub_agent.tools import get_vector_collection

    # Retrieve paper records from RAG metadata for the project
    try:
        collection = get_vector_collection(project)
        results = collection.get(include=["metadatas"])
        metas = results.get("metadatas") or []
    except Exception:
        metas = []

    # Build deduplicated paper records from RAG metadata
    seen_dois: dict[str, dict[str, Any]] = {}
    seen_sources: set[str] = set()
    for meta in metas:
        doi = meta.get("doi", "")
        source = meta.get("source", "")
        key = doi if doi else source
        if key and key not in seen_sources and key not in seen_dois:
            record = {
                "title": meta.get("doc_title") or source,
                "authors": [
                    a.strip()
                    for a in (meta.get("doc_authors") or "").split(";")
                    if a.strip()
                ],
                "year": meta.get("doc_year") or meta.get("pub_date", ""),
                "doi": doi,
                "journal": meta.get("journal", ""),
                "abstract": meta.get("paper_abstract", ""),
                "citation_count": int(meta.get("citation_count") or 0),
                "method": meta.get("pdf_method", ""),
            }
            if doi:
                seen_dois[doi] = record
            else:
                seen_sources.add(source)
                seen_dois[f"_src_{source}"] = record

    paper_records = list(seen_dois.values())

    # Determine output path
    if not output_filename:
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in project
        )[:50]
        output_filename = (
            f"{safe_name.strip().replace(' ', '_')}_literature_review.docx"
        )

    output_path = _PROJECT_ROOT / "literature_reports" / output_filename

    try:
        path = generate_docx_report(
            project=project,
            query=query,
            synthesis_text=synthesis_text,
            paper_records=paper_records,
            output_path=output_path,
        )
        return (
            f"[OK] Report generated: {path}\n"
            f"Papers included: {len(paper_records)}\n"
            f"File size: {path.stat().st_size:,} bytes"
        )
    except Exception as e:
        return f"[ERROR] Report generation failed: {e}"


# ── Tool registration list ────────────────────────────────────────────────────

LITERATURE_REVIEW_TOOLS = [run_literature_review, generate_literature_report]
