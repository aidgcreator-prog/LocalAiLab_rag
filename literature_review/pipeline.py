"""Full literature review pipeline.

Orchestrates: search → filter/rank → resolve → download/fallback → (confirm) → ingest → report
"""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from literature_review.pdf_resolver import (
    build_abstract_document,
    download_pdf,
    playwright_pdf_download,
    resolve_pdf_url,
)
from literature_review.search_apis import PaperRecord, search_papers

_DOWNLOAD_DIR = _PROJECT_ROOT / "temp_litreview"

_POLITE_DELAY = 0.5  # seconds between concurrent paper-level ops

# Type alias for progress callbacks: (step, current_index, total, detail_message)
ProgressCallback = Callable[[str, int, int, str], None]


async def search_papers_only(
    query: str,
    max_papers: int = 20,
    on_progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Search academic APIs and return paper metadata WITHOUT downloading.

    Returns ``paper_records`` list with titles, DOIs, abstracts etc.
    The caller should present these to the user and let them select which
    to download before calling ``download_selected_papers``.
    """

    def _progress(step: str, idx: int, total: int, msg: str) -> None:
        if on_progress:
            on_progress(step, idx, total, msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    _progress("search", 0, 1, f"Searching APIs for up to {max_papers} papers…")
    papers = await search_papers(query, max_results=max_papers)
    _progress("search", 1, 1, f"Found {len(papers)} papers")

    if _stopped():
        return {"stopped": True, "papers_found": len(papers), "paper_records": [], "query": query}

    paper_records: list[dict[str, Any]] = []
    for paper in papers:
        paper_records.append({
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "journal": paper.journal,
            "abstract": paper.abstract,
            "citation_count": paper.citation_count,
            "open_access": paper.open_access,
            "pdf_url": paper.pdf_url or "",   # may already have a direct URL
            "source_api": paper.source_api or "",
            "paper_id": getattr(paper, "paper_id", "") or "",
            "openalex_id": getattr(paper, "openalex_id", "") or "",
            "url": paper.url or "",            # OA publisher landing page
            "extra": paper.extra or {},        # contains ArXiv/PMC IDs
            "method": "",
            "local_path": None,
        })

    return {
        "stopped": _stopped(),
        "papers_found": len(papers),
        "paper_records": paper_records,
        "query": query,
    }


async def download_selected_papers(
    paper_records: list[dict[str, Any]],
    download_dir: Path = _DOWNLOAD_DIR,
    on_progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Download PDFs (or build abstract docs) for a list of paper records.

    Returns paths_to_ingest, extra_metas, and updated paper_records with
    local_path and method fields populated.
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    def _progress(step: str, idx: int, total: int, msg: str) -> None:
        if on_progress:
            on_progress(step, idx, total, msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    papers_downloaded = 0
    papers_abstract_only = 0
    papers_skipped = 0
    failures: list[str] = []
    paths_to_ingest: list[Path] = []
    extra_metas: list[dict[str, str]] = []
    updated_records: list[dict[str, Any]] = []
    total = len(paper_records)

    for i, rec in enumerate(paper_records):
        if _stopped():
            break
        try:
            await asyncio.sleep(_POLITE_DELAY)
            safe_title = _safe_filename(rec["title"], i)
            pdf_dest = download_dir / f"{safe_title}.pdf"
            txt_dest = download_dir / f"{safe_title}.txt"

            _progress("download", i + 1, total, f"Resolving: {rec['title'][:60]}…")

            # Build a lightweight PaperRecord-like object for resolve_pdf_url
            _raw_authors = rec.get("authors") or []
            _authors_list = (
                _raw_authors if isinstance(_raw_authors, list)
                else [a.strip() for a in str(_raw_authors).split(",") if a.strip()]
            )
            _paper = PaperRecord(
                title=rec.get("title") or "",
                authors=_authors_list,
                year=str(rec.get("year") or ""),
                doi=rec.get("doi") or "",
                journal=rec.get("journal") or "",
                abstract=rec.get("abstract") or "",
                citation_count=int(rec.get("citation_count") or 0),
                open_access=bool(rec.get("open_access", False)),
                pdf_url=rec.get("pdf_url") or "",
                source_api=rec.get("source_api") or "",
                url=rec.get("url") or "",
                extra=rec.get("extra") or {},
            )

            pdf_url, method = await resolve_pdf_url(_paper)
            used_abstract = False

            if pdf_url:
                _progress("download", i + 1, total, f"Downloading PDF: {rec['title'][:50]}…")
                ok = await download_pdf(pdf_url, pdf_dest)
                if not ok and method == "known_oa_url":
                    ok = await playwright_pdf_download(pdf_url, pdf_dest)
                if ok:
                    paths_to_ingest.append(pdf_dest)
                    papers_downloaded += 1
                else:
                    used_abstract = True
            else:
                used_abstract = True

            if used_abstract:
                if _paper.abstract:
                    build_abstract_document(_paper, txt_dest)
                    paths_to_ingest.append(txt_dest)
                    papers_abstract_only += 1
                else:
                    papers_skipped += 1
                    failures.append(f"No PDF or abstract for: {rec['title'][:80]}")
                    continue

            extra_metas.append({
                "doi": _paper.doi,
                "journal": _paper.journal,
                "pub_date": _paper.year,
                "citation_count": str(_paper.citation_count),
                "pdf_method": method or ("abstract_fallback" if used_abstract else ""),
                "paper_abstract": _paper.abstract[:500] if _paper.abstract else "",
            })

            updated_rec = {**rec}
            updated_rec["local_path"] = str(pdf_dest if not used_abstract else txt_dest)
            updated_rec["method"] = method or ("abstract_fallback" if used_abstract else "")
            updated_rec["pdf_url"] = pdf_url
            updated_records.append(updated_rec)

            _progress(
                "download", i + 1, total,
                f"[{i + 1}/{total}] {rec['title'][:50]} — "
                + ("PDF" if not used_abstract else "abstract"),
            )

        except Exception as e:
            import traceback as _tb
            _detail = _tb.format_exc().strip().splitlines()[-1]
            failures.append(f"[{rec['title'][:60]}] {e} — {_detail}")
            papers_skipped += 1

    return {
        "stopped": _stopped(),
        "papers_downloaded": papers_downloaded,
        "papers_abstract_only": papers_abstract_only,
        "papers_skipped": papers_skipped,
        "failures": failures,
        "paper_records": updated_records,
        "paths_to_ingest": [str(p) for p in paths_to_ingest],
        "extra_metas": extra_metas,
    }


async def search_papers_pipeline(
    query: str,
    max_papers: int = 20,
    download_dir: Path = _DOWNLOAD_DIR,
    on_progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Search and download papers WITHOUT ingesting (legacy wrapper).

    Returns a dict with paper_records, paths_to_ingest, extra_metas, and stats.
    The caller should present the results to the user and let them confirm
    before calling ``ingest_papers_pipeline``.
    """
    search_res = await search_papers_only(
        query=query, max_papers=max_papers,
        on_progress=on_progress, stop_event=stop_event,
    )
    if search_res.get("stopped") or not search_res.get("paper_records"):
        return {**search_res, "papers_downloaded": 0, "papers_abstract_only": 0,
                "papers_skipped": 0, "paths_to_ingest": [], "extra_metas": []}

    dl_res = await download_selected_papers(
        paper_records=search_res["paper_records"],
        download_dir=download_dir,
        on_progress=on_progress,
        stop_event=stop_event,
    )
    return {
        **dl_res,
        "papers_found": search_res["papers_found"],
        "query": query,
    }


def ingest_papers_pipeline(
    paths_to_ingest: list[str],
    extra_metas: list[dict[str, str]],
    project: str,
    theme: str = "Literature Review",
    on_progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    *,
    chunking_method: str = "recursive",
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
    breakpoint_threshold: float = 95,
) -> dict[str, Any]:
    """Ingest previously-downloaded papers into the RAG store.

    Call this only after the user has reviewed and confirmed the search results
    from ``search_papers_pipeline``.
    """
    from ragsub_agent.tools import ingest_rag_paths

    def _progress(step: str, idx: int, total: int, msg: str) -> None:
        if on_progress:
            on_progress(step, idx, total, msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    total_chunks = 0
    failures: list[str] = []
    total = len(paths_to_ingest)

    for i, (path_str, emeta) in enumerate(zip(paths_to_ingest, extra_metas)):
        if _stopped():
            break
        path = Path(path_str)
        _progress("ingest", i + 1, total, f"Ingesting: {path.name}…")
        try:
            result = ingest_rag_paths(
                paths=[path],
                project=project,
                theme=theme,
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                breakpoint_threshold=breakpoint_threshold,
                extra_meta=emeta,
            )
            chunks = result.get("added_chunks", 0)
            total_chunks += chunks
            _progress("ingest", i + 1, total, f"[{i + 1}/{total}] {path.name} — {chunks} chunks")
        except Exception as e:
            failures.append(f"Ingest failed for {path.name}: {e}")

    return {
        "stopped": _stopped(),
        "ingested_chunks": total_chunks,
        "ingested_files": total if not _stopped() else sum(1 for _ in range(min(total, len(paths_to_ingest)))),
        "failures": failures,
    }


async def run_literature_search(
    query: str,
    project: str,
    theme: str = "Literature Review",
    max_papers: int = 20,
    download_dir: Path = _DOWNLOAD_DIR,
    on_progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """End-to-end literature search and ingestion pipeline (legacy API).

    Kept for backward compatibility. New callers should use
    ``search_papers_pipeline`` + ``ingest_papers_pipeline`` separately
    to allow human-in-the-loop confirmation.
    """
    search_result = await search_papers_pipeline(
        query=query,
        max_papers=max_papers,
        download_dir=download_dir,
        on_progress=on_progress,
        stop_event=stop_event,
    )

    if search_result.get("stopped"):
        return {**search_result, "project": project, "theme": theme, "ingested_chunks": 0}

    ingest_result = ingest_papers_pipeline(
        paths_to_ingest=search_result["paths_to_ingest"],
        extra_metas=search_result["extra_metas"],
        project=project,
        theme=theme,
        on_progress=on_progress,
        stop_event=stop_event,
    )

    return {
        "papers_found": search_result["papers_found"],
        "papers_downloaded": search_result["papers_downloaded"],
        "papers_abstract_only": search_result["papers_abstract_only"],
        "papers_skipped": search_result["papers_skipped"],
        "ingested_chunks": ingest_result["ingested_chunks"],
        "failures": search_result["failures"] + ingest_result["failures"],
        "paper_records": search_result["paper_records"],
        "project": project,
        "theme": theme,
        "query": query,
    }


def _safe_filename(title: str, idx: int) -> str:
    """Convert a title to a safe filesystem name."""
    clean = re.sub(r"[^a-zA-Z0-9 _-]", "", title or "")
    clean = clean.strip().replace(" ", "_")[:60]
    return f"{idx:03d}_{clean}" if clean else f"{idx:03d}_paper"
