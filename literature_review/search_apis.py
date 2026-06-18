"""Academic database search clients using free, legal APIs.

Sources:
- Semantic Scholar: https://api.semanticscholar.org  (free, no key, 100 req/sec)
- OpenAlex:         https://api.openalex.org          (free, no key, 100k req/day)
- CrossRef:         https://api.crossref.org          (free, no key, metadata + DOI)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_OA_BASE = "https://api.openalex.org"
_CR_BASE = "https://api.crossref.org"

_SS_FIELDS = (
    "paperId,title,authors,year,externalIds,openAccessPdf,"
    "publicationVenue,citationCount,abstract,isOpenAccess"
)

# Polite delay between requests (seconds) to avoid rate-limiting
_POLITE_DELAY = 0.5


@dataclass
class PaperRecord:
    """Normalised representation of an academic paper."""

    title: str
    authors: list[str]
    year: str
    doi: str
    journal: str
    abstract: str
    citation_count: int
    open_access: bool
    pdf_url: str  # empty string if unknown/paywalled
    source_api: str
    paper_id: str = ""  # Semantic Scholar ID if available
    openalex_id: str = ""
    url: str = ""  # landing page URL
    extra: dict[str, Any] = field(default_factory=dict)

    def apa_reference(self) -> str:
        """Build an APA 7th-edition reference string."""
        author_part = _format_apa_authors(self.authors)
        year = self.year or "n.d."
        title = self.title or "Untitled"
        journal = self.journal or ""
        doi_part = f" https://doi.org/{self.doi}" if self.doi else ""
        if journal:
            return f"{author_part} ({year}). {title}. {journal}.{doi_part}"
        return f"{author_part} ({year}). {title}.{doi_part}"

    def intext_citation(self) -> str:
        """Short in-text citation: (First Author, Year)."""
        first = self.authors[0] if self.authors else "Unknown"
        # Use surname only
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        return f"({surname}, {self.year or 'n.d.'})"


def _format_apa_authors(authors: list[str]) -> str:
    """Format author list in APA style (up to 20 authors)."""
    if not authors:
        return "Unknown Author."
    formatted = []
    for a in authors[:20]:
        a = a.strip()
        if not a:
            continue
        # If already "Last, F." keep as-is, otherwise try to reformat
        if "," in a:
            formatted.append(a)
        else:
            parts = a.split()
            if len(parts) >= 2:
                surname = parts[-1]
                initials = ". ".join(p[0] for p in parts[:-1] if p) + "."
                formatted.append(f"{surname}, {initials}")
            else:
                formatted.append(a)
    if len(authors) > 20:
        return "; ".join(formatted) + "; et al."
    return "; ".join(formatted) + ("." if formatted else "")


# ── Semantic Scholar ──────────────────────────────────────────────────────────

async def search_semantic_scholar(
    query: str,
    limit: int = 20,
    client: httpx.AsyncClient | None = None,
) -> list[PaperRecord]:
    """Search Semantic Scholar and return normalised PaperRecord list."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30)
    try:
        await asyncio.sleep(_POLITE_DELAY)
        resp = await client.get(
            f"{_SS_BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": _SS_FIELDS},
            headers={"User-Agent": "DeepAgentsLitReview/1.0 (research tool; contact: user@example.com)"},
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own_client:
            await client.aclose()

    records: list[PaperRecord] = []
    for paper in data.get("data", []):
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        pdf_url = ""
        oa = paper.get("openAccessPdf")
        if oa and isinstance(oa, dict):
            pdf_url = oa.get("url", "")

        venue = paper.get("publicationVenue") or {}
        journal = venue.get("name", "") or ""

        records.append(PaperRecord(
            title=paper.get("title", ""),
            authors=[a.get("name", "") for a in (paper.get("authors") or [])],
            year=str(paper.get("year") or ""),
            doi=doi,
            journal=journal,
            abstract=paper.get("abstract") or "",
            citation_count=paper.get("citationCount") or 0,
            open_access=bool(paper.get("isOpenAccess")),
            pdf_url=pdf_url,
            source_api="semantic_scholar",
            paper_id=paper.get("paperId", ""),
            url=f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
        ))
    return records


# ── OpenAlex ──────────────────────────────────────────────────────────────────

async def search_openalex(
    query: str,
    limit: int = 20,
    client: httpx.AsyncClient | None = None,
) -> list[PaperRecord]:
    """Search OpenAlex and return normalised PaperRecord list."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30)
    try:
        await asyncio.sleep(_POLITE_DELAY)
        resp = await client.get(
            f"{_OA_BASE}/works",
            params={
                "search": query,
                "per-page": limit,
                "filter": "is_paratext:false",
                "select": "id,doi,title,authorships,publication_year,primary_location,"
                          "cited_by_count,abstract_inverted_index,open_access,best_oa_location",
                "mailto": "research@deepagents.local",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own_client:
            await client.aclose()

    records: list[PaperRecord] = []
    for work in data.get("results", []):
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        oa = work.get("open_access") or {}
        best_oa = work.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or ""

        authors = []
        for auth in (work.get("authorships") or []):
            author = auth.get("author") or {}
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        journal = source.get("display_name", "") or ""

        # Reconstruct abstract from inverted index
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        records.append(PaperRecord(
            title=work.get("title", ""),
            authors=authors,
            year=str(work.get("publication_year") or ""),
            doi=doi,
            journal=journal,
            abstract=abstract,
            citation_count=work.get("cited_by_count") or 0,
            open_access=oa.get("is_oa", False),
            pdf_url=pdf_url,
            source_api="openalex",
            openalex_id=work.get("id", ""),
            url=f"https://doi.org/{doi}" if doi else work.get("id", ""),
        ))
    return records


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


# ── CrossRef (metadata enrichment) ───────────────────────────────────────────

async def enrich_crossref(
    doi: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch CrossRef metadata for a DOI (journal, volume, issue, pages)."""
    if not doi:
        return {}
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20)
    try:
        await asyncio.sleep(_POLITE_DELAY)
        resp = await client.get(
            f"{_CR_BASE}/works/{doi}",
            params={"mailto": "research@deepagents.local"},
        )
        if resp.status_code != 200:
            return {}
        msg = resp.json().get("message", {})
        container = (msg.get("container-title") or [""])[0]
        volume = msg.get("volume", "")
        issue = msg.get("issue", "")
        page = msg.get("page", "")
        return {
            "journal": container,
            "volume": volume,
            "issue": issue,
            "page": page,
        }
    except Exception:
        return {}
    finally:
        if own_client:
            await client.aclose()


# ── Unified search ────────────────────────────────────────────────────────────

async def search_papers(
    query: str,
    max_results: int = 20,
) -> list[PaperRecord]:
    """Run parallel search across Semantic Scholar and OpenAlex, deduplicate by DOI.

    Papers are scored by:
      - Open-access availability (PDF accessible)
      - Citation count (normalised to [0, 1])
      - Recency (last 5 years preferred)

    Returns up to ``max_results`` papers sorted by score descending.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        ss_task = asyncio.create_task(search_semantic_scholar(query, limit=max_results, client=client))
        oa_task = asyncio.create_task(search_openalex(query, limit=max_results, client=client))
        ss_results, oa_results = await asyncio.gather(ss_task, oa_task, return_exceptions=True)

    all_papers: list[PaperRecord] = []
    if isinstance(ss_results, list):
        all_papers.extend(ss_results)
    if isinstance(oa_results, list):
        all_papers.extend(oa_results)

    # Deduplicate by DOI (keep highest citation count version)
    seen_dois: dict[str, PaperRecord] = {}
    no_doi: list[PaperRecord] = []
    for p in all_papers:
        if p.doi:
            doi_key = p.doi.lower().strip()
            if doi_key not in seen_dois or p.citation_count > seen_dois[doi_key].citation_count:
                seen_dois[doi_key] = p
        else:
            no_doi.append(p)

    # Also deduplicate no-DOI papers by title similarity
    seen_titles: set[str] = {_title_key(p.title) for p in seen_dois.values()}
    for p in no_doi:
        tk = _title_key(p.title)
        if tk not in seen_titles:
            seen_titles.add(tk)
            seen_dois[f"_notitle_{tk}"] = p

    candidates = list(seen_dois.values())

    # Score and rank
    max_cites = max((p.citation_count for p in candidates), default=1) or 1
    import datetime
    current_year = datetime.datetime.now().year

    def _score(p: PaperRecord) -> float:
        oa_score = 1.0 if p.pdf_url else (0.5 if p.open_access else 0.0)
        cite_score = min(p.citation_count / max_cites, 1.0)
        year_int = int(p.year) if (p.year or "").isdigit() else 0
        recency = max(0.0, 1.0 - (current_year - year_int) / 10) if year_int else 0.0
        return 0.4 * oa_score + 0.35 * cite_score + 0.25 * recency

    candidates.sort(key=_score, reverse=True)
    return candidates[:max_results]


def _title_key(title: str) -> str:
    """Normalise title to a deduplication key."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:60]
