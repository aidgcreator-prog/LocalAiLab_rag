"""PDF resolution pipeline.

Resolution order per paper:
  1. Unpaywall API          — finds legal open-access PDF URLs via DOI
  2. arXiv API              — if arXiv ID present in external IDs
  3. PubMed Central         — if PMCID present
  4. Playwright fallback    — whitelisted open-access publisher pages only
  5. Abstract-only fallback — stores title + abstract as a virtual document

All downloads are polite (delays between requests, proper User-Agent).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .search_apis import PaperRecord

_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
_PMC_PDF = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"

# Only attempt Playwright on known open-access publishers to avoid ToS violations
OA_PUBLISHER_WHITELIST = [
    "plos.org",
    "frontiersin.org",
    "mdpi.com",
    "springeropen.com",
    "biomedcentral.com",
    "ncbi.nlm.nih.gov/pmc",
    "elifesciences.org",
    "royalsocietypublishing.org",
    "f1000research.com",
    "peerj.com",
    "hindawi.com",
    "jmlr.org",
    "openreview.net",
]

_POLITE_DELAY = 1.0  # seconds between download requests
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB cap


async def resolve_pdf_url(
    paper: "PaperRecord",
    email: str = "research@deepagents.local",
) -> tuple[str, str]:
    """Resolve the best available PDF URL for a paper.

    Returns:
        (pdf_url, method) where method is one of:
        'direct', 'unpaywall', 'arxiv', 'pmc', 'known_oa_url', or '' (not found)
    """
    # 0. Already have a direct PDF URL (from search API)
    if paper.pdf_url and paper.pdf_url.endswith(".pdf"):
        return paper.pdf_url, "direct"

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        # 1. Unpaywall (requires DOI)
        if paper.doi:
            url, method = await _resolve_unpaywall(paper.doi, email, client)
            if url:
                return url, method

        # 2. arXiv
        arxiv_id = _extract_arxiv_id(paper)
        if arxiv_id:
            url = _ARXIV_PDF.format(arxiv_id=arxiv_id)
            if await _url_is_accessible(url, client):
                return url, "arxiv"

        # 3. PubMed Central
        pmcid = _extract_pmcid(paper)
        if pmcid:
            url = _PMC_PDF.format(pmcid=pmcid)
            if await _url_is_accessible(url, client):
                return url, "pmc"

        # 4. Known OA publisher URL in paper.url
        if paper.url and _is_whitelisted(paper.url):
            return paper.url, "known_oa_url"

    return "", ""


async def _resolve_unpaywall(
    doi: str,
    email: str,
    client: httpx.AsyncClient,
) -> tuple[str, str]:
    """Query Unpaywall for a legal open-access PDF URL."""
    try:
        await asyncio.sleep(_POLITE_DELAY)
        resp = await client.get(
            f"{_UNPAYWALL_BASE}/{doi}",
            params={"email": email},
        )
        if resp.status_code != 200:
            return "", ""
        data = resp.json()
        best = data.get("best_oa_location") or {}
        url = best.get("url_for_pdf") or best.get("url") or ""
        return url, "unpaywall"
    except Exception:
        return "", ""


async def _url_is_accessible(url: str, client: httpx.AsyncClient) -> bool:
    """Check if a URL returns 200 with a HEAD request."""
    try:
        await asyncio.sleep(_POLITE_DELAY)
        r = await client.head(url, headers={"User-Agent": "DeepAgentsLitReview/1.0"})
        return r.status_code == 200
    except Exception:
        return False


def _extract_arxiv_id(paper: "PaperRecord") -> str:
    """Extract arXiv ID from paper metadata."""
    # Check extra dict (from Semantic Scholar externalIds)
    arxiv = paper.extra.get("ArXiv", "") or paper.extra.get("arxiv", "")
    if arxiv:
        return arxiv
    # Try URL
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)", paper.url or "")
    if m:
        return m.group(1)
    return ""


def _extract_pmcid(paper: "PaperRecord") -> str:
    """Extract PubMed Central ID from paper metadata."""
    pmcid = paper.extra.get("PubMedCentral", "") or paper.extra.get("pmcid", "")
    if pmcid:
        return str(pmcid).replace("PMC", "")
    return ""


def _is_whitelisted(url: str) -> bool:
    return any(domain in url for domain in OA_PUBLISHER_WHITELIST)


# ── Download ──────────────────────────────────────────────────────────────────

async def download_pdf(
    url: str,
    dest: Path,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Download a PDF from ``url`` to ``dest``. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=60, follow_redirects=True)
    try:
        await asyncio.sleep(_POLITE_DELAY)
        async with client.stream(
            "GET",
            url,
            headers={"User-Agent": "DeepAgentsLitReview/1.0 (research tool)"},
        ) as resp:
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("content-type", "").lower()
            # Accept PDF, octet-stream, binary, or unknown content types
            # We validate using magic bytes below
            _rejected_types = ("text/html", "text/plain", "application/json")
            if any(t in content_type for t in _rejected_types):
                return False
            data = b""
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                data += chunk
                if len(data) > _MAX_PDF_BYTES:
                    return False  # Too large
        # Validate it's actually a PDF via magic bytes
        if not data.startswith(b"%PDF"):
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False
    finally:
        if own_client:
            await client.aclose()


async def playwright_pdf_download(
    url: str,
    dest: Path,
) -> bool:
    """Use Playwright to find and download a PDF from a whitelisted OA publisher page.

    Only proceeds if the URL matches the OA publisher whitelist.
    Returns True if a PDF was successfully downloaded.
    """
    if not _is_whitelisted(url):
        return False

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            pdf_url: str = ""

            # Intercept requests to detect direct PDF URLs
            async def _intercept(request):
                nonlocal pdf_url
                if request.resource_type == "document" and ".pdf" in request.url:
                    pdf_url = request.url

            page.on("request", _intercept)

            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            await asyncio.sleep(2)  # Let JS settle

            # If PDF URL detected via interception, download it directly
            if pdf_url:
                await browser.close()
                return await download_pdf(pdf_url, dest)

            # Try finding a PDF link on the page
            selectors = [
                "a[href$='.pdf']",
                "a[data-track-action='download pdf']",
                "a:has-text('PDF')",
                "a:has-text('Download PDF')",
                "a:has-text('Full text PDF')",
            ]
            for selector in selectors:
                try:
                    link = page.locator(selector).first
                    href = await link.get_attribute("href")
                    if href:
                        if not href.startswith("http"):
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                        await browser.close()
                        return await download_pdf(href, dest)
                except Exception:
                    continue

            await browser.close()
            return False
    except Exception:
        return False


# ── Abstract-only document builder ───────────────────────────────────────────

def build_abstract_document(paper: "PaperRecord", dest: Path) -> Path:
    """Write a plain-text abstract document as fallback when no PDF is available.

    The file is suitable for ingestion into the RAG pipeline and includes
    all available bibliographic metadata.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Title: {paper.title}",
        f"Authors: {'; '.join(paper.authors)}",
        f"Year: {paper.year}",
        f"Journal: {paper.journal}",
        f"DOI: {paper.doi}",
        f"Citations: {paper.citation_count}",
        "",
        "Abstract:",
        paper.abstract or "[No abstract available]",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
