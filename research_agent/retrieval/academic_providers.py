import os
import re
import threading
import time
from dataclasses import dataclass, field

import requests


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
_S2_RATE_LOCK = threading.Lock()
_S2_LAST_REQUEST_STARTED_AT = 0.0


def _semantic_scholar_interval_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("S2_MIN_REQUEST_INTERVAL_SECONDS", "1.5")))
    except ValueError:
        return 1.5


def _wait_for_semantic_scholar_slot() -> None:
    """进程内统一限制 Semantic Scholar 请求起始频率，默认低于每秒一次。"""
    global _S2_LAST_REQUEST_STARTED_AT
    interval = _semantic_scholar_interval_seconds()
    with _S2_RATE_LOCK:
        now = time.monotonic()
        wait_seconds = interval - (now - _S2_LAST_REQUEST_STARTED_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _S2_LAST_REQUEST_STARTED_AT = time.monotonic()


@dataclass
class AcademicSearchResult:
    papers: list = field(default_factory=list)
    provider: str = "none"
    errors: list[str] = field(default_factory=list)


def format_authors(authors: list, limit: int = 20) -> str:
    return ", ".join(
        str(author.get("name") or "").strip()
        for author in (authors or [])[:limit]
        if str(author.get("name") or "").strip()
    ) or "未知作者"


def paper_summary_fields(paper_data: dict) -> dict | None:
    """直接把学术 API 元数据转成证据，避免每篇摘要再串行调用本地 LLM。"""
    title = str(paper_data.get("title") or "未知标题").strip()
    abstract = str(paper_data.get("abstract") or "").strip()
    tldr_obj = paper_data.get("tldr") or {}
    tldr_text = str(tldr_obj.get("text") or "").strip() if isinstance(tldr_obj, dict) else ""
    evidence_text = abstract or tldr_text
    if not evidence_text:
        return None

    external_ids = paper_data.get("externalIds") or {}
    fallback_url = str(paper_data.get("url") or "").strip()
    pdf_url = (paper_data.get("openAccessPdf") or {}).get("url") or (
        f"https://arxiv.org/pdf/{external_ids.get('ArXiv')}.pdf"
        if external_ids.get("ArXiv") else fallback_url
    )
    return {
        "title": title,
        "authors": format_authors(paper_data.get("authors") or []),
        "year": str(paper_data.get("year") or "未知年份"),
        "source": f"🔗网络链接: {pdf_url}",
        "core_method": evidence_text[:1800],
        "key_findings": (tldr_text or evidence_text)[:1800],
        "doi": str(external_ids.get("DOI") or "未知"),
        "venue": str(paper_data.get("venue") or "未知"),
    }


def _openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positioned_words = []
    for word, positions in inverted_index.items():
        positioned_words.extend((int(position), str(word)) for position in positions or [])
    return " ".join(word for _, word in sorted(positioned_words))


def _openalex_year_filter(year: str) -> str:
    years = re.findall(r"(?:19|20)\d{2}", str(year or ""))
    if not years:
        return ""
    start, end = min(years), max(years)
    return f"from_publication_date:{start}-01-01,to_publication_date:{end}-12-31"


def _normalize_openalex_work(work: dict) -> dict:
    authors = [
        {"name": str((entry.get("author") or {}).get("display_name") or "").strip()}
        for entry in work.get("authorships") or []
        if str((entry.get("author") or {}).get("display_name") or "").strip()
    ]
    primary_location = work.get("primary_location") or {}
    pdf_url = primary_location.get("pdf_url") or ""
    landing_url = primary_location.get("landing_page_url") or work.get("doi") or work.get("id") or ""
    return {
        "title": work.get("display_name") or "未知标题",
        "authors": authors,
        "year": work.get("publication_year"),
        "abstract": _openalex_abstract(work.get("abstract_inverted_index")),
        "citationCount": work.get("cited_by_count", 0),
        "externalIds": {"DOI": work.get("doi")} if work.get("doi") else {},
        "venue": str((primary_location.get("source") or {}).get("display_name") or "未知"),
        "url": landing_url,
        "tldr": None,
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
    }


def search_academic_papers_detailed(
    query: str,
    year: str = "",
    limit: int = 20,
    max_retries: int = 2,
) -> AcademicSearchResult:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,abstract,citationCount,externalIds,url,tldr,openAccessPdf",
    }
    if year:
        params["year"] = year
    headers = {"x-api-key": os.environ.get("S2_API_KEY")} if os.environ.get("S2_API_KEY") else {}
    errors = []

    for attempt in range(max_retries):
        try:
            _wait_for_semantic_scholar_slot()
            response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=(6, 18))
            if response.status_code == 429:
                errors.append("Semantic Scholar HTTP 429")
                if attempt + 1 < max_retries:
                    time.sleep(min(2 * (attempt + 1), 4))
                continue
            response.raise_for_status()
            papers = response.json().get("data", [])
            if papers:
                return AcademicSearchResult(papers=papers, provider="Semantic Scholar", errors=errors)
            break
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            errors.append(f"Semantic Scholar {type(exc).__name__}" + (f" HTTP {status}" if status else ""))
            if status in {400, 401, 403}:
                break
            if attempt + 1 < max_retries:
                time.sleep(1)

    openalex_params = {"search": query, "per-page": min(max(1, limit), 100)}
    year_filter = _openalex_year_filter(year)
    if year_filter:
        openalex_params["filter"] = year_filter
    try:
        response = requests.get(OPENALEX_URL, params=openalex_params, timeout=(6, 18))
        response.raise_for_status()
        papers = [_normalize_openalex_work(work) for work in response.json().get("results", [])]
        if papers:
            return AcademicSearchResult(papers=papers, provider="OpenAlex", errors=errors)
    except requests.exceptions.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        errors.append(f"OpenAlex {type(exc).__name__}" + (f" HTTP {status}" if status else ""))
    return AcademicSearchResult(provider="none", errors=errors)


def search_academic_papers(query: str, year: str = "", limit: int = 20, max_retries: int = 2) -> list:
    return search_academic_papers_detailed(query, year, limit, max_retries).papers
