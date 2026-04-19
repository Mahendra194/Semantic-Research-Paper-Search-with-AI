"""
Live arXiv Search Module
========================
Queries the arXiv public API (export.arxiv.org) to search for research papers
in real-time. Returns structured paper metadata compatible with the rest of
the application.

No heavy dependencies — uses only Python standard library (urllib + xml).

Rate limit: arXiv asks for ≤1 request every 3 seconds.
API docs: https://info.arxiv.org/help/api/index.html
"""

import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# arXiv API endpoint
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# XML namespaces used in arXiv Atom responses
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Cooldown between API requests (arXiv rate limit)
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 3.0  # seconds


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    category: Optional[str] = None,
    start: int = 0,
) -> Dict[str, Any]:
    """
    Search arXiv for papers matching the query.

    Args:
        query: Search query string (supports arXiv search syntax).
        max_results: Maximum number of results to return (1-50).
        sort_by: Sort criterion — 'relevance', 'lastUpdatedDate', or 'submittedDate'.
        sort_order: Sort direction — 'ascending' or 'descending'.
        category: Optional arXiv category filter (e.g., 'cs.AI', 'cs.CL').
        start: Offset for pagination.

    Returns:
        Dict with 'papers' (list of paper dicts), 'total_results', 'query'.
    """
    global _last_request_time

    # Rate limiting
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    # Build the search query
    search_query = f"all:{query}"
    if category:
        search_query = f"cat:{category} AND all:{query}"

    # URL parameters
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": min(max_results, 50),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info(f"arXiv API request: {url}")

    try:
        import ssl

        # Try to create a proper SSL context (handles macOS cert issues)
        try:
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # certifi not available — try the default context,
            # and fall back to unverified if that fails too
            ssl_context = ssl.create_default_context()

        req = urllib.request.Request(url, headers={"User-Agent": "SemanticPaperSearch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
                xml_data = response.read().decode("utf-8")
        except ssl.SSLCertVerificationError:
            # Last resort: unverified context (safe for read-only arXiv queries)
            logger.warning("SSL verification failed — using unverified context for arXiv API")
            ssl_context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
                xml_data = response.read().decode("utf-8")

        _last_request_time = time.time()
    except Exception as e:
        logger.error(f"arXiv API request failed: {e}")
        return {"papers": [], "total_results": 0, "query": query, "error": str(e)}

    # Parse XML response
    papers = _parse_arxiv_response(xml_data)

    # Get total results count
    root = ET.fromstring(xml_data)
    total_el = root.find("opensearch:totalResults", NS)
    total_results = int(total_el.text) if total_el is not None else len(papers)

    return {
        "papers": papers,
        "total_results": total_results,
        "query": query,
    }


def _parse_arxiv_response(xml_data: str) -> List[Dict[str, Any]]:
    """Parse arXiv Atom XML response into a list of paper dicts."""
    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", NS):
        paper = _parse_entry(entry)
        if paper:
            papers.append(paper)

    return papers


def _parse_entry(entry: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a single <entry> element into a paper dict."""
    try:
        # Title
        title_el = entry.find("atom:title", NS)
        title = _clean_text(title_el.text) if title_el is not None else "Untitled"

        # Abstract / summary
        summary_el = entry.find("atom:summary", NS)
        abstract = _clean_text(summary_el.text) if summary_el is not None else ""

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", NS):
            name_el = author_el.find("atom:name", NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        authors_str = ", ".join(authors)

        # arXiv ID — extract from the <id> URL
        id_el = entry.find("atom:id", NS)
        arxiv_url = id_el.text.strip() if id_el is not None else ""
        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url

        # Remove version suffix for cleaner display (e.g., "2301.12345v2" -> "2301.12345")
        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        # Categories
        categories = []
        for cat_el in entry.findall("arxiv:primary_category", NS):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall("atom:category", NS):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)
        categories_str = " ".join(categories)

        # Published / updated dates
        published_el = entry.find("atom:published", NS)
        published = published_el.text.strip()[:10] if published_el is not None else ""

        updated_el = entry.find("atom:updated", NS)
        updated = updated_el.text.strip()[:10] if updated_el is not None else ""

        # Year
        year = published[:4] if published else ""

        # PDF link
        pdf_url = ""
        for link_el in entry.findall("atom:link", NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        # Comment (often contains page count, conference info)
        comment_el = entry.find("arxiv:comment", NS)
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ""

        return {
            "id": arxiv_id_clean,
            "title": title,
            "abstract": abstract,
            "authors": authors_str,
            "categories": categories_str,
            "year": year,
            "published": published,
            "updated": updated,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
            "comment": comment,
            "source": "arxiv_live",
        }

    except Exception as e:
        logger.warning(f"Failed to parse arXiv entry: {e}")
        return None


def _clean_text(text: str) -> str:
    """Clean whitespace from arXiv text fields."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick demo
    result = search_arxiv("attention mechanism transformer", max_results=3)
    print(f"Total results: {result['total_results']}")
    for p in result["papers"]:
        print(f"\n  📄 {p['title']}")
        print(f"     Authors: {p['authors'][:60]}...")
        print(f"     Categories: {p['categories']}")
        print(f"     Published: {p['published']}")
        print(f"     URL: {p['arxiv_url']}")
