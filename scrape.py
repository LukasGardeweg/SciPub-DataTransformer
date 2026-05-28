"""
Searches Semantic Scholar for highly-cited sports science intervention studies,
attempts to download open-access PDFs via Unpaywall, and saves metadata.

Setup:
    pip install requests
    set UNPAYWALL_EMAIL=your-email@example.com   (required for Unpaywall API)

Usage:
    python scrape.py
"""

import json
import os
import re
import time
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────────────────────
SEARCH_QUERY  = "exercise training intervention randomized controlled trial sports"
MAX_RESULTS   = 100   # total papers to fetch from Semantic Scholar
MIN_CITATIONS = 30    # minimum citation count to keep a paper
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "your-email@example.com")

DATA_DIR  = Path("data")
PDF_DIR   = DATA_DIR / "pdfs"
META_FILE = DATA_DIR / "metadata.json"

S2_SEARCH  = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS  = "title,year,citationCount,externalIds,authors"
UNPAYW_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"
# ────────────────────────────────────────────────────────────────────────────


def search_semantic_scholar(query: str, limit: int) -> list[dict]:
    results = []
    offset = 0
    while len(results) < limit:
        batch = min(100, limit - len(results))
        resp = requests.get(S2_SEARCH, params={
            "query":  query,
            "fields": S2_FIELDS,
            "limit":  batch,
            "offset": offset,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        results.extend(data)
        offset += len(data)
        time.sleep(1)
    return results


def get_oa_pdf_url(doi: str) -> str | None:
    try:
        resp = requests.get(
            UNPAYW_URL.format(doi=doi, email=UNPAYWALL_EMAIL),
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        loc = resp.json().get("best_oa_location")
        return loc.get("url_for_pdf") if loc else None
    except Exception:
        return None


def doi_to_filename(doi: str) -> str:
    return re.sub(r"[^\w\-]", "_", doi) + ".pdf"


def download_pdf(url: str, path: Path) -> bool:
    try:
        resp = requests.get(
            url, timeout=60,
            headers={"User-Agent": "SciPub-DataTransformer/1.0 (research use)"},
        )
        if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", "").lower():
            path.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


def main() -> None:
    if UNPAYWALL_EMAIL == "your-email@example.com":
        print("Warning: set UNPAYWALL_EMAIL env var to enable PDF downloads.")

    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    print(f"Searching: '{SEARCH_QUERY}'")
    papers = search_semantic_scholar(SEARCH_QUERY, MAX_RESULTS)
    print(f"  {len(papers)} papers retrieved")

    papers = [
        p for p in papers
        if p.get("citationCount", 0) >= MIN_CITATIONS
        and p.get("externalIds", {}).get("DOI")
    ]
    papers.sort(key=lambda p: p["citationCount"], reverse=True)
    print(f"  {len(papers)} papers with >={MIN_CITATIONS} citations and a DOI")

    metadata = []
    for i, paper in enumerate(papers):
        doi     = paper["externalIds"]["DOI"]
        title   = paper.get("title", "")
        year    = paper.get("year")
        cites   = paper.get("citationCount", 0)
        author_list = paper.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in author_list[:3])
        if len(author_list) > 3:
            authors += " et al."

        pdf_path   = PDF_DIR / doi_to_filename(doi)
        pdf_status = "already_exists"

        if not pdf_path.exists():
            print(f"[{i+1}/{len(papers)}] {cites:>5}x  {title[:65]}…")
            pdf_url = get_oa_pdf_url(doi)
            if pdf_url:
                ok = download_pdf(pdf_url, pdf_path)
                pdf_status = "downloaded" if ok else "download_failed"
            else:
                pdf_status = "no_oa_pdf"
            print(f"         PDF: {pdf_status}")
            time.sleep(1)

        metadata.append({
            "doi":        doi,
            "title":      title,
            "year":       year,
            "authors":    authors,
            "citations":  cites,
            "pdf_file":   str(pdf_path) if pdf_status in ("downloaded", "already_exists") else None,
            "pdf_status": pdf_status,
        })

    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    downloaded = sum(1 for m in metadata if m["pdf_status"] in ("downloaded", "already_exists"))
    print(f"\nDone. {downloaded}/{len(metadata)} PDFs available → {META_FILE}")


if __name__ == "__main__":
    main()
