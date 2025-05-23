#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scopus_dblp_by_affiliation.py · v2.0
------------------------------------
Harvest author-level metrics from **Scopus** and **DBLP** for *all* researchers
currently affiliated with a user-supplied institution, compute DBLP-side
h-indices with Semantic Scholar citation counts, and export the merged table in
both **CSV** and **Excel** formats.

Major features added in v2.0
• optional `--scopus-key` flag (overrides / replaces env-var)
• graceful fallback if the `python-slugify` package is absent
• Python 3.7-compatible type hints (Optional[…], Dict[…]) instead of 3.10 union syntax
• clearer progress bars & error messages
"""

# ───────────────────────── Imports & constants ────────────────────────────
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from pybliometrics.scopus import AuthorRetrieval
from tqdm import tqdm

# ---------- slugify: use library if present, else a minimal fallback -------
try:
    from slugify import slugify  # pip install python-slugify
except ModuleNotFoundError:
    def slugify(text: str) -> str:                         # very simple ASCII slug
        return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


# ──────────────────────── HTTP helper with back-off ────────────────────────
def safe_get(url: str,
             params: Optional[Dict] = None,
             headers: Optional[Dict] = None,
             tries: int = 5,
             pause: float = 0.6) -> requests.Response:
    """
    Robust HTTP GET with exponential back-off for 429 / 5xx responses.
    Raises for all non-recoverable HTTP errors.
    """
    for attempt in range(1, tries + 1):
        r = requests.get(url, params=params or {}, headers=headers or {})
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(pause * 2 ** attempt)
        else:
            r.raise_for_status()
    r.raise_for_status()     # if everything failed


# ─────────────────────────── Scopus helpers ────────────────────────────────
def scopus_affiliation_search(affil: str,
                              scopus_key: str,
                              max_authors: int = 9999) -> List[Dict]:
    """
    Query the Scopus Author Search API for every author whose *current* affiliation
    exactly matches `affil`. Returns a list of {'scopus_id', 'name'} dicts.
    """
    url = "https://api.elsevier.com/content/search/author"
    query = f'AFFIL("{affil}")'          # exact phrase search
    authors: List[Dict] = []

    start, page = 0, 1
    page_size = 200                      # API max
    while len(authors) < max_authors:
        params = {
            "apiKey": scopus_key,
            "query": query,
            "start": start,
            "count": page_size,
            "view": "STANDARD"
        }
        r = safe_get(url, params=params)
        entries = r.json().get("search-results", {}).get("entry", [])
        if not entries:
            break
        for e in entries:
            authors.append({
                "scopus_id": e["dc:identifier"].split(":")[1],
                "name": f'{e["preferred-name"]["given-name"]} {e["preferred-name"]["surname"]}'
            })
            if len(authors) >= max_authors:
                break
        start += page_size
        page += 1
    return authors


def scopus_profile(author_id: str, scopus_key: str) -> Dict:
    """
    Retrieve (and cache) the full Scopus profile.  Returns a trimmed dict with key metrics.
    """
    cache = Path("_cache") / f"scopus_{author_id}.json"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        prof_json = json.loads(cache.read_text())
    else:
        # pybliometrics picks the key from ~/.pybliometrics/config.ini,
        # but we can override via env var for the current call
        os.environ["SCOPUS_API_KEY"] = scopus_key
        ar = AuthorRetrieval(author_id, refresh=True)
        prof_json = ar._json
        cache.write_text(json.dumps(prof_json, indent=2))

    core = prof_json["author-profile"]["coredata"]
    pref = prof_json["author-profile"]["preferred-name"]
    current_affil = prof_json["author-profile"].get("affiliation-current", {})

    return {
        "name": f'{pref["given-name"]} {pref["surname"]}',
        "scopus_id": author_id,
        "scopus_h": int(core["h-index"]),
        "scopus_docs": int(core["document-count"]),
        "affiliation": current_affil.get("affiliation-name")
    }


# ─────────────────────────── DBLP helpers ──────────────────────────────────
def dblp_pid(name: str) -> Optional[str]:
    """
    Resolve a human name to a stable DBLP Person ID (PID). Returns None if not found.
    """
    r = safe_get("https://dblp.org/search/author/api",
                 params={"q": name, "format": "json"})
    hits = r.json()["result"]["hits"]["hit"]
    return hits[0]["info"]["authorid"] if hits else None


def dblp_publications(pid: str) -> List[Dict]:
    """
    Retrieve (and cache) the full DBLP publication list for a PID.
    """
    cache = Path("_cache") / f"dblp_{pid}.json"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        r = safe_get(f"https://dblp.org/pid/{pid}.json")
        data = r.json()["result"]["hits"]["hit"]
        cache.write_text(json.dumps(data, indent=2))
    return data


# ─────────────── Semantic Scholar & h-index computation ────────────────────
def semantic_scholar_citations(doi: str,
                               s2_headers: Dict[str, str]) -> int:
    """
    Return Semantic Scholar citation count for a DOI. Missing → 0.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    r = safe_get(url, params={"fields": "citationCount"}, headers=s2_headers)
    return int(r.json().get("citationCount", 0))


def dblp_h_index(records: List[Dict],
                 s2_headers: Dict[str, str]) -> int:
    """
    Compute the h-index over a DBLP record list using Semantic Scholar citations.
    """
    citations: List[int] = []
    for rec in records:
        doi = rec["info"].get("doi")
        if doi:
            citations.append(semantic_scholar_citations(doi, s2_headers))
            time.sleep(0.1)              # respect rate limits
    citations.sort(reverse=True)
    return sum(c >= i + 1 for i, c in enumerate(citations))


# ─────────────────── Per-author orchestration routine ───────────────────────
def process_author(author_stub: Dict,
                   scopus_key: str,
                   s2_headers: Dict[str, str]) -> Dict:
    """
    Merge Scopus & DBLP metrics for one author stub from the search list.
    """
    prof = scopus_profile(author_stub["scopus_id"], scopus_key)

    pid = dblp_pid(prof["name"])
    if pid:
        recs = dblp_publications(pid)
        h_dblp = dblp_h_index(recs, s2_headers)
        dblp_docs = len(recs)
    else:
        h_dblp = 0
        dblp_docs = 0

    return {
        "name": prof["name"],
        "affiliation": prof["affiliation"],
        "scopus_id": prof["scopus_id"],
        "dblp_pid": pid,
        "scopus_h": prof["scopus_h"],
        "dblp_h": h_dblp,
        "scopus_docs": prof["scopus_docs"],
        "dblp_docs": dblp_docs
    }


# ──────────────────────────────── main() ───────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest Scopus & DBLP author metrics for an institution"
    )
    parser.add_argument("affiliation",
                        help='Exact affiliation string, e.g. "Stanford University"')
    parser.add_argument("--max-authors", type=int, default=500,
                        help="Upper cap to avoid excessive API calls (default 500)")
    parser.add_argument("--scopus-key",
                        help="Scopus API key (overrides SCOPUS_API_KEY environment variable)")
    args = parser.parse_args()

    # -- API keys ------------------------------------------------------------
    scopus_key = args.scopus_key or os.getenv("SCOPUS_API_KEY")
    if not scopus_key:
        sys.exit("❌  Supply the Scopus key via --scopus-key or SCOPUS_API_KEY env var.")
    s2_key = os.getenv("S2_API_KEY")
    s2_headers: Dict[str, str] = {"x-api-key": s2_key} if s2_key else {}

    affil = args.affiliation
    file_stem = f"metrics_{slugify(affil)}"

    # -- Scopus Author Search -------------------------------------------------
    print(f"🔍  Fetching author list for “{affil}”…")
    authors = scopus_affiliation_search(affil, scopus_key, args.max_authors)
    if not authors:
        sys.exit("No authors found — verify the affiliation string.")

    # -- Per-author processing ------------------------------------------------
    merged_rows: List[Dict] = []
    for stub in tqdm(authors, desc="Processing authors"):
        try:
            merged_rows.append(process_author(stub, scopus_key, s2_headers))
        except Exception as exc:           # continue on individual failures
            print(f"⚠️  {stub['name']} skipped: {exc}")

    if not merged_rows:
        sys.exit("All look-ups failed — nothing to write.")

    df = pd.DataFrame(merged_rows).sort_values("scopus_h", ascending=False)

    csv_path = Path(f"{file_stem}.csv")
    xlsx_path = Path(f"{file_stem}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"\n✅  Saved {len(df)} author records → {csv_path} & {xlsx_path}")


if __name__ == "__main__":
    main()