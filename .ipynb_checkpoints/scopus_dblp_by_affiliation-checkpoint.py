#!/usr/bin/env python3
"""
scopus_dblp_by_affiliation.py
=============================
Harvest author-level metrics from Scopus *and* DBLP for **all** researchers
affiliated with a user-supplied institution.

Outputs
-------
 - metrics_<slug>.csv   (comma-separated summary)
 - metrics_<slug>.xlsx  (Excel workbook, single sheet)
 - per-author raw JSON  (Scopus + DBLP data cached locally)
"""

###############################################################################
# 0. Imports & Constants
###############################################################################
import argparse, os, sys, json, time, math
from pathlib import Path
from typing import Optional, Dict, List

import requests, pandas as pd
from tqdm import tqdm
from slugify import slugify               # pip install python-slugify
from pybliometrics.scopus import AuthorRetrieval

# ─── ENVIRONMENT CHECK ────────────────────────────────────────────────────────
SCOPUS_KEY = os.getenv("SCOPUS_API_KEY")
if not SCOPUS_KEY:
    sys.exit("❌  Please set the SCOPUS_API_KEY environment variable and retry.")

S2_KEY = os.getenv("S2_API_KEY")          # optional
S2_HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}

# ─── RATE-AWARE SAFE GET ──────────────────────────────────────────────────────
def _safe_get(url: str,
              params: Optional[Dict] = None,
              headers: Optional[Dict] = None,
              tries: int = 5,
              pause: float = 0.6):
    """GET with exponential back-off for 429/5xx responses."""
    for attempt in range(1, tries + 1):
        r = requests.get(url, params=params or {}, headers=headers or {})
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            sleep = pause * 2 ** attempt
            time.sleep(sleep)
        else:
            r.raise_for_status()
    r.raise_for_status()                  # if still failing after retries

###############################################################################
# 1. Scopus helpers
###############################################################################
def scopus_affiliation_search(affil: str,
                              max_authors: int = 9999) -> List[Dict]:
    """
    Return list of {id, name} dicts for all authors linked to `affil`.
    """
    authors = []
    url = "https://api.elsevier.com/content/search/author"
    count_per_call = 200                  # Scopus max per page
    start = 0

    query = f'AFFIL("{affil}")'           # exact-phrase affiliation query

    while len(authors) < max_authors:
        params = {
            "apiKey": SCOPUS_KEY,
            "query": query,
            "start": start,
            "count": count_per_call,
            "view": "STANDARD",
        }
        r = _safe_get(url, params=params)
        entries = r.json().get("search-results", {}).get("entry", [])
        if not entries:
            break
        for e in entries:
            authors.append({
                "scopus_id": e["dc:identifier"].split(":")[1],
                "name": e["preferred-name"]["given-name"] + " " +
                        e["preferred-name"]["surname"]
            })
            if len(authors) >= max_authors:
                break
        start += count_per_call
    return authors


def scopus_profile(author_id: str) -> Dict:
    """
    Download full profile (cached on disk). Returns essential metrics dict.
    """
    cache_file = Path(f"_cache/scopus_{author_id}.json")
    cache_file.parent.mkdir(exist_ok=True)
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        ar = AuthorRetrieval(author_id, refresh=True)
        data = ar._json
        cache_file.write_text(json.dumps(data, indent=2))

    return {
        "name": data["author-profile"]["preferred-name"]["given-name"] + " " +
                data["author-profile"]["preferred-name"]["surname"],
        "scopus_id": author_id,
        "scopus_h": int(data["author-profile"]["coredata"]["h-index"]),
        "scopus_docs": int(data["author-profile"]["coredata"]["document-count"]),
        "affiliation": (data["author-profile"]
                            .get("affiliation-current", {})
                            .get("affiliation-name")),
    }

###############################################################################
# 2. DBLP helpers
###############################################################################
def dblp_pid(name: str) -> Optional[str]:
    r = _safe_get("https://dblp.org/search/author/api",
                  params={"q": name, "format": "json"})
    hits = r.json()["result"]["hits"]["hit"]
    return hits[0]["info"]["authorid"] if hits else None


def dblp_publications(pid: str) -> List[Dict]:
    cache_file = Path(f"_cache/dblp_{pid}.json")
    cache_file.parent.mkdir(exist_ok=True)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    r = _safe_get(f"https://dblp.org/pid/{pid}.json")
    records = r.json()["result"]["hits"]["hit"]
    cache_file.write_text(json.dumps(records, indent=2))
    return records


def semantic_scholar_citations(doi: str) -> int:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    r = _safe_get(url, params={"fields": "citationCount"}, headers=S2_HEADERS)
    return int(r.json().get("citationCount", 0))


def dblp_h_index(records: List[Dict]) -> int:
    citations = []
    for rec in records:
        doi = rec["info"].get("doi")
        if doi:
            citations.append(semantic_scholar_citations(doi))
            time.sleep(0.1)               # be polite
    citations.sort(reverse=True)
    return sum(c >= i + 1 for i, c in enumerate(citations))


###############################################################################
# 3. Main orchestration per author
###############################################################################
def process_author(a: Dict) -> Dict:
    """
    Takes {'scopus_id', 'name'} dict → returns merged metrics record.
    """
    prof = scopus_profile(a["scopus_id"])

    pid = dblp_pid(prof["name"])
    if pid:
        records = dblp_publications(pid)
        h_dblp = dblp_h_index(records)
        dblp_docs = len(records)
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
        "dblp_docs": dblp_docs,
    }

###############################################################################
# 4. Command-line interface
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Harvest Scopus + DBLP author metrics for a whole institution"
    )
    parser.add_argument("affiliation", help="Exact institution name in Scopus")
    parser.add_argument("--max-authors", type=int, default=500,
                        help="Hard cap to avoid huge API bills (default 500)")
    args = parser.parse_args()

    affil = args.affiliation
    slug = slugify(affil)                 # nice file stem

    print(f"🔍  Querying Scopus for authors at “{affil}”…")
    author_list = scopus_affiliation_search(affil, args.max_authors)
    if not author_list:
        sys.exit("Nothing found—check the spelling of the affiliation string.")

    rows = []
    for a in tqdm(author_list, desc="Processing authors"):
        try:
            rows.append(process_author(a))
        except Exception as exc:
            print(f"⚠️  {a['name']} skipped: {exc}")

    if not rows:
        sys.exit("No author processed successfully—stopping.")

    df = pd.DataFrame(rows).sort_values("scopus_h", ascending=False)

    out_csv = f"metrics_{slug}.csv"
    out_xlsx = f"metrics_{slug}.xlsx"
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"\n✅  Saved {len(df)} records → {out_csv} & {out_xlsx}")

if __name__ == "__main__":
    main()
