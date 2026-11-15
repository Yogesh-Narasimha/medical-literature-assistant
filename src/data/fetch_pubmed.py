"""Download PubMed abstracts through NCBI E-utilities (no key needed).

Output: data/raw/pubmed_abstracts.jsonl  (one record per article)
Run:    python -m src.data.fetch_pubmed
"""
import json
import sys
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from src.config import PUBMED_PER_QUERY, PUBMED_QUERIES, RAW_DIR

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PAUSE = 0.4  # NCBI allows 3 requests/second without a key


def search(query: str, n: int) -> list[str]:
    params = {"db": "pubmed", "term": f"{query} AND hasabstract", "retmax": n,
              "retmode": "json", "sort": "relevance"}
    resp = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch(pmids: list[str]) -> list[dict]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    resp = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID", default="")
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() \
            if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")
            if text:
                parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts)
        year = art.findtext(".//PubDate/Year", default="")
        if pmid and abstract:
            articles.append({"pmid": pmid, "title": title, "abstract": abstract, "year": year})
    return articles


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "pubmed_abstracts.jsonl"
    seen, done_queries = set(), set()
    resume = "--resume" in sys.argv and out_path.exists()
    if resume:                        # keep what is on disk, fetch only queries not yet run
        for line in open(out_path, encoding="utf-8"):
            rec = json.loads(line)
            seen.add(rec["doc_id"].split(":", 1)[1])
            done_queries.add(rec["query"])
    count = 0
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for query in tqdm([q for q in PUBMED_QUERIES if q not in done_queries], desc="PubMed queries"):
            try:
                pmids = [p for p in search(query, PUBMED_PER_QUERY) if p not in seen]
                time.sleep(PAUSE)
                if not pmids:
                    continue
                for a in fetch(pmids):
                    seen.add(a["pmid"])
                    record = {
                        "doc_id": f"pubmed:{a['pmid']}",
                        "source_type": "pubmed",
                        "query": query,
                        "title": a["title"],
                        "year": a["year"],
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/",
                        "sections": {"abstract": a["abstract"]},
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                time.sleep(PAUSE)
            except (requests.RequestException, ET.ParseError) as e:
                print(f"  ! '{query}': {e}")
    print(f"Saved {count} abstracts to {out_path}")


if __name__ == "__main__":
    main()
