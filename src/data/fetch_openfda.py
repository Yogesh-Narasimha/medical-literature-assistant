"""Download FDA drug labels from the public openFDA API (no key needed).

Output: data/raw/drug_labels.jsonl  (one record per drug)
Run:    python -m src.data.fetch_openfda
"""
import json
import sys
import time

import requests
from tqdm import tqdm

from src.config import DRUGS, LABEL_SECTIONS, RAW_DIR

API = "https://api.fda.gov/drug/label.json"


def _get(params: dict, tries: int = 5) -> requests.Response:
    """GET with exponential backoff: openFDA sometimes returns transient 5xx."""
    for attempt in range(tries):
        try:
            resp = requests.get(API, params=params, timeout=30)
            if resp.status_code < 500:
                return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    return resp


CANDIDATES = 25   # labels examined per drug


def richness(label: dict) -> int:
    """How much indexable text a label has: sections present, weighted by length (capped per section)."""
    return sum(min(len(" ".join(label[s])), 20000) for s in LABEL_SECTIONS if s in label)


BRAND_CANDIDATES = 100   # labels examined per drug when looking for brand names


def candidates(drug: str, limit: int) -> list[dict]:
    """Up to `limit` labels whose generic or substance name matches the drug."""
    for field in ("openfda.generic_name", "openfda.substance_name"):
        resp = _get({"search": f'{field}:"{drug}"', "limit": limit})
        if resp.status_code == 404:          # openFDA returns 404 for "no match"
            continue
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results
    return []


def fetch_label(drug: str) -> dict | None:
    """Return the most complete label for a generic drug name, or None.

    openFDA holds many labels per drug (every manufacturer and repackager). Taking the first one gave
    thin OTC or repackager labels for some drugs, so look at several and keep the one with the most
    indexable text. Brand names (Zoloft for sertraline) sit on the brand-owner's label, which is rarely the
    richest, so they are collected from a deeper list of labels."""
    results = candidates(drug, CANDIDATES)
    if not results:
        return None
    best = max(results, key=richness)
    brands = brands_of(drug, candidates(drug, BRAND_CANDIDATES))
    return {**best, "openfda": {**best.get("openfda", {}), "brand_name": brands or
                                best.get("openfda", {}).get("brand_name", [])}}


def _is_single_ingredient(drug: str, generic_names: list[str]) -> bool:
    """True if the label's generic name is this drug (or a salt of it: "sertraline hydrochloride") and not a
    combination product. A combination's brand (Janumet under metformin) would map to the wrong drug."""
    d = drug.lower()
    if len(generic_names) != 1:
        return False
    g = generic_names[0].lower()
    rest = g[len(d):] if g.startswith(d) else None
    return rest is not None and "," not in g and " and " not in rest


def brands_of(drug: str, labels: list[dict]) -> list[str]:
    """Brand names (sorted, de-duplicated by case) of the single-ingredient labels for a drug."""
    names: dict[str, str] = {}
    for r in labels:
        o = r.get("openfda", {})
        if _is_single_ingredient(drug, o.get("generic_name", [])):
            for b in o.get("brand_name", []):
                names.setdefault(b.lower(), b)
    return sorted(names.values())


def to_record(drug: str, label: dict) -> dict:
    sections = {}
    for name in LABEL_SECTIONS:
        if name in label:
            text = " ".join(label[name]).strip()
            if text:
                sections[name] = text
    openfda = label.get("openfda", {})
    set_id = label.get("set_id", "")
    return {
        "doc_id": f"fda:{set_id or drug}",
        "source_type": "fda_label",
        "drug": drug,
        "brand_names": openfda.get("brand_name", []),
        "title": f"FDA label: {drug}",
        "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else "",
        "sections": sections,
    }


def refresh_brands():
    """Re-collect brand names for the labels already downloaded, without touching their text, so an index
    built from them stays consistent. Run: python -m src.data.fetch_openfda --brands-only"""
    path = RAW_DIR / "drug_labels.jsonl"
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    for rec in tqdm(records, desc="brand names"):
        try:
            brands = brands_of(rec["drug"], candidates(rec["drug"], BRAND_CANDIDATES))
        except requests.RequestException as e:
            print(f"  ! {rec['drug']}: {e}")
            continue
        rec["brand_names"] = brands or rec.get("brand_names", [])
        time.sleep(0.3)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    print(f"Updated brand names for {len(records)} labels")


MIN_LABEL_CHARS = 3000      # a usable label has at least this much indexable text
REQUIRED_SECTIONS = ("indications_and_usage",)
JUNK_WORDS = ("secretion", "cutaneous gland", "streptococcus", "extract", "pollen", "venom")


def usable(rec: dict) -> bool:
    """Quality gate for labels pulled from openFDA's name counts, which also contain homeopathic products,
    allergen extracts, and thin entries: keep a label only if it states what the drug is for and has real text."""
    text = sum(len(t) for t in rec.get("sections", {}).values())
    return (text >= MIN_LABEL_CHARS
            and all(sec in rec["sections"] for sec in REQUIRED_SECTIONS)
            and not any(w in rec["drug"].lower() for w in JUNK_WORDS))


def prune():
    """Drop labels that fail `usable`. Run: python -m src.data.fetch_openfda --prune"""
    path = RAW_DIR / "drug_labels.jsonl"
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    keep = [r for r in records if usable(r)]
    dropped = [r["drug"] for r in records if not usable(r)]
    with open(path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(f"Kept {len(keep)} labels, dropped {len(dropped)}: {', '.join(dropped)}")


def main():
    """Download labels for DRUGS. With --resume, keep the labels already on disk and fetch only the drugs that
    are missing (each label is written as it arrives, so an interrupted run loses nothing)."""
    if "--brands-only" in sys.argv:
        return refresh_brands()
    if "--prune" in sys.argv:
        return prune()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "drug_labels.jsonl"
    resume = "--resume" in sys.argv and out_path.exists()
    have = {json.loads(line)["drug"] for line in open(out_path, encoding="utf-8")} if resume else set()
    todo = [d for d in DRUGS if d not in have]
    found, missing = 0, []
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for drug in tqdm(todo, desc="openFDA labels"):
            try:
                label = fetch_label(drug)
            except requests.RequestException as e:
                print(f"  ! {drug}: {e}")
                label = None
            if label is None:
                missing.append(drug)
            else:
                f.write(json.dumps(to_record(drug, label), ensure_ascii=False) + chr(10))
                f.flush()
                found += 1
            time.sleep(0.3)  # stay well under the public rate limit
    print(f"Saved {found} labels to {out_path} ({len(have)} were already there)")
    if missing:
        print(f"No label found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
