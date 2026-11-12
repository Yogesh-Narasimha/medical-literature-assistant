"""Tools the agent can call. Each returns (sources, info).

Sources are plain dicts; the graph assigns each a global citation number [n].
`info` is the message shown to the LLM when no sources came back.
"""
import difflib
import json
import re
from functools import lru_cache

from src.config import LABEL_SECTIONS, LABEL_SNIPPET_CHARS, RAW_DIR
from src.retrieval.corrective import corrective_search
from src.retrieval.hybrid import HybridRetriever, default_retriever

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = default_retriever()
    return _retriever


@lru_cache(maxsize=1)
def _labels() -> dict[str, dict]:
    with open(RAW_DIR / "drug_labels.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return {r["drug"].lower(): r for r in recs}


# Brand and lay names -> the generic names used as keys in the label store.
SYNONYMS = {
    "paracetamol": "acetaminophen", "tylenol": "acetaminophen", "panadol": "acetaminophen",
    "coumadin": "warfarin", "jantoven": "warfarin", "eliquis": "apixaban", "plavix": "clopidogrel",
    "advil": "ibuprofen", "motrin": "ibuprofen", "aleve": "naproxen", "glucophage": "metformin",
    "januvia": "sitagliptin", "jardiance": "empagliflozin", "lantus": "insulin glargine",
    "lipitor": "atorvastatin", "crestor": "rosuvastatin", "norvasc": "amlodipine", "zestril": "lisinopril",
    "prinivil": "lisinopril", "cozaar": "losartan", "lopressor": "metoprolol", "toprol": "metoprolol",
    "prilosec": "omeprazole", "nexium": "omeprazole", "protonix": "pantoprazole", "synthroid": "levothyroxine",
    "deltasone": "prednisone", "cipro": "ciprofloxacin", "zithromax": "azithromycin", "vibramycin": "doxycycline",
    "zoloft": "sertraline", "prozac": "fluoxetine", "lexapro": "escitalopram", "wellbutrin": "bupropion",
    "xanax": "alprazolam", "neurontin": "gabapentin", "keppra": "levetiracetam", "singulair": "montelukast",
    "ventolin": "albuterol", "proair": "albuterol", "zyrtec": "cetirizine", "lasix": "furosemide",
    "aldactone": "spironolactone", "entresto": "sacubitril and valsartan", "ultram": "tramadol",
    "aspirin": "aspirin", "bayer": "aspirin",
}


@lru_cache(maxsize=1)
def _brand_index() -> dict[str, str]:
    """Brand names that appear in the downloaded labels themselves (lowercase brand -> generic)."""
    idx = {}
    for generic, rec in _labels().items():
        for b in rec.get("brand_names", []):
            idx.setdefault(b.lower(), generic)
    return idx


# Words the model puts in a drug slot when it means "no second drug".
PLACEHOLDERS = {"nil", "none", "null", "n/a", "na", "any", "any drug", "any other drug", "other drugs",
                "other drug", "all", "all drugs", "everything", "unknown", "other", "others", ""}

# Drug classes people name instead of a drug -> member drugs that have a label in the store.
DRUG_CLASSES = {
    "blood thinner": ["warfarin", "apixaban", "clopidogrel"], "blood thinners": ["warfarin", "apixaban", "clopidogrel"],
    "anticoagulant": ["warfarin", "apixaban"], "anticoagulants": ["warfarin", "apixaban"],
    "antiplatelet": ["clopidogrel", "aspirin"], "nsaid": ["ibuprofen", "naproxen", "aspirin"],
    "nsaids": ["ibuprofen", "naproxen", "aspirin"], "painkiller": ["ibuprofen", "naproxen", "acetaminophen"],
    "ssri": ["sertraline", "fluoxetine", "escitalopram"], "antidepressant": ["sertraline", "fluoxetine", "escitalopram"],
    "statin": ["atorvastatin", "rosuvastatin"], "statins": ["atorvastatin", "rosuvastatin"],
    "ace inhibitor": ["lisinopril"], "diuretic": ["furosemide", "hydrochlorothiazide", "spironolactone"],
    "antibiotic": ["amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline"],
    "proton pump inhibitor": ["omeprazole", "pantoprazole"], "ppi": ["omeprazole", "pantoprazole"],
}


def normalize_drug(name: str | None) -> str:
    """Map a brand/lay/misspelt drug name to the generic key used by the label store."""
    n = str(name or "").strip().lower()
    if n in _labels():
        return n
    hit = SYNONYMS.get(n) or _brand_index().get(n)
    if hit:
        return hit
    if n and n not in DRUG_CLASSES and n not in PLACEHOLDERS:
        words = lambda s: set(re.findall(r"[a-z0-9]+", s)) - {"and"}          # noqa: E731
        # "sacubitril valsartan" == "sacubitril and valsartan": same words once "and" is ignored
        same = next((k for k in _labels() if words(k) == words(n)), None)
        if same:
            return same
        # "ibuprofin" -> "ibuprofen": a close spelling of exactly one known drug
        close = difflib.get_close_matches(n, list(_labels()), n=1, cutoff=0.85)
        if close:
            return close[0]
    return n


def search_documents(query: str) -> tuple[list[dict], str]:
    """Semantic search over FDA label sections and PubMed abstracts."""
    hits, correction = corrective_search(query, get_retriever(), 5)
    sources = [{
        "key": h.chunk["chunk_id"], "title": h.chunk["title"], "section": h.chunk["section"],
        "url": h.chunk.get("url", ""), "text": h.chunk["text"], "score": round(h.score, 3),
    } for h in hits]
    if not sources:
        return [], ("No relevant documents were found for this query, even after rewriting it "
                    f"({correction.note()}). Tell the user you could not find this in the indexed documents.")
    return sources, correction.note()


def lookup_drug_label(drug: str, section: str = "drug_interactions") -> tuple[list[dict], str]:
    """Exact lookup of one section of one drug's FDA label."""
    rec = _labels().get(normalize_drug(drug))
    if rec is None:
        return [], f"No label for '{drug}'. Available drugs: {', '.join(sorted(_labels()))}"
    section = str(section or "drug_interactions").strip().lower()
    if section not in rec["sections"]:
        # Many OTC labels only have "warnings"; fall back rather than fail.
        alt = next((x for x in ("warnings_and_cautions", "warnings") if x in rec["sections"]), None)
        if alt is None:
            return [], f"Label for {drug} has no section '{section}'. Available sections: {', '.join(rec['sections'])}"
        section = alt
    return [{
        "key": f"{rec['doc_id']}#{section}#label", "title": rec["title"], "section": section,
        "url": rec["url"], "text": rec["sections"][section][:LABEL_SNIPPET_CHARS], "score": None,
    }], ""


# Terms under which a label may refer to a drug (class names, plain-language names).
ALIASES = {
    "ibuprofen": ["nsaid", "ibuprofen", "salicylate"], "naproxen": ["nsaid", "naproxen", "salicylate"],
    "aspirin": ["aspirin", "salicylate", "nsaid"], "warfarin": ["warfarin", "anticoagulant", "blood thinn", "coumadin"],
    "apixaban": ["apixaban", "anticoagulant", "blood thinn", "factor xa"],
    "clopidogrel": ["clopidogrel", "antiplatelet", "p2y12"],
    "sertraline": ["sertraline", "ssri", "serotonergic", "serotonin"], "fluoxetine": ["fluoxetine", "ssri", "serotonergic", "serotonin"],
    "escitalopram": ["escitalopram", "ssri", "serotonergic", "serotonin"], "tramadol": ["tramadol", "opioid", "serotonergic"],
    "lisinopril": ["lisinopril", "ace inhibitor", "ace-inhibitor"], "losartan": ["losartan", "angiotensin receptor"],
    "metformin": ["metformin"], "omeprazole": ["omeprazole", "proton pump"], "pantoprazole": ["pantoprazole", "proton pump"],
}
_SPLIT = re.compile(r"(?<=[.;])\s+")


def _mentions(rec: dict, terms: list[str]) -> list[dict]:
    out = []
    for section in ("drug_interactions", "warnings_and_cautions", "warnings", "contraindications"):
        text = rec["sections"].get(section, "")
        hits = [t.strip() for t in _SPLIT.split(text) if any(k in t.lower() for k in terms)]
        if hits:
            out.append({
                "key": f"{rec['doc_id']}#{section}#interaction", "title": rec["title"], "section": section,
                "url": rec["url"], "text": " ".join(hits)[:LABEL_SNIPPET_CHARS], "score": None,
            })
            if section == "drug_interactions":
                break    # the interactions section is authoritative when it mentions the other drug
    return out


def check_interaction(drug_a: str, drug_b: str) -> tuple[list[dict], str]:
    """Find what each drug's FDA label says about the other drug (or its class)."""
    if not drug_a or not drug_b:
        # Only one drug named: show that drug's interactions section instead.
        return lookup_drug_label(drug_a or drug_b or "", "drug_interactions")
    a, b = normalize_drug(drug_a), normalize_drug(drug_b)
    labels, sources = _labels(), []
    # One name is not a drug or class (the model often fills the blank with "nil", "any", "other drugs"):
    # fall back to the known drug's own interactions section instead of failing.
    known = lambda x: x in labels or x in DRUG_CLASSES          # noqa: E731
    if known(a) != known(b):
        have, other = (a, b) if known(a) else (b, a)
        if other in PLACEHOLDERS:
            return lookup_drug_label(have, "drug_interactions")
        if have in labels:
            # A real drug name we have no label for ("lithium"): does the label we DO have mention it?
            found = _mentions(labels[have], [other])
            if found:
                return found, ""
            return [], (f"The {have} label does not mention {other}, and there is no label for {other} in the "
                        "index. Do not assume the combination is safe; say the indexed documents have no "
                        "information on it.")
        return lookup_drug_label(have, "drug_interactions")
    # A drug class instead of a drug ("Tylenol with a blood thinner"): check each member that has a label.
    for one, cls_name in ((a, b), (b, a)):
        members = DRUG_CLASSES.get(cls_name.lower())
        if members and one in labels:
            merged: list[dict] = []
            for m in members:
                if m in labels and m != one:
                    found, _ = check_interaction(one, m)
                    merged += [s for s in found if all(s["key"] != t["key"] for t in merged)]
            return merged[:4], "" if merged else f"No label text links {one} with {cls_name}."
    for me, other in ((a, b), (b, a)):
        if me not in labels:
            return [], f"No label for '{me}'. Available drugs: {', '.join(sorted(labels))}"
        rec = labels[me]
        terms = ALIASES.get(other, [other]) + [x.lower() for x in labels.get(other, {}).get("brand_names", [])]
        sources += _mentions(rec, terms)
    return sources, "" if sources else f"Neither label mentions {a} or {b} with respect to the other."


TOOL_FUNCS = {"search_documents": search_documents, "lookup_drug_label": lookup_drug_label,
              "check_interaction": check_interaction}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Semantic search over FDA drug labels and PubMed abstracts. Use for general medical "
                       "questions, side effects, evidence, mechanisms, or when no specific drug+section is named.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query in plain English"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_interaction",
        "description": "Check what the FDA labels of TWO named drugs say about each other. Use this whenever "
                       "the question asks if two drugs can be taken together or interact.",
        "parameters": {"type": "object", "properties": {
            "drug_a": {"type": "string", "description": "First generic drug name, e.g. ibuprofen"},
            "drug_b": {"type": "string", "description": "Second generic drug name, e.g. warfarin"}},
            "required": ["drug_a", "drug_b"]}}},
    {"type": "function", "function": {
        "name": "lookup_drug_label",
        "description": "Fetch one exact section of one drug's FDA label. Use ONLY for boxed warning, dosage, "
                       "contraindications or the interactions list of a NAMED drug. Not for side effects, "
                       "mechanisms, indications or general questions: use search_documents for those.",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string", "description": "Generic drug name, e.g. warfarin"},
            "section": {"type": "string", "enum": LABEL_SECTIONS, "description": "Label section"}},
            "required": ["drug", "section"]}}},
]
