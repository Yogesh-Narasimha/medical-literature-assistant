"""Tools the agent can call. Each returns (sources, info).

Sources are plain dicts; the graph assigns each a global citation number [n].
`info` is the message shown to the LLM when no sources came back.
"""
import json
import re
from functools import lru_cache

from src.config import LABEL_SECTIONS, LABEL_SNIPPET_CHARS, RAW_DIR
from src.retrieval.dense import DenseRetriever

_retriever: DenseRetriever | None = None


def get_retriever() -> DenseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever()
    return _retriever


@lru_cache(maxsize=1)
def _labels() -> dict[str, dict]:
    with open(RAW_DIR / "drug_labels.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return {r["drug"].lower(): r for r in recs}


def search_documents(query: str) -> tuple[list[dict], str]:
    """Semantic search over FDA label sections and PubMed abstracts."""
    hits = get_retriever().search(query, 5)
    sources = [{
        "key": h.chunk["chunk_id"], "title": h.chunk["title"], "section": h.chunk["section"],
        "url": h.chunk.get("url", ""), "text": h.chunk["text"], "score": round(h.score, 3),
    } for h in hits]
    return sources, "No results."


def lookup_drug_label(drug: str, section: str = "drug_interactions") -> tuple[list[dict], str]:
    """Exact lookup of one section of one drug's FDA label."""
    rec = _labels().get(str(drug or "").strip().lower())
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
    a, b = str(drug_a).strip().lower(), str(drug_b).strip().lower()
    labels, sources = _labels(), []
    for me, other in ((a, b), (b, a)):
        if me not in labels:
            return [], f"No label for '{me}'. Available drugs: {', '.join(sorted(labels))}"
        rec = labels[me]
        terms = ALIASES.get(other, [other]) + [x.lower() for x in labels.get(other, {}).get("brand_names", [])]
        sources += _mentions(rec, terms)
    return sources, f"Neither label mentions {a} or {b} with respect to the other."


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
        "description": "Fetch one exact section of one drug's FDA label. Use for drug interactions, dosing, "
                       "contraindications or boxed warnings of a NAMED drug. For an interaction between two "
                       "drugs, call it once per drug.",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string", "description": "Generic drug name, e.g. warfarin"},
            "section": {"type": "string", "enum": LABEL_SECTIONS, "description": "Label section"}},
            "required": ["drug", "section"]}}},
]
