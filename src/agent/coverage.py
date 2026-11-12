"""Coverage check: does the index actually contain the drug the user is asking about?

Dense retrieval will happily return chunks that merely MENTION an absent drug (lithium shows up inside the
interaction sections of naproxen, losartan, furosemide...). An answer built from those looks grounded but
answers a different question ("lithium side effects" -> "lithium interactions"). So before the agent runs:

  1. a short LLM call extracts the drug names in the question,
  2. each name is checked against the index: a label for it, a brand name, a drug class we know, or a drug
     named in the title of a document (PubMed abstracts carry the drug in the title),
  3. if NONE of the named drugs is covered -> refuse and say which drug is missing.

If extraction fails or returns nothing, the question simply proceeds (fail open): this check only adds a
refusal when it is confident a named drug is absent.
"""
import json
from functools import lru_cache

import ollama

from src.agent.tools import DRUG_CLASSES, _brand_index, _labels, normalize_drug
from src.config import LLM_MODEL, OLLAMA_HOST, PROCESSED_DIR

_client = ollama.Client(host=OLLAMA_HOST)

EXTRACT_PROMPT = """Extract the medication names from the sentence below. Output only a comma-separated list of names.

Sentence: {q}
Medication names:"""


@lru_cache(maxsize=1)
def _covered_terms() -> set[str]:
    """Lowercase names the index covers: label keys, label brand names, and document titles."""
    terms = set(_labels()) | set(_brand_index())
    with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            terms.add(json.loads(line)["title"].lower())
    return terms


def extract_drugs(question: str) -> list[str]:
    """Drug names the question actually contains. Names the model invented (not literally in the question)
    are dropped, since a made-up absent drug would cause a false refusal."""
    try:
        resp = _client.chat(model=LLM_MODEL, options={"temperature": 0.0, "num_predict": 30},
                            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(q=question)}])
    except Exception:
        return []
    raw = resp["message"]["content"].strip()
    q = question.lower()
    names = [n.strip(" .\"'").lower() for n in (raw.splitlines()[0] if raw else "").split(",")]
    return [n for n in names if 1 < len(n) < 40 and n not in ("none", "n/a") and n in q]


def is_covered(name: str) -> bool:
    name = normalize_drug(name)
    if name in DRUG_CLASSES or name in _labels():
        return True
    return any(name in term or (len(term) > 3 and term in name) for term in _covered_terms())


def missing_subject(question: str) -> list[str]:
    """Drug names to report as missing, but ONLY when no drug in the question is covered.

    "Side effects of lithium" -> ["lithium"] (the question is about a drug we have nothing on).
    "Lithium with ibuprofen"  -> []          (ibuprofen's label can legitimately discuss lithium).
    "Vitamin B12 with metformin" -> []       (metformin is covered and its label discusses B12)."""
    names = extract_drugs(question)
    if names and not any(is_covered(n) for n in names):
        return names
    return []
