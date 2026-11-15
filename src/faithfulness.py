"""Claim-level faithfulness check: does the cited source actually SUPPORT the sentence that cites it?

Citation validation (src/citations.py) only proves that [2] points at a real source. The model can still attach
[2] to a sentence the source does not say, or change a number on the way. This module checks each cited sentence
against the text of the source(s) it cites, two ways:

  1. Numbers. Every number in the sentence (500, 2.5, 10,000) must appear in a cited source. Cheap, exact, and
     aimed at the most dangerous error in this domain: a wrong dose or frequency.
  2. Entailment. A small natural-language-inference cross-encoder reads (source passage, sentence) and scores
     whether the passage entails or contradicts the sentence. Sources longer than the model's window are scored in
     overlapping windows. A sentence is flagged if the source contradicts it, or if it is neither entailed nor
     near-verbatim from the source (word overlap).

A sentence that fails a check is reported as "unsupported": the UI shows it as a warning; the answer text is
not changed. This is a screen, not a proof: see the README for what it catches and what it misses, measured by
`python -m eval.faithfulness`.

Needs torch + sentence-transformers (same as the reranker). The model loads lazily on first use.
"""
import re

from src.citations import _CITE, _FRAMING, _LEADING_CITE, _SENTENCE

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
# A sentence is flagged if a number is missing from its source, OR the source contradicts it, OR it is neither
# near-verbatim from the source nor entailed by it. The three cut-offs were chosen on the dev half of the benchmark.
ENTAIL_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.7
OVERLAP_THRESHOLD = 0.5
REPAIR_FROM, REPAIR_TO = 0.5, 0.8   # re-point a citation when overlap with the cited source is under the first
                                    # and overlap with another retrieved source is at least the second
WINDOW_WORDS, WINDOW_STEP = 150, 100
MIN_CLAIM_CHARS = 40

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
          "nine": "9", "ten": "10", "once": "1", "twice": "2"}

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "for", "with", "that", "this", "are", "was", "were", "can", "may", "has", "have", "its", "from",
         "not", "but", "also", "such", "which", "who", "been", "being", "than", "these", "those", "their", "there",
         "should", "would", "could", "will", "include", "includes", "including"}

_model = None
_entail_index, _contra_index = 1, 0


def _load():
    global _model, _entail_index, _contra_index
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(NLI_MODEL, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
        labels = {i: str(name).lower() for i, name in _model.model.config.id2label.items()}
        _entail_index = next(i for i, name in labels.items() if name.startswith("entail"))
        _contra_index = next(i for i, name in labels.items() if name.startswith("contra"))
    return _model


def _claims_raw(answer: str) -> list[tuple[str, str, list[int]]]:
    """(sentence as written, sentence without markers, cited numbers); see `claims`."""
    out = []
    for paragraph in answer.split("\n"):
        paragraph = _LEADING_CITE.sub(r"\1\2", paragraph)
        sentences = [s.strip(" -*•") for s in _SENTENCE.split(paragraph) if s.strip(" -*•")]
        marks = [[int(n) for n in _CITE.findall(s)] for s in sentences]
        lead = marks[0] if sentences and re.match(r"\s*(?:\[\d+\])+", sentences[0]) else []
        for i, s in enumerate(sentences):
            cited = marks[i] or next((m for m in marks[i + 1:] if m), []) or lead
            text = _CITE.sub("", s).strip()
            if cited and len(text) >= MIN_CLAIM_CHARS and not _FRAMING.search(text):
                text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", text))   # "embolism ." -> "embolism."
                out.append((s, text, sorted(set(cited))))
    return out


def claims(answer: str) -> list[tuple[str, list[int]]]:
    """(sentence without markers, source numbers it cites) for every factual sentence that has a citation.

    A citation covers the sentences before it in the same paragraph (models often cite once at the end), and a
    marker at the very start of a paragraph covers the paragraph. Sentences with no citation at all are not
    returned here: `uncited_sentences` already reports those."""
    return [(text, cited) for _, text, cited in _claims_raw(answer)]


def repair_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Re-point citations that name the wrong source.

    Measured on 79 real answers: 1 in 5 cited sentences shared under half of its words with the source it cited,
    and about half of those were near-verbatim from ANOTHER retrieved source (the model numbers list items [1],
    [2], [3] in order instead of citing where each came from). When a sentence carries its own marker, barely
    overlaps the cited source, and is near-verbatim (numbers included) in exactly one better source, the marker is
    replaced. Nothing else in the answer is touched. Returns (answer, [{sentence, was, now}])."""
    texts = [s["text"] for s in sources]
    fixes = []
    for raw, text, cited in _claims_raw(answer):
        own = [int(n) for n in _CITE.findall(raw)]
        if not own or raw not in answer or any(not 1 <= n <= len(texts) for n in cited):
            continue
        if overlap(text, [texts[n - 1] for n in cited]) >= REPAIR_FROM:
            continue
        scored = sorted(((overlap(text, [t]), i + 1) for i, t in enumerate(texts) if i + 1 not in cited), reverse=True)
        if not scored or scored[0][0] < REPAIR_TO or unsupported_numbers(text, [texts[scored[0][1] - 1]]):
            continue
        new = scored[0][1]
        fixed = _CITE.sub("", raw).rstrip()
        fixed = (fixed[:-1].rstrip() + f" [{new}]" + fixed[-1]) if fixed[-1:] in ".!?" else fixed + f" [{new}]"
        answer = answer.replace(raw, fixed, 1)
        fixes.append({"sentence": text, "was": cited, "now": [new]})
    return answer, fixes


def _numbers(text: str) -> set[str]:
    found = {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}
    return {n[:-2] if n.endswith(".0") else n for n in found}


def unsupported_numbers(claim: str, premises: list[str]) -> list[str]:
    """Numbers in the claim that appear in none of the cited sources. Number words in a source ("twice daily")
    count as their digit, so "2 times a day" is not flagged against "twice a day"."""
    have = set()
    for p in premises:
        have |= _numbers(p)
        have |= {_WORDS[w] for w in re.findall(r"[a-z]+", p.lower()) if w in _WORDS}
    return sorted(n for n in _numbers(claim) if n not in have)


def _windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WINDOW_WORDS:
        return [text]
    return [" ".join(words[i:i + WINDOW_WORDS]) for i in range(0, max(len(words) - WINDOW_WORDS + WINDOW_STEP, 1),
                                                                 WINDOW_STEP)]


def nli(claim: str, premises: list[str]) -> tuple[float, float]:
    """(entailment, contradiction): the highest probability of each, over every window of every cited source."""
    pairs = [(w, claim) for p in premises for w in _windows(p)]
    if not pairs:
        return 0.0, 0.0
    probs = _load().predict(pairs, apply_softmax=True, show_progress_bar=False)
    return float(max(row[_entail_index] for row in probs)), float(max(row[_contra_index] for row in probs))


def entailment(claim: str, premises: list[str]) -> float:
    return nli(claim, premises)[0]


def overlap(claim: str, premises: list[str]) -> float:
    """Share of the claim's content words found in the best cited source. The model mostly copies label text, and
    the NLI model scores copied list fragments ("for the treatment of DVT and PE...") as unsupported because they
    are not full sentences; near-verbatim text is supported by definition."""
    words = {w for w in _TOKEN.findall(claim.lower()) if len(w) > 2 and w not in _STOP}
    if not words:
        return 1.0
    return max(len(words & set(_TOKEN.findall(p.lower()))) / len(words) for p in premises)


def verdict(claim: str, premises: list[str]) -> dict:
    """All signals for one claim, and whether it is flagged."""
    numbers = unsupported_numbers(claim, premises)
    entail, contra = nli(claim, premises)
    share = overlap(claim, premises)
    flagged = bool(numbers) or contra >= CONTRA_THRESHOLD or (share < OVERLAP_THRESHOLD and entail < ENTAIL_THRESHOLD)
    return {"flagged": flagged, "entailment": round(entail, 2), "contradiction": round(contra, 2),
            "overlap": round(share, 2), "numbers": numbers}


def check(answer: str, sources: list[dict]) -> list[dict]:
    """Cited sentences the cited sources do not support: [{sentence, cited, entailment, contradiction, overlap, numbers}]."""
    flagged = []
    for sentence, cited in claims(answer):
        premises = [sources[n - 1]["text"] for n in cited if 1 <= n <= len(sources)]
        if not premises:
            continue
        v = verdict(sentence, premises)
        if v.pop("flagged"):
            flagged.append({"sentence": sentence, "cited": cited, **v})
    return flagged
