"""Citation parsing and checking, shared by the plain-RAG answerer and the agent."""
import re

_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|(?<=\]|[.!?])\s+(?=[A-Z])")
_LEADING_CITE = re.compile(r"([.!?])\s+((?:\[\d+\])+)")   # ". [1] Next" -> ".[1] Next"
# Sentences that are advice or framing rather than factual claims need no citation.
_FRAMING = re.compile(r"consult|healthcare|health care|clinician|pharmacist|doctor|not medical advice|"
                      r"i could not find", re.I)


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) sorted citation numbers found as [n] in the answer.
    Valid means 1 <= n <= n_sources; anything else is a made-up source number."""
    used = sorted({int(n) for n in _CITE.findall(answer)})
    valid = [n for n in used if 1 <= n <= n_sources]
    return valid, [n for n in used if n not in valid]


def has_citation(answer: str) -> bool:
    return bool(_CITE.search(answer))


def strip_invalid(answer: str, n_sources: int) -> str:
    """Remove [n] markers that point at non-existent sources (they would mislead the reader)."""
    return _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)


def strip_markers(answer: str) -> str:
    """The answer text with every [n] marker removed (to measure how much real content is left)."""
    return _CITE.sub("", answer).strip()


def uncited_sentences(answer: str) -> list[str]:
    """Factual-looking sentences with no [n] marker, so the UI can flag them as unverified.

    Models often cite once at the end of a paragraph or bullet, so a citation covers the sentences
    before it in the same paragraph. Sentences AFTER the last citation of their paragraph are the
    ones the model made without pointing at any source."""
    out = []
    for paragraph in answer.split("\n"):
        paragraph = _LEADING_CITE.sub(r"\1\2", paragraph)   # attach a marker to the sentence before it
        sentences = [s.strip(" -*•") for s in _SENTENCE.split(paragraph) if s.strip(" -*•")]
        last_cited = max((i for i, s in enumerate(sentences) if _CITE.search(s)), default=-1)
        leading = bool(sentences) and bool(re.match(r"\s*(?:\[\d+\])+", sentences[0]))
        for i, s in enumerate(sentences):
            covered = i <= last_cited or leading   # a leading [n] ("[1] Sentence. Sentence.") covers its paragraph
            if len(s) > 40 and not covered and not _FRAMING.search(s):
                out.append(s)
    return out
