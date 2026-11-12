"""Citation parsing shared by the plain-RAG answerer and the agent."""
import re

_CITE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) sorted citation numbers found as [n] in the answer.
    Valid means 1 <= n <= n_sources; anything else is a made-up source number."""
    used = sorted({int(n) for n in _CITE.findall(answer)})
    valid = [n for n in used if 1 <= n <= n_sources]
    return valid, [n for n in used if n not in valid]


def has_citation(answer: str) -> bool:
    return bool(_CITE.search(answer))
