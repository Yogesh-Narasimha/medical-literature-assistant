"""Unit tests. No Ollama, index or network needed: everything runs on tiny fake data.

Run:  python -m pytest -q
"""
import pytest

from src.citations import has_citation, parse_citations
from src.ingest.chunk import chunk_document, split_words
from src.retrieval import corrective
from src.retrieval.dense import Hit
from src.retrieval.hybrid import BM25Retriever, HybridRetriever, tokenize


def chunk(cid, text, drug="", section="warnings", source="fda_label", title="t"):
    return {"chunk_id": cid, "doc_id": cid.split("#")[0], "source_type": source, "drug": drug,
            "title": title, "section": section, "url": "", "text": text}


# ---------------------------------------------------------------- chunking
def test_short_text_is_one_chunk():
    assert split_words("a b c", 200, 40) == ["a b c"]
    assert split_words("", 200, 40) == []


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(500)]
    chunks = split_words(" ".join(words), 100, 20)
    assert len(chunks) > 4
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()          # consecutive chunks share words
    assert set(" ".join(chunks).split()) == set(words)         # nothing dropped


def test_chunk_snaps_to_sentence_end():
    text = " ".join(["word"] * 85 + ["end."] + ["more"] * 100)
    first = split_words(text, 100, 10)[0]
    assert first.endswith("end.")


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_words("a b c", 10, 10)


def test_chunk_document_ids_are_per_section():
    doc = {"doc_id": "fda:1", "source_type": "fda_label", "drug": "x", "title": "T",
           "sections": {"warnings": "w " * 10, "adverse_reactions": "a " * 10}}
    ids = [c["chunk_id"] for c in chunk_document(doc, 200, 40)]
    assert ids == ["fda:1#warnings#0", "fda:1#adverse_reactions#0"]


# --------------------------------------------------------------- citations
def test_parse_citations_valid_and_invalid():
    assert parse_citations("Bleeding risk [1][3]. Also [2], and made up [9] [0].", 3) == ([1, 2, 3], [0, 9])
    assert parse_citations("no markers here", 3) == ([], [])
    assert has_citation("see [2]") and not has_citation("see 2 or (2)")


# -------------------------------------------------------------------- BM25
def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What are the side effects of Metformin?") == ["side", "effects", "metformin"]


def test_bm25_ranks_exact_term_first():
    chunks = [chunk("a#s#0", "warfarin interacts with many drugs", drug="warfarin"),
              chunk("b#s#0", "metformin causes nausea", drug="metformin"),
              chunk("c#s#0", "completely unrelated text about weather")]
    hits = BM25Retriever(chunks).search("warfarin interaction", 3)
    assert hits[0].chunk["chunk_id"] == "a#s#0"
    assert BM25Retriever(chunks).search("zzzunknown", 3) == []


# ------------------------------------------------------------------ hybrid
class FakeDense:
    def __init__(self, chunks, scores):
        self.chunks, self._scores = chunks, scores

    def search(self, query, k):
        order = sorted(range(len(self.chunks)), key=lambda i: -self._scores[i])[:k]
        return [Hit(self._scores[i], self.chunks[i]) for i in order]


def test_hybrid_keeps_same_chunk_under_two_drugs_separate():
    """A combination-drug label is stored once per drug with the same chunk_id; fusion must not merge them."""
    chunks = [chunk("L#boxed#0", "lactic acidosis warning", drug="sitagliptin"),
              chunk("L#boxed#0", "lactic acidosis warning", drug="metformin")]
    hits = HybridRetriever(FakeDense(chunks, [0.9, 0.8])).search("metformin lactic acidosis", 5)
    assert {h.chunk["drug"] for h in hits} == {"sitagliptin", "metformin"}


def test_drug_boost_promotes_named_drug():
    chunks = [chunk("a#s#0", "interaction bleeding", drug="aspirin"),
              chunk("b#s#0", "interaction bleeding", drug="warfarin")]
    dense = FakeDense(chunks, [0.80, 0.79])
    assert HybridRetriever(dense).search("interaction bleeding", 2)[0].chunk["drug"] == "aspirin"
    boosted = HybridRetriever(dense, drug_boost=0.01).search("warfarin interaction bleeding", 2)
    assert boosted[0].chunk["drug"] == "warfarin"


def test_hybrid_score_is_dense_cosine_so_refusal_threshold_keeps_meaning():
    chunks = [chunk("a#s#0", "alpha beta", drug="x")]
    (hit,) = HybridRetriever(FakeDense(chunks, [0.77])).search("alpha", 1)
    assert hit.score == 0.77


# --------------------------------------------------------------- corrective
def hit(score, cid="a#s#0"):
    return Hit(score, chunk(cid, "text"))


def test_grade_thresholds():
    assert corrective.grade([hit(0.80)]) == "correct"
    assert corrective.grade([hit(0.68)]) == "ambiguous"
    assert corrective.grade([hit(0.50)]) == "incorrect"
    assert corrective.grade([]) == "incorrect"


class FakeRetriever:
    def __init__(self, by_query):
        self.by_query = by_query

    def search(self, q, k):
        return self.by_query.get(q, [])


def test_out_of_domain_is_refused_without_rewrite(monkeypatch):
    monkeypatch.setattr(corrective, "rewrite_query", lambda q: pytest.fail("must not rewrite"))
    hits, corr = corrective.corrective_search("fix a bike", FakeRetriever({"fix a bike": [hit(0.5)]}))
    assert hits == [] and corr.grade_before == "incorrect"


def test_ambiguous_query_is_rewritten_and_merged(monkeypatch):
    monkeypatch.setattr(corrective, "rewrite_query", lambda q: "acetaminophen anticoagulant")
    r = FakeRetriever({"tylenol blood thinner": [hit(0.68, "a#s#0")],
                       "acetaminophen anticoagulant": [hit(0.80, "b#s#0")]})
    hits, corr = corrective.corrective_search("tylenol blood thinner", r)
    assert corr.grade_before == "ambiguous" and corr.grade_after == "correct"
    assert hits[0].chunk["chunk_id"] == "b#s#0"                       # best score first
    assert "acetaminophen anticoagulant" in corr.note()


def test_correct_retrieval_is_untouched(monkeypatch):
    monkeypatch.setattr(corrective, "rewrite_query", lambda q: pytest.fail("must not rewrite"))
    hits, corr = corrective.corrective_search("q", FakeRetriever({"q": [hit(0.9)]}))
    assert len(hits) == 1 and corr.note() == ""


# ------------------------------------------------------------------- tools
@pytest.fixture
def tools(monkeypatch):
    from src.agent import tools as t
    labels = {
        "warfarin": {"drug": "warfarin", "doc_id": "fda:w", "title": "FDA label: warfarin", "url": "u",
                     "brand_names": ["Coumadin"],
                     "sections": {"drug_interactions": "NSAIDs increase bleeding risk. Other sentence."}},
        "ibuprofen": {"drug": "ibuprofen", "doc_id": "fda:i", "title": "FDA label: ibuprofen", "url": "u",
                      "brand_names": ["Advil"],
                      "sections": {"warnings": "Ask a doctor if taking a blood thinning (anticoagulant) drug."}},
        "acetaminophen": {"drug": "acetaminophen", "doc_id": "fda:a", "title": "FDA label: acetaminophen",
                          "url": "u", "brand_names": ["Tylenol"], "sections": {"warnings": "Liver warning. Ask a doctor if taking the blood thinning drug warfarin."}},
    }
    monkeypatch.setattr(t, "_labels", lambda: labels)
    monkeypatch.setattr(t, "_brand_index", lambda: {"coumadin": "warfarin", "advil": "ibuprofen"})
    return t


def test_normalize_drug(tools):
    assert tools.normalize_drug("Paracetamol") == "acetaminophen"
    assert tools.normalize_drug("COUMADIN") == "warfarin"
    assert tools.normalize_drug("warfarin") == "warfarin"
    assert tools.normalize_drug(None) == "" and tools.normalize_drug("unknowndrug") == "unknowndrug"


def test_lookup_falls_back_to_warnings_when_section_missing(tools):
    sources, _ = tools.lookup_drug_label("ibuprofen", "drug_interactions")
    assert sources[0]["section"] == "warnings"


def test_lookup_unknown_drug_lists_available(tools):
    sources, info = tools.lookup_drug_label("nonsense")
    assert sources == [] and "warfarin" in info


def test_check_interaction_finds_class_mentions_in_both_labels(tools):
    sources, info = tools.check_interaction("Ibuprofen", "warfarin")
    assert {s["title"] for s in sources} == {"FDA label: ibuprofen", "FDA label: warfarin"}
    assert info == ""


def test_check_interaction_single_drug_does_not_crash(tools):
    sources, _ = tools.check_interaction("warfarin", None)
    assert sources and sources[0]["section"] == "drug_interactions"


# --------------------------------------------------------------- fine-tuning data
def test_split_is_deterministic_and_by_document():
    from src.finetune.make_pairs import parse_questions, split_of
    assert split_of("fda:abc") == split_of("fda:abc")
    assert {split_of(f"doc{i}") for i in range(200)} == {"train", "test"}
    raw = "1. What are the side effects of metformin?\n- Can I take it with food?\nnot a question\nShort?"
    assert parse_questions(raw) == ["What are the side effects of metformin?", "Can I take it with food?"]


# ------------------------------------------------- citation checking (agent safety net)
def test_strip_invalid_removes_only_made_up_markers():
    from src.citations import strip_invalid
    assert strip_invalid("Fact [1][5]. Other [2].", 2) == "Fact [1]. Other [2]."


def test_uncited_sentences_flags_unsourced_facts_but_not_framing():
    from src.citations import uncited_sentences
    answer = ("Ibuprofen raises the risk of bleeding with warfarin [1]. "
              "It is important to monitor your INR levels if you are taking both. "
              "Consult your healthcare provider for further guidance.")
    assert uncited_sentences(answer) == ["It is important to monitor your INR levels if you are taking both."]
    assert uncited_sentences("Short.") == []


def test_trailing_citation_covers_the_paragraph_before_it():
    from src.citations import uncited_sentences
    covered = "Warfarin passes into breast milk in small amounts and may need monitoring. Infants can bruise easily. [1]"
    assert uncited_sentences(covered) == []
    after = covered + " Dose changes are common during the first weeks of treatment."
    assert uncited_sentences(after) == ["Dose changes are common during the first weeks of treatment."]


def test_routing_hint_only_for_special_populations():
    from src.agent.graph import routing_hint
    assert "search_documents" in routing_hint("Can pregnant women take warfarin?")
    assert "search_documents" in routing_hint("Is metformin safe with kidney disease?")
    assert routing_hint("What is the boxed warning for fluoxetine?") == ""


def test_personal_advice_hint_and_no_false_triggers():
    from src.agent.graph import routing_hint
    for q in ["I have a headache, what should I take?", "Should I take ibuprofen for back pain?"]:
        assert "do NOT recommend" in routing_hint(q)
    for q in ["Can I take ibuprofen with warfarin?", "What are the side effects of metformin?",
              "What is the boxed warning for fluoxetine?"]:
        assert "do NOT recommend" not in routing_hint(q)


def test_based_on_the_information_is_not_exempt_framing():
    from src.citations import uncited_sentences
    assert uncited_sentences("Based on the information available, Tylenol can be taken with warfarin.") != []


def test_check_interaction_expands_drug_classes(tools):
    # "blood thinner" is a class, not a drug: it should expand to the labelled members (warfarin here)
    sources, info = tools.check_interaction("tylenol", "blood thinner")
    assert sources and info == ""
    assert {s["title"] for s in sources} <= {"FDA label: acetaminophen", "FDA label: warfarin"}


def test_check_interaction_placeholder_second_drug_falls_back(tools):
    for placeholder in ("nil", "any other drug", "none"):
        sources, _ = tools.check_interaction("warfarin", placeholder)
        assert sources and sources[0]["section"] == "drug_interactions"


# ------------------------------------------------------------------ reranker
def test_reranker_reorders_but_keeps_dense_scores():
    from src.retrieval.rerank import RerankedRetriever

    class Base:
        def search(self, q, k):
            return [Hit(0.90, chunk("a#s#0", "irrelevant text")), Hit(0.70, chunk("b#s#0", "the real answer"))]

    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            return [0.1 if "irrelevant" in text else 5.0 for _, text in pairs]

    rr = RerankedRetriever(Base())
    rr._model = FakeCrossEncoder()
    hits = rr.search("q", 2)
    assert [h.chunk["chunk_id"] for h in hits] == ["b#s#0", "a#s#0"]       # cross-encoder order wins
    assert [h.score for h in hits] == [0.70, 0.90]                         # dense cosine preserved for thresholds
    assert RerankedRetriever(type("E", (), {"search": lambda s, q, k: []})()).search("q", 3) == []


# ------------------------------------------------------- coverage check (absent drugs)
@pytest.fixture
def coverage(monkeypatch, tools):
    from src.agent import coverage as cov
    monkeypatch.setattr(cov, "_labels", lambda: {"warfarin": {}, "ibuprofen": {}, "metformin": {}})
    monkeypatch.setattr(cov, "_covered_terms", lambda: {"warfarin", "ibuprofen", "metformin",
                                                        "metformin and digestive disorders"})
    return cov


def test_missing_subject_refuses_only_when_no_named_drug_is_covered(coverage, monkeypatch):
    def with_names(names):
        monkeypatch.setattr(coverage, "extract_drugs", lambda q: names)
        return coverage.missing_subject("q")
    assert with_names(["lithium"]) == ["lithium"]                      # only an absent drug: refuse
    assert with_names(["lithium", "ibuprofen"]) == []                  # a covered drug is named: proceed
    assert with_names(["vitamin b12", "metformin"]) == []
    assert with_names([]) == []                                        # nothing extracted: fail open


def test_extract_drugs_drops_names_the_model_invented(coverage, monkeypatch):
    class FakeClient:
        def chat(self, **kw):
            return {"message": {"content": "Aspirin, Ibuprofen"}}
    monkeypatch.setattr(coverage, "_client", FakeClient())
    assert coverage.extract_drugs("What painkiller is safest for my stomach ulcer?") == []
    assert coverage.extract_drugs("Is ibuprofen safe?") == ["ibuprofen"]


def test_is_covered_uses_labels_brands_classes_and_titles(coverage):
    assert coverage.is_covered("warfarin") and coverage.is_covered("blood thinner")
    assert not coverage.is_covered("lithium")


def test_check_interaction_unlabelled_drug_searches_the_other_label(tools):
    # "lithium" has no label: report what the warfarin label says about it, or say nothing is known
    tools._labels()["warfarin"]["sections"]["drug_interactions"] += " Lithium levels may change."
    sources, info = tools.check_interaction("warfarin", "lithium")
    assert sources and "Lithium" in sources[0]["text"] and info == ""
    sources, info = tools.check_interaction("ibuprofen", "lithium")        # ibuprofen label never mentions it
    assert sources == [] and "no label for lithium" in info and "Do not assume" in info


def test_citation_repair_that_loses_the_content_is_rejected(monkeypatch):
    from src.agent import graph

    class Client:
        def __init__(self, reply):
            self.reply = reply

        def chat(self, **kw):
            return {"message": {"content": self.reply}}

    original = "The boxed warning says metformin can cause lactic acidosis, a rare but serious complication."
    monkeypatch.setattr(graph, "_client", Client("[1]"))                     # degenerate repair
    assert graph._add_citations([{"role": "user", "content": "q"}, {"role": "assistant", "content": original}],
                                original, 1) == original
    good = "The boxed warning says metformin can cause lactic acidosis, a rare but serious complication [1]."
    monkeypatch.setattr(graph, "_client", Client(good))                      # a real repair is accepted
    assert graph._add_citations([{"role": "user", "content": "q"}, {"role": "assistant", "content": original}],
                                original, 1) == good


def test_leading_marker_covers_its_paragraph():
    from src.citations import uncited_sentences
    leading = ("[1] The boxed warning says metformin can cause lactic acidosis, a rare complication. "
               "Symptoms include malaise, muscle pain and trouble breathing in affected patients.")
    assert uncited_sentences(leading) == []
    midway = ("Metformin can cause lactic acidosis, a rare complication of treatment [1]. "
              "Symptoms include malaise, muscle pain and trouble breathing in affected patients.")
    assert uncited_sentences(midway) == [
        "Symptoms include malaise, muscle pain and trouble breathing in affected patients."]


def test_normalize_drug_handles_missing_and_and_misspellings(tools, monkeypatch):
    labels = tools._labels()
    labels["sacubitril and valsartan"] = {"sections": {}, "brand_names": []}
    assert tools.normalize_drug("sacubitril valsartan") == "sacubitril and valsartan"
    assert tools.normalize_drug("Sacubitril/Valsartan") == "sacubitril and valsartan"
    assert tools.normalize_drug("ibuprofin") == "ibuprofen"                    # close spelling
    assert tools.normalize_drug("lithium") == "lithium"                        # genuinely unknown stays unknown
    assert tools.normalize_drug("blood thinner") == "blood thinner"            # classes are left alone
