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
    monkeypatch.setattr(cov, "_covered_terms", lambda: {"warfarin", "ibuprofen", "metformin"})
    monkeypatch.setattr(cov, "_titles", lambda: ["metformin and digestive disorders", "lithium and drug interactions",
                                                 "tirzepatide weight loss", "tirzepatide and cardiovascular risk"])
    monkeypatch.setattr(cov, "_mention_docs", lambda name: {"lithium": 165, "tirzepatide": 8}.get(name, 1))
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


def test_named_covered_drug_makes_a_low_cosine_one_word_query_correct():
    """'cetirizine' as a one-word query scores low on cosine but names a drug whose chunk was retrieved."""
    class Retr:
        _query_drugs = staticmethod(lambda q: {"cetirizine"} if "cetirizine" in q else set())

        def __init__(self, hits):
            self.hits = hits

        def search(self, q, k):
            return self.hits

    own = [Hit(0.55, chunk("c#s#0", "cetirizine relieves allergy symptoms", drug="cetirizine"))]
    hits, corr = corrective.corrective_search("cetirizine", Retr(own))
    assert hits and corr.grade_before == "correct"
    other = [Hit(0.55, chunk("d#s#0", "unrelated warfarin text", drug="warfarin"))]
    hits, corr = corrective.corrective_search("cetirizine", Retr(other))     # named drug absent from the hits
    assert hits == [] and corr.grade_before == "incorrect"
    off_topic = corrective.corrective_search("how tall is everest", Retr(other))
    assert off_topic[0] == []                                                # no drug named: still refused


def test_reranker_passage_carries_its_drug_name_and_section():
    from src.retrieval.rerank import RerankedRetriever, passage_for_reranking
    otc = chunk("c#indications_and_usage#0", "Uses temporarily relieves hay fever symptoms", drug="cetirizine",
                section="indications_and_usage", title="FDA label: cetirizine")
    text = passage_for_reranking(otc)
    assert "cetirizine" in text and "indications and usage" in text and "hay fever" in text

    seen = []

    class Spy:
        def predict(self, pairs, show_progress_bar=False):
            seen.extend(t for _, t in pairs)
            return [1.0] * len(pairs)

    class Base:
        def search(self, q, k):
            return [Hit(0.0, otc)]

    rr = RerankedRetriever(Base())
    rr._model = Spy()
    rr.search("What is cetirizine for?", 1)
    assert "cetirizine" in seen[0]          # the cross-encoder is given the drug name, not just the body text


def test_rewrite_that_adds_a_drug_is_discarded(monkeypatch):
    """'symptoms of the flu' -> 'acetaminophen flu symptoms' must not retrieve acetaminophen's label."""
    class Retr:
        def __init__(self):
            self.queries = []

        _query_drugs = staticmethod(lambda q: {"acetaminophen"} if "acetaminophen" in q.lower() else set())

        def search(self, q, k):
            self.queries.append(q)
            return [Hit(0.68, chunk("a#s#0", "relieves flu aches", drug="acetaminophen"))]

    monkeypatch.setattr(corrective, "rewrite_query", lambda q: "acetaminophen flu symptoms")
    r = Retr()
    hits, corr = corrective.corrective_search("symptoms of the flu", r, implied=lambda text: set())
    assert hits == [] and corr.rejected_drugs == ["acetaminophen"] and "never mentioned" in corr.note()
    assert r.queries == ["symptoms of the flu"]                      # the rewritten query was never searched


def test_rewrite_may_swap_a_lay_name_for_the_generic_drug(monkeypatch):
    class Retr:
        _query_drugs = staticmethod(lambda q: {"acetaminophen"} if "acetaminophen" in q.lower() else set())

        def search(self, q, k):
            score = 0.80 if "acetaminophen" in q else 0.68
            return [Hit(score, chunk("a#s#0", "x", drug="acetaminophen"))]

    monkeypatch.setattr(corrective, "rewrite_query", lambda q: "acetaminophen safety with anticoagulant")
    hits, corr = corrective.corrective_search("is tylenol ok with a blood thinner", Retr(),
                                              implied=lambda text: {"acetaminophen", "warfarin"})
    assert hits and corr.rejected_drugs is None                      # 'tylenol' implies acetaminophen: allowed


def test_implied_drugs_covers_brands_synonyms_and_classes(tools):
    assert tools.implied_drugs("Is Tylenol ok with a blood thinner?") >= {"acetaminophen", "warfarin"}
    assert tools.implied_drugs("symptoms of the flu") == set()
    assert tools.implied_drugs("warfarin and ibuprofen") == {"warfarin", "ibuprofen"}


# ----------------------------------- gaps found by the adversarial set (partial coverage, empty answers, vague asks)
def test_coverage_reports_uncovered_names_when_another_drug_is_covered(coverage, monkeypatch):
    def with_names(names):
        monkeypatch.setattr(coverage, "extract_drugs", lambda q: names)
        return coverage.coverage("q")
    assert with_names(["lithium"]) == (["lithium"], [])                  # nothing covered: refuse
    assert with_names(["lithium", "ibuprofen"]) == ([], ["lithium"])     # answer, but warn about lithium
    assert with_names(["warfarin", "ibuprofen"]) == ([], [])             # all covered: nothing to say
    assert with_names([]) == ([], [])


def test_underspecified_only_for_short_generic_questions(coverage):
    assert coverage.is_underspecified("drug interactions", drugs_named=False)
    assert coverage.is_underspecified("What are side effects?", drugs_named=False)
    assert not coverage.is_underspecified("drug interactions", drugs_named=True)
    assert not coverage.is_underspecified("serotonin syndrome", drugs_named=False)   # a real topic
    assert not coverage.is_underspecified("What are the side effects of metformin?", drugs_named=False)


def test_degenerate_answers_are_detected():
    from src.agent import graph
    assert graph.is_degenerate("The usual starting dose of lisinopril for hypertension is [1].")
    assert graph.is_degenerate("Yes [1].")
    assert not graph.is_degenerate("Warfarin carries a boxed warning for bleeding risk [1].")
    assert not graph.is_degenerate("The usual starting dose is 10 mg once daily [1].")


# ------------------------------------------------ brand names, and answers with a hole in them (adversarial set)
@pytest.fixture(autouse=True)
def _no_real_brands(monkeypatch):
    """Keep tests independent of whatever labels are downloaded on this machine."""
    from src.retrieval import hybrid
    monkeypatch.setattr(hybrid, "_load_brands", lambda drugs: {})


def test_brand_name_in_query_boosts_the_generic_drugs_chunks():
    """'Zoloft side effects' names no generic drug; without the brand map nothing is boosted and a look-alike
    name (zoledronic acid) can win."""
    chunks = [chunk("a#s#0", "side effects dizziness", drug="zoledronic acid"),
              chunk("b#s#0", "side effects dizziness", drug="sertraline")]
    dense = FakeDense(chunks, [0.80, 0.79])
    assert HybridRetriever(dense, drug_boost=0.01).search("Zoloft side effects", 2)[0].chunk["drug"] == "zoledronic acid"
    with_brands = HybridRetriever(dense, drug_boost=0.01, brands={"zoloft": "sertraline"})
    assert with_brands._query_drugs("Zoloft side effects") == {"sertraline"}
    assert with_brands.search("Zoloft side effects", 2)[0].chunk["drug"] == "sertraline"


def test_brands_come_only_from_single_ingredient_labels():
    from src.data import fetch_openfda as f
    labels = [
        {"openfda": {"generic_name": ["SERTRALINE HYDROCHLORIDE"], "brand_name": ["Zoloft", "Sertraline"]}},
        {"openfda": {"generic_name": ["SITAGLIPTIN AND METFORMIN HYDROCHLORIDE"], "brand_name": ["Janumet"]}},
        {"openfda": {"generic_name": ["METFORMIN HYDROCHLORIDE"], "brand_name": ["Glucophage"]}},
    ]
    assert f.brands_of("sertraline", labels) == ["Sertraline", "Zoloft"]
    assert f.brands_of("metformin", labels) == ["Glucophage"]          # Janumet is a combination: excluded
    assert f._is_single_ingredient("sacubitril and valsartan", ["SACUBITRIL AND VALSARTAN"])
    assert not f._is_single_ingredient("valsartan", ["VALSARTAN AND HYDROCHLOROTHIAZIDE"])


def test_degenerate_detects_a_hole_in_the_middle_of_an_answer():
    from src.agent import graph
    assert graph.is_degenerate("The usual starting dose of lisinopril for hypertension is [1]. However, the dose "
                               "can be adjusted based on clinical response and tolerability.")
    assert not graph.is_degenerate("The usual starting dose is 10 mg once daily [1]. The dose can be adjusted "
                                   "based on clinical response [1].")


def test_brand_expansion_brings_the_generics_chunks_into_the_candidate_pool():
    """The real failure: dense and BM25 never saw sertraline's chunks for 'Zoloft side effects', so the drug
    boost had nothing to boost. The query must be expanded with the generic before searching."""
    seen = []

    class Recording(FakeDense):
        def search(self, query, k):
            seen.append(query)
            return super().search(query, k)

    chunks = [chunk("a#s#0", "side effects dizziness", drug="zoledronic acid"),
              chunk("b#s#0", "side effects nausea", drug="sertraline")]
    r = HybridRetriever(Recording(chunks, [0.80, 0.79]), brands={"zoloft": "sertraline"})
    assert r.expand("Zoloft side effects") == "Zoloft side effects sertraline"
    assert r.expand("sertraline side effects") == "sertraline side effects"      # generic already present
    assert r.expand("side effects of ibuprofen") == "side effects of ibuprofen"  # no brand: untouched
    r.search("Zoloft side effects", 2)
    assert seen == ["Zoloft side effects sertraline"]
    # BM25 sees the expanded query too: it finds the sertraline chunk through the generic's name
    assert BM25Retriever(chunks).search("Zoloft side effects sertraline", 1)[0].chunk["drug"] == "sertraline"


def test_misspelled_drug_names_are_recognised_but_ordinary_words_are_not():
    chunks = [chunk(f"{d}#s#0", "text", drug=d) for d in ("metformin", "ibuprofen", "warfarin", "diazepam", "insulin")]
    r = HybridRetriever(FakeDense(chunks, [0.5] * 5), brands={})
    assert r._query_drugs("side effects of metformine") == {"metformin"}
    assert r._query_drugs("Can I take ibuprofin with warfrin?") == {"ibuprofen", "warfarin"}
    assert r.expand("side effects of metformine") == "side effects of metformine metformin"
    for ordinary in ("pregnancy dosage interactions", "hypertension medication symptoms", "breastfeeding children elderly"):
        assert r._query_drugs(ordinary) == set()


def test_normalize_drug_rejoins_a_brand_name_the_model_split(tools):
    assert tools.normalize_drug("coum adin") == "warfarin"       # 'coumadin' is a brand in the fixture
    assert tools.normalize_drug("ibu-profen") == "ibuprofen"
    assert tools.normalize_drug("no such drug") == "no such drug"


def test_inverted_index_bm25_gives_the_same_ranking_as_scoring_every_chunk():
    """The inverted index is an optimisation; it must not change a single score or tie-break."""
    import random
    random.seed(7)
    vocab = ["warfarin", "bleeding", "kidney", "dose", "liver", "pregnancy", "nausea", "interaction", "renal", "label"]
    chunks = [chunk(f"c{i}#s#0", " ".join(random.choices(vocab, k=random.randint(3, 25))), drug=f"d{i % 7}")
              for i in range(120)]
    bm = BM25Retriever(chunks)

    def brute(query, k):
        terms = [t for t in tokenize(query) if t in bm.idf]
        scored = []
        for i, tf in enumerate(bm.tfs):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += bm.idf[t] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * bm.lens[i] / bm.avg_len))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(round(s, 12), bm.chunks[i]["chunk_id"]) for s, i in scored[:k]]

    for q in ("warfarin bleeding", "kidney dose renal kidney", "pregnancy liver nausea interaction label", "zzz", "dose"):
        assert [(round(h.score, 12), h.chunk["chunk_id"]) for h in bm.search(q, 15)] == brute(q, 15)


def test_a_single_abstract_title_does_not_make_a_labelless_drug_covered(coverage):
    """Real failure at 744 drugs: one PubMed title 'Lithium and drug interactions' made lithium 'covered', and the
    agent answered 'side effects of lithium' from lithium mentions inside NSAID labels."""
    assert not coverage.is_covered("lithium")          # named in one title, in 165 documents
    assert coverage.is_covered("tirzepatide")          # named in two titles: the index really has material on it
    assert coverage.is_covered("warfarin")             # label


# ---- incremental index build, approximate index, label quality gate ----

def test_incremental_build_reuses_unchanged_vectors_and_embeds_only_new(tmp_path, monkeypatch):
    import json as _json

    import faiss
    import numpy as np

    from src.ingest import build_index as bi

    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        out = np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    idx_dir, proc = tmp_path / "index", tmp_path / "processed"
    idx_dir.mkdir(), proc.mkdir()
    monkeypatch.setattr(bi, "INDEX_DIR", idx_dir)
    monkeypatch.setattr(bi, "PROCESSED_DIR", proc)
    monkeypatch.setattr(bi, "embed_documents", fake_embed)
    monkeypatch.setattr("sys.argv", ["build_index"])

    def write(texts):
        with open(proc / "chunks.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                f.write(_json.dumps({"chunk_id": str(i), "text": t}) + "\n")

    write(["alpha", "beta"])
    bi.main()
    assert [t for batch in calls for t in batch] == ["alpha", "beta"]
    calls.clear()
    write(["alpha", "beta", "gamma gamma"])          # one new chunk
    bi.main()
    assert [t for batch in calls for t in batch] == ["gamma gamma"]
    index = faiss.read_index(str(idx_dir / "faiss.index"))
    assert index.ntotal == 3
    # reused vectors must be the same numbers a fresh embedding would give
    expect = fake_embed(["alpha", "beta", "gamma gamma"])
    assert np.allclose(index.reconstruct_n(0, 3), expect, atol=1e-6)


def test_hnsw_index_returns_same_top_hit_as_exact_search():
    import faiss
    import numpy as np

    from src.ingest.build_index import build_hnsw

    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 16)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    exact = faiss.IndexFlatIP(16)
    exact.add(x)
    approx = build_hnsw(x)
    approx.hnsw.efSearch = 64
    q = x[:50] + 0.01 * rng.normal(size=(50, 16)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    assert (exact.search(q, 1)[1] == approx.search(q, 1)[1]).mean() >= 0.98


def test_label_quality_gate_drops_thin_homeopathic_and_unindicated_labels():
    from src.data.fetch_openfda import usable

    good = {"drug": "phentermine", "sections": {"indications_and_usage": "x" * 2000, "warnings": "y" * 2000}}
    assert usable(good)
    assert not usable({"drug": "phentermine", "sections": {"indications_and_usage": "short"}})
    assert not usable({"drug": "phentermine", "sections": {"warnings": "y" * 5000}})
    assert not usable({**good, "drug": "gonorrheal urethral secretion human"})


def test_label_less_drug_needs_a_real_share_of_its_mentions_to_be_titled(coverage, monkeypatch):
    # tirzepatide is titled in 2 documents of the fixture; whether that counts depends on the share
    monkeypatch.setattr(coverage, "_mention_docs", lambda name: 10)
    assert coverage.is_covered("tirzepatide")                  # 2 titled / 10 mentions = 0.2
    monkeypatch.setattr(coverage, "_mention_docs", lambda name: 100)
    assert not coverage.is_covered("tirzepatide")              # 2 / 100: the corpus only names it in passing
    monkeypatch.setattr(coverage, "_mention_docs", lambda name: 165)
    assert not coverage.is_covered("lithium")                  # the real case: 2 titled, 165 mentions


# ---- claim-level faithfulness (no model needed for these: sentence/citation pairing and the number check) ----

def test_claims_pair_each_sentence_with_the_citation_that_covers_it():
    from src.faithfulness import claims

    answer = ("Metformin can cause lactic acidosis in patients with renal impairment. "
              "The risk is higher in people who drink alcohol heavily [1].\n"
              "[2] The usual starting dose is 500 mg twice a day with meals.\n"
              "Please consult a healthcare professional before changing your dose of any medicine.\n"
              "This long sentence has no citation at all and so it is left to the uncited check.")
    got = claims(answer)
    assert got[0] == ("Metformin can cause lactic acidosis in patients with renal impairment.", [1])   # covered by the next sentence's [1]
    assert got[1][1] == [1] and got[2][1] == [2]            # a leading [2] covers its paragraph
    assert len(got) == 3                                     # framing and uncited sentences are not claims


def test_number_check_flags_a_changed_dose_but_not_formatting_or_number_words():
    from src.faithfulness import unsupported_numbers

    source = ["The recommended dose is 500 mg twice daily. Do not exceed 4,000 mg in 24 hours."]
    assert unsupported_numbers("Take 500 mg 2 times a day, up to 4000 mg per day.", source) == []
    assert unsupported_numbers("Take 650 mg twice daily.", source) == ["650"]
    assert unsupported_numbers("The maximum is 4,000 mg per 24 hours.", source) == []


def test_faithfulness_check_reports_unsupported_sentences(monkeypatch):
    from src import faithfulness as fa

    monkeypatch.setattr(fa, "nli", lambda claim, premises: (0.9, 0.0) if "lactic" in claim else (0.1, 0.0))
    sources = [{"text": "Metformin can cause lactic acidosis. The dose is 500 mg."}]
    answer = ("Metformin can cause lactic acidosis in some patients [1]. "
              "Metformin is also known to cause severe hair loss in most patients [1]. "
              "The usual dose of metformin is 850 mg and lactic acidosis is a risk [1].")
    flagged = fa.check(answer, sources)
    assert [f["sentence"][:30] for f in flagged] == ["Metformin is also known to cau", "The usual dose of metformin is"]
    assert flagged[0]["numbers"] == [] and flagged[1]["numbers"] == ["850"]     # one fails entailment, one a number


def test_contradicted_sentence_is_flagged_even_when_it_copies_the_source_words(monkeypatch):
    from src import faithfulness as fa

    monkeypatch.setattr(fa, "nli", lambda claim, premises: (0.05, 0.95))
    sources = [{"text": "Naproxen should not be used in the third trimester of pregnancy."}]
    flagged = fa.check("Naproxen should be used in the third trimester of pregnancy [1].", sources)
    assert len(flagged) == 1 and flagged[0]["overlap"] == 1.0 and flagged[0]["contradiction"] == 0.95


def test_repair_citations_repoints_a_marker_to_the_source_that_contains_the_sentence():
    from src.faithfulness import repair_citations

    sources = [{"text": "Apixaban is indicated for the treatment of deep vein thrombosis and pulmonary embolism."},
               {"text": "Common adverse reactions of apixaban are bleeding and bruising in most patients."}]
    answer = ("Apixaban is indicated for the treatment of deep vein thrombosis and pulmonary embolism [2]. "
              "Common adverse reactions of apixaban are bleeding and bruising [2].")
    fixed, fixes = repair_citations(answer, sources)
    assert fixed.startswith("Apixaban is indicated for the treatment of deep vein thrombosis and pulmonary embolism [1].")
    assert fixed.endswith("bleeding and bruising [2].")                # a correct citation is left alone
    assert fixes == [{"sentence": "Apixaban is indicated for the treatment of deep vein thrombosis and pulmonary embolism.",
                      "was": [2], "now": [1]}]
    # a paraphrase that no source contains is not re-pointed (it is left for the unsupported check)
    assert repair_citations("Apixaban is a wonderful medicine that everybody should take daily [2].", sources)[1] == []
    # a changed number blocks the repair: the other source does not support "20 mg"
    sources[0]["text"] += " The dose is 5 mg."
    odd = "Apixaban is indicated for the treatment of deep vein thrombosis and pulmonary embolism at 20 mg [2]."
    assert repair_citations(odd, sources)[1] == []
