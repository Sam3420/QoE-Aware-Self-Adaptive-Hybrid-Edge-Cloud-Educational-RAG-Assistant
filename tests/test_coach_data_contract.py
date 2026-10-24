"""The coaching data contract's implementations: matching.match_question, matching.video_coverage,
formative_pool.get_formative_items, the reflection template and the tagging evaluation (offline, deterministic)."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

import formative_pool
import knowledge_prep.concepts as concepts_mod
import matching
import tracker
from agent import coach_data
from agent.reflection_template import render
from app.db.repositories import AdaptationRepository
from app.llm.types import LLMGenerateResult
from app.services.errors import MCQGenerationError
from config import settings
from knowledge_prep.concepts import Vocabulary, save_vocabulary
from services.vector_store import Chunk, FaissVectorStore

TOPIC = "photosynthesis"  # the fake services' topic


def concept(cid, name, description, aliases=()):
    return {"id": cid, "name": name, "description": description, "aliases": list(aliases),
            "evidence": "Plants make food from light."}


VOCAB = Vocabulary(topic_id=TOPIC, generated_by="test", generated_at=datetime(2026, 10, 24, tzinfo=timezone.utc),
                   concepts=[
                       concept("villi", "Villi", "Finger-like projections that absorb food.", ["villus"]),
                       concept("stomata", "Stomata", "Leaf pores opened by guard cells.", ["guard cells"]),
                       concept("xylem", "Xylem", "Tubes that carry water up the plant.", ["transpiration pull"]),
                       concept("heart", "Heart", "Pumps blood around the body.", ["double circulation"]),
                       concept("kidney", "Kidney", "Filters blood to make urine.", ["nephron"]),
                       concept("enzymes", "Enzymes", "Digest food in the stomach.", ["pepsin"]),
                   ])
KEYWORDS = ["absorb", "pore", "water", "blood", "urine", "digest"]  # one dimension per concept


class KeywordEmbedder:
    """Deterministic embedder: one dimension per keyword (concept i <-> KEYWORDS[i])."""

    name = "keyword-test"

    def _vec(self, text):
        v = np.array([text.lower().count(k) for k in KEYWORDS], dtype="float32") + 0.01
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def vocab_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path / "concepts")
    save_vocabulary(VOCAB)
    matching.set_embedder(None)
    yield
    matching.set_embedder(None)


# --- match_question --------------------------------------------------------------------------------------------------

def test_question_naming_one_concept_is_a_lexical_match(services, vocab_dir):
    assert matching.match_question("What do the villi do?", TOPIC) == ("villi", settings.match_lexical_confidence)
    assert matching.match_question("How do GUARD CELLS work?", TOPIC)[0] == "stomata"  # alias, any case


def test_embedding_match_and_ambiguity_penalty(services, vocab_dir):
    matching.set_embedder(KeywordEmbedder())
    concept_id, confidence = matching.match_question("Which tube carries water upward?", TOPIC)
    assert concept_id == "xylem" and confidence >= settings.coach_match_min_confidence
    # equally about blood and water: the runner-up is too close, so confidence drops below the record threshold
    concept_id, confidence = matching.match_question("Is it about blood or water?", TOPIC)
    assert concept_id in {"heart", "xylem"} and confidence < settings.coach_match_min_confidence


def test_several_named_concepts_are_resolved_by_embedding(services, vocab_dir):
    matching.set_embedder(KeywordEmbedder())
    assert matching.match_question("Do villi or stomata absorb food?", TOPIC)[0] == "villi"


def test_no_embedder_and_no_name_or_empty_question(services, vocab_dir):
    assert matching.match_question("Tell me something interesting", TOPIC) == (None, 0.0)
    assert matching.match_question("   ", TOPIC) == (None, 0.0)
    concept_id, confidence = matching.match_question("Villi or stomata?", TOPIC)  # two named, no embedder
    assert concept_id == "villi" and confidence == settings.match_lexical_confidence / 2


def test_unknown_topic_raises(services, vocab_dir):
    with pytest.raises(tracker.VocabularyNotFoundError):
        matching.match_question("anything", "no_such_topic")


# --- video_coverage --------------------------------------------------------------------------------------------------

def test_video_not_indexed_is_unknown_not_empty(services, vocab_dir):
    with pytest.raises(matching.CoverageUnknownError):
        matching.video_coverage("notIndexed1", TOPIC)
    assert coach_data.video_coverage.__module__ == "agent.coach_data"  # callers treat the error as unknown (None)


def test_coverage_from_transcript_mentions_without_vectors(services, vocab_dir):
    services.crag.store.build("yt_vidA", [Chunk(chunk_id="c0", text="The villi absorb food. Villi are tiny."),
                                          Chunk(chunk_id="c1", text="The heart pumps blood once here.")])
    assert matching.video_coverage("vidA", TOPIC) == ["villi"]  # villi named twice; heart only once


def test_coverage_from_stored_chunk_vectors(services, vocab_dir, tmp_path):
    store = FaissVectorStore(base_dir=tmp_path / "vec", embedder=KeywordEmbedder())
    store.build("yt_vidB", [Chunk(chunk_id="c0", text="Blood flows through blood vessels and blood returns."),
                            Chunk(chunk_id="c1", text="The kidney makes urine from waste.")])
    services.crag.store = store
    calls = []
    original = store._load
    store._load = lambda rid: calls.append(rid) or original(rid)
    assert matching.video_coverage("vidB", TOPIC) == ["heart", "kidney"]  # vocabulary order
    assert matching.video_coverage("vidB", TOPIC) == ["heart", "kidney"] and len(calls) == 1  # cached


# --- formative pool --------------------------------------------------------------------------------------------------

class FocusLLM:
    """Generates 5 MCQs per call: 2 on the requested concept (label = its name), 2 on another, 1 bank duplicate."""

    def __init__(self, bank_question=None, fail=False):
        self.calls, self.bank_question, self.fail = 0, bank_question, fail

    def generate(self, request):
        self.calls += 1
        if self.fail:
            from app.llm.types import LLMProviderError

            raise LLMProviderError("down")
        focus = request.prompt.split("Focus on these concepts the learner found difficult: ")[1].split(":")[0]
        items = [{"question": f"{focus} question {self.calls}.{i}?", "options": ["a", "b", "c", "d"],
                  "answer_index": i % 4, "concept": focus, "explanation": "Because the lesson says so."}
                 for i in range(2)]
        items += [{"question": f"Other question {self.calls}.{i}?", "options": ["a", "b", "c", "d"],
                   "answer_index": 0, "concept": "Kidney", "explanation": "x"} for i in range(2)]
        items.append({"question": self.bank_question or f"Spare {self.calls}?", "options": ["a", "b", "c", "d"],
                      "answer_index": 0, "concept": focus, "explanation": "x"})
        return LLMGenerateResult(text=json.dumps(items), model_provider="fake", model_name="focus-fake")


def _bank(services):
    services.app.ensure_ready()
    with services.app._db() as db:
        AdaptationRepository(db).save_question_bank(topic_id=TOPIC, model_provider="x", model_name="y", items=[
            {"question": "Bank question?", "options": ["a", "b", "c", "d"], "answer_index": 0, "concept": "Villi"}])


def test_pool_is_generated_outside_the_bank_and_keeps_only_the_concept(services, vocab_dir):
    _bank(services)
    services.app.mcq_llm = llm = FocusLLM(bank_question="Bank question?")
    items = formative_pool.get_formative_items(TOPIC, "villi", 2, exclude_ids=[f"{TOPIC}:0"])
    assert [i.question for i in items] == ["Villi question 1.0?", "Villi question 1.1?"]
    assert all(i.item_id.startswith("fp_") and i.concept_id == "villi" for i in items)
    with services.app._db() as db:
        bank = AdaptationRepository(db).get_question_bank(TOPIC)
        assert [q["question"] for q in bank["items"]] == ["Bank question?"]  # the bank is untouched
        assert formative_pool.pool_size(TOPIC, "villi", db=db) == settings.formative_pool_target  # topped up to 4
    assert llm.calls == 2  # 2 on-concept items per call, the Kidney ones and the bank duplicate dropped


def test_pool_serves_stored_items_and_respects_exclude_ids(services, vocab_dir):
    _bank(services)
    services.app.mcq_llm = llm = FocusLLM()
    first = formative_pool.get_formative_items(TOPIC, "villi", 2, [])
    calls = llm.calls
    second = formative_pool.get_formative_items(TOPIC, "villi", 2, [i.item_id for i in first])
    assert llm.calls == calls  # served from the pool, no generation
    assert not {i.item_id for i in first} & {i.item_id for i in second} and len(second) == 2


def test_failed_top_up_serves_what_exists_or_raises(services, vocab_dir):
    _bank(services)
    services.app.mcq_llm = FocusLLM(fail=True)
    with pytest.raises(MCQGenerationError):
        formative_pool.get_formative_items(TOPIC, "villi", 2, [])
    services.app.mcq_llm = FocusLLM()
    one = formative_pool.get_formative_items(TOPIC, "villi", 1, [])
    services.app.mcq_llm = FocusLLM(fail=True)
    seen = [one[0].item_id]
    rest = formative_pool.get_formative_items(TOPIC, "villi", 10, seen)  # wants 10, generator down
    assert 0 < len(rest) < 10 and one[0].item_id not in {i.item_id for i in rest}


def test_pool_contract_errors(services, vocab_dir):
    with pytest.raises(tracker.UnknownConceptError):
        formative_pool.get_formative_items(TOPIC, "photosynthesis_magic", 2, [])
    with pytest.raises(tracker.VocabularyNotFoundError):
        formative_pool.get_formative_items("no_such_topic", "villi", 2, [])
    assert formative_pool.get_formative_items(TOPIC, "villi", 0, []) == []


# --- reflection template ---------------------------------------------------------------------------------------------

def reflection(goals_met=(True, False), formative=1):
    goals = [{"concept_id": f"g{i}", "name": f"Goal {i}", "met": met} for i, met in enumerate(goals_met)]
    return {"goals": goals, "formative_checks": formative, "suggestions_shown": 3, "suggestions_accepted": 2,
            "weak_next_time": ["g1"], "concepts": [
                {"concept_id": "g0", "name": "Goal 0", "before": 0.33, "after": 0.6, "gain": 0.27, "attempts": 2},
                {"concept_id": "g1", "name": "Goal 1", "before": 0.5, "after": 0.4, "gain": -0.1, "attempts": 1}]}


def test_reflection_template_variants():
    text = render(reflection(), {"g1": "Goal 1"})
    assert text.startswith("You met 1 of your 2 goals (**Goal 0**).")
    assert "**Goal 0** (33% → 60%)" in text and "**Goal 1** slipped a little" in text
    assert "1 practice check and 2 of 3 suggestions taken" in text and "start with **Goal 1**" in text
    assert render(reflection((True, True)), {}).startswith("You met all 2 of your goals for this session")
    assert render(reflection((False, False)), {}).startswith("You did not reach your 2 goals yet")
    empty = {"goals": [], "concepts": [], "formative_checks": 0, "suggestions_shown": 0, "weak_next_time": []}
    assert render(empty, {}).startswith("Session complete.")


# --- tagging evaluation ----------------------------------------------------------------------------------------------

def test_tagging_score_and_agreement():
    import scripts.eval_tagging as ev

    rows = [{"id": "a", "topic_id": TOPIC, "question": "q", "gold_concept_id": "villi", "gold_concept_id_2": "villi"},
            {"id": "b", "topic_id": TOPIC, "question": "q", "gold_concept_id": "heart", "gold_concept_id_2": "xylem"},
            {"id": "c", "topic_id": TOPIC, "question": "q", "gold_concept_id": "none", "gold_concept_id_2": "none"},
            {"id": "d", "topic_id": TOPIC, "question": "q", "gold_concept_id": "", "gold_concept_id_2": ""}]
    predictions = {"a": ("villi", 0.9), "b": ("xylem", 0.7), "c": ("heart", 0.3)}
    s = ev.score(rows, predictions, 0.6)
    assert (s["labelled"], s["accuracy"], s["coverage"], s["precision"]) == (3, round(2 / 3, 4), 1.0, 0.5)
    assert [e["id"] for e in s["errors"]] == ["b"]  # wrong and recorded
    assert ev.score(rows, predictions, 0.8)["accuracy"] == round(2 / 3, 4)  # b no longer recorded: still wrong
    agree = ev.agreement(rows)
    assert agree["rows"] == 3 and agree["agreement"] == round(2 / 3, 4) and 0 < agree["cohens_kappa"] < 1


def test_tagging_labels_are_validated(vocab_dir):
    import scripts.eval_tagging as ev

    rows = [{"id": "a", "topic_id": TOPIC, "question": "q", "gold_concept_id": "villus", "gold_concept_id_2": ""}]
    assert ev.validate(rows) == ["a: gold_concept_id 'villus' is not a concept of photosynthesis"]
