import json

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from qbank import agent
from qbank.agent import (ai_review, flag_question, get_questions, review_batch, run_checks,
                   search_questions, stream_turn)
from qbank.bank import get_bank


def call(tool, **args):
    return tool.invoke({"type": "tool_call", "id": "t1", "name": tool.name, "args": args})


def test_tools_handle_bad_input():
    assert "not found" in get_questions.invoke({"ids": ["PH-99999"]}).lower()
    assert "no topic" in search_questions.invoke({"topic": "cardio-xyz"}).lower()
    assert "unknown check" in run_checks.invoke({"check": "bogus"}).lower()
    assert "not found" in flag_question.invoke({"question_id": "PH-99999", "issue": "x", "reason": "y"}).lower()


def test_list_topics_shows_every_exact_name():
    out = agent.list_topics.invoke({})
    assert "NEET 2019 (16)" in out and "Anaesthesia (97)" in out and "Anesthesia (142)" in out
    assert agent.list_topics.invoke({"match": "hemato"}).count("(") == 2
    assert "no topic" in agent.list_topics.invoke({"match": "cardio-xyz"}).lower()
    # a lookup shows every name containing the text, so a shortened name can't hide the one the user meant
    assert "Miscellaneous (Pharmacology) (105)" in agent.list_topics.invoke({"match": "miscellaneous"})
    assert "Miscellaneous (Pharmacology)" in agent.list_topics.invoke({"match": "miscellaneous (pharmaclogy)"})


def test_list_topics_lets_the_users_own_words_pick_among_look_alikes():
    out = agent.list_topics.invoke({"match": "diuretic"})
    assert "Diuretic (46)" in out and "Diuretics and Antidiuretics (13)" in out  # nothing hidden
    assert "don't mention the others unless the user asks" in out  # but look-alikes aren't volunteered
    # the agent may search a shortened word ("immunomodulator" for "immunomodulators"): the search text must not be
    # declared the answer, or it answers about Immunomodulator (58) when the user asked about Immunomodulators (60)
    out = agent.list_topics.invoke({"match": "immunomodulator"})
    assert "Immunomodulators (60)" in out and "what the user wrote" in out and "Exact match" not in out


def test_topic_tools_say_which_topics_they_covered():
    assert "Hematology and CVS" in call(run_checks, topic="hemato").content
    assert "Miscellaneous (Pharmacology) (105), Miscellaneous (23)" in search_questions.invoke({"topic": "misc"})
    assert "Miscellaneous (Pharmacology)" not in search_questions.invoke({"topic": "miscellaneous"})  # exact name wins


def test_search_reports_how_many_questions_contain_the_word():
    out = search_questions.invoke({"query": "amilodipine"})
    assert out.startswith("7 questions contain 'amilodipine'")
    assert search_questions.invoke({"query": "pirenzipine", "limit": 2}).startswith("5 questions contain 'pirenzipine'; showing the top 2")


def test_run_checks_artifact_and_cap():
    msg = call(run_checks)
    assert len(msg.content) <= 4000 and msg.artifact
    assert {f["check"] for f in msg.artifact} == {"key_conflict", "duplicate_options", "blank_option"}
    assert all(f["source"] == "exact" for f in msg.artifact)
    one = call(run_checks, check="duplicate_options")
    assert one.artifact and all(f["check"] == "duplicate_options" for f in one.artifact)


def test_get_questions_shows_key_and_caps():
    ids = [x["id"] for x in get_bank().items[:10]]
    out = get_questions.invoke({"ids": ids})
    assert len(out) <= 4000 and "Key:" in out and ids[0] in out


def test_flag_question_makes_agent_finding():
    qid = get_bank().items[0]["id"]
    msg = call(flag_question, question_id=qid, issue="Two defensible answers", reason="B and C both fit", severity="urgent")
    f = msg.artifact[0]
    assert f["source"] == "agent" and f["question_id"] == qid and f["severity"] == "medium" and f["key"] == f"agent_flag:{qid}"


def test_make_model_accepts_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    assert agent.make_model(reasoning=False).reasoning is False
    assert agent.make_model().reasoning is True


class Reply:
    def __init__(self, text): self.text = text
    def invoke(self, *_a, **_k): return AIMessage(self.text)


def test_review_batch_malformed_returns_none():
    assert review_batch(Reply("sorry, cannot"), get_bank().items[:3]) is None


def test_review_batch_retries_once_after_a_failed_call(monkeypatch):
    monkeypatch.setattr(agent, "RETRY_DELAY", 0.01)
    qs = get_bank().items[:1]

    class Flaky:
        calls = 0
        def invoke(self, *_a, **_k):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("429 too many requests")
            return AIMessage(json.dumps([{"id": qs[0]["id"], "verdict": "ok", "issue": None, "suggested": None, "reason": ""}]))

    assert review_batch(Flaky(), qs) == [{"id": qs[0]["id"], "verdict": "ok", "issue": None, "suggested": None, "reason": ""}]
    assert Flaky.calls == 2


def test_review_batch_retries_once_after_an_unreadable_reply():
    qs = get_bank().items[:1]
    replies = iter(["thinking out loud, no json", json.dumps([{"id": qs[0]["id"], "verdict": "ok", "issue": None, "suggested": None, "reason": ""}])])

    class Model:
        def invoke(self, *_a, **_k): return AIMessage(next(replies))

    assert review_batch(Model(), qs) == [{"id": qs[0]["id"], "verdict": "ok", "issue": None, "suggested": None, "reason": ""}]


def test_ai_review_turns_problems_into_findings(monkeypatch):
    qs = get_bank().items[:4]
    alt = (qs[0]["answer"] + 1) % 4
    seen = []

    class Reviewer:
        def invoke(self, prompt, **_k):
            seen.append(prompt)
            return AIMessage("```json\n" + json.dumps([
                {"id": qs[0]["id"], "verdict": "problem", "issue": "explanation_conflict", "suggested": "abcd"[alt], "reason": "explanation names it"},
                {"id": qs[1]["id"], "verdict": "ok", "issue": None, "suggested": None, "reason": ""},
                {"id": qs[2]["id"], "verdict": "problem", "issue": "made_up_issue", "suggested": None, "reason": "?"},
                {"id": qs[3]["id"], "verdict": "problem", "issue": "flawed_wording", "suggested": None, "reason": "all of the above"},
            ]) + "\n```")

    monkeypatch.setattr(agent, "make_model", lambda **_: Reviewer())
    msg = call(ai_review, ids=[q["id"] for q in qs])
    assert [f["question_id"] for f in msg.artifact] == [qs[0]["id"]]  # unknown issue dropped; wording not asked for
    f = msg.artifact[0]
    assert f["source"] == "ai" and f["check"] == "ai_explanation_conflict" and f["severity"] == "high"
    assert "AI opinion" in f["detail"] and f["suggested"] == alt and f["key"] == f"ai:{qs[0]['id']}"
    assert "explanation:" in seen[0] and "flawed_wording" not in seen[0]
    item = next(x for x in msg.content.splitlines() if x.startswith(qs[0]["id"]))
    assert "Explanation backs a different option" in item  # each listed item names its issue, so the agent can't mix them up
    msg = call(ai_review, ids=[q["id"] for q in qs], include_wording=True)
    assert "flawed_wording" in seen[-1] and {f["check"] for f in msg.artifact} == {"ai_explanation_conflict", "ai_flawed_wording"}


def test_run_checks_tells_the_agent_which_copies_disagree():
    msg = call(run_checks, check="key_conflict")
    f = msg.artifact[0]
    assert f"{f['question_id']} (copies: {', '.join(f['related_ids'])})" in msg.content


class EchoReviewer:
    """Fake AI reviewer: says every question it is shown has a wrong key and points to option a."""
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt, **_k):
        self.prompts.append(prompt)
        ids = [line[4:].strip() for line in prompt.splitlines() if line.startswith("id: ")]
        return AIMessage(json.dumps([{"id": i, "verdict": "problem", "issue": "wrong_key", "suggested": "a", "reason": "r"} for i in ids]))


def test_check_questions_checks_pasted_questions_against_the_bank(monkeypatch):
    rev = EchoReviewer()
    monkeypatch.setattr(agent, "make_model", lambda **_: rev)
    src = get_bank().by_id["PH-07208"]  # in the bank twice: PH-07208 keyed (A) oral vancomycin, PH-11986 keyed (B) metronidazole
    copy = {"question": src["question"], "options": src["options"], "key": "(B)"}
    messy = {"question": "Which of these is a beta blocker?", "options": ["Atenolol", "atenolol.", "Amlodipine", ""], "key": "a"}
    msg = call(agent.check_questions, questions=[copy, messy])
    by = {}
    for f in msg.artifact:
        by.setdefault(f["question"]["question"], {})[f["check"]] = f
        assert f["question_id"].startswith("NEW-") and f["question"]["topic"] == "Pasted question"
    conflict = by[src["question"]]["key_conflict"]  # exact: the bank has this question keyed differently
    assert conflict["source"] == "exact" and conflict["related_ids"] == ["PH-07208"] and conflict["question"]["answer"] == 1
    assert {"duplicate_options", "blank_option"} <= set(by["Which of these is a beta blocker?"])
    assert by[src["question"]]["ai_wrong_key"]["suggested"] == 0  # the AI reviewer judged both pasted questions
    assert by[src["question"]]["ai_wrong_key"]["reason"] == "r"  # the card shows the reviewer's reason on its own
    assert "PH-" not in conflict["detail"] and "NEW-" not in conflict["detail"]  # the card names no ids
    assert "2 copies" in msg.content and "Oral vancomycin" in msg.content and "Metronidazole" in msg.content
    assert "PH-" not in msg.content and "NEW-" not in msg.content  # ids the agent never sees can't leak into replies
    assert "flawed_wording" in rev.prompts[0]  # new questions get the wording check too
    again = call(agent.check_questions, questions=[copy])
    assert again.artifact[0]["question_id"] == conflict["question_id"]  # same question, same id: no duplicates in the queue


def test_check_questions_rejects_a_key_that_is_not_an_option(monkeypatch):
    monkeypatch.setattr(agent, "make_model", lambda **_: EchoReviewer())
    out = call(agent.check_questions, questions=[{"question": "Q?", "options": ["x", "y"], "key": "D"}])
    assert out.artifact == [] and "key" in out.content.lower()
    five = call(agent.check_questions, questions=[{"question": "Q?", "options": list("vwxyz"), "key": "A"}])
    assert five.artifact == [] and "4" in five.content


def test_ai_review_progress_is_tagged_with_its_own_call_id(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "_progress", seen.append)
    monkeypatch.setattr(agent, "make_model", lambda **_: Reply("[]"))
    call(ai_review, ids=[q["id"] for q in get_bank().items[:3]])
    assert seen and all(e["id"] == "t1" and e["tool"] == "ai_review" for e in seen)
    assert "tool_call_id" not in ai_review.tool_call_schema.model_json_schema()["properties"]


def test_ai_review_counts_skipped_batches(monkeypatch):
    monkeypatch.setattr(agent, "make_model", lambda **_: Reply("not json"))
    msg = call(ai_review, ids=[q["id"] for q in get_bank().items[:3]])
    assert msg.artifact == [] and "1 batch" in msg.content and "skipped" in msg.content
    assert "reviewed 0 questions" in msg.content


def test_ai_review_respects_the_shared_concurrency_cap(monkeypatch):
    import threading
    import time
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class Slow:
        def invoke(self, *_a, **_k):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return AIMessage("[]")

    monkeypatch.setattr(agent, "REVIEW_SLOTS", threading.BoundedSemaphore(2))
    monkeypatch.setattr(agent, "make_model", lambda **_: Slow())
    call(ai_review, ids=[q["id"] for q in get_bank().items[:120]])  # 10 batches, 8 workers
    assert state["peak"] == 2


def test_agent_retries_a_rate_limited_model_call(monkeypatch):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    calls = []

    class RateLimitedOnce(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("too many concurrent requests (status code: 429)")
            return ChatResult(generations=[ChatGeneration(message=AIMessage("recovered"))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "rate-limited-once"

    monkeypatch.setattr(agent, "MODEL_RETRY_DELAY", 0.01)
    ev = list(stream_turn(agent.build_agent(RateLimitedOnce()), [HumanMessage("hi")]))
    assert ev[-1]["type"] == "done" and ev[-1]["messages"][-1].content == "recovered" and len(calls) == 2


def test_agent_runs_at_most_one_ai_review_per_turn(monkeypatch):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    two = AIMessage("", tool_calls=[{"name": "ai_review", "args": {"topic": "Chemotherapy"}, "id": "r1"},
                                    {"name": "ai_review", "args": {"topic": "Endocrinology"}, "id": "r2"}])
    replies = iter([two, AIMessage("done")])

    class Greedy(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=next(replies))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "greedy"

    reviewed = []
    monkeypatch.setattr(agent, "review_batch", lambda m, qs, w=False, cancel=None: reviewed.extend(qs) or [])
    monkeypatch.setattr(agent, "make_model", lambda **_: object())
    ev = list(stream_turn(agent.build_agent(Greedy()), [HumanMessage("review two topics")]))
    ends = {e["id"]: e["content"] for e in ev if e["type"] == "tool_end"}
    assert set(ends) == {"r1", "r2"} and "AI reviewed" in ends["r1"] and "limit" in ends["r2"]
    assert {q["topic"] for q in reviewed} == {"Chemotherapy"} and ev[-1]["type"] == "done"


class Fake:
    def __init__(self, events=None, exc=None): self.events, self.exc = events or [], exc
    def stream(self, *_a, **_k):
        if self.exc:
            raise self.exc
        yield from self.events


def test_stream_turn_maps_agent_stream_to_ui_events():
    call_msg = AIMessage("", tool_calls=[{"name": "bank_overview", "args": {}, "id": "c1"}])
    finding = {"key": "k", "question_id": "PH-00001"}
    tool_msg = ToolMessage("13760 questions\nmore", tool_call_id="c1", name="bank_overview", artifact=[finding])
    final = AIMessage("Done")
    ev = list(stream_turn(Fake([
        ("messages", (AIMessageChunk("Look"), {"langgraph_node": "model"})),
        ("messages", (AIMessageChunk("[reviewer json]"), {"langgraph_node": "tools"})),  # inner LLM: hidden
        ("updates", {"model": {"messages": [call_msg]}}),
        ("custom", {"tool": "ai_review", "done": 20, "total": 40}),
        ("updates", {"tools": {"messages": [tool_msg]}}),
        ("messages", (AIMessageChunk("Done"), {"langgraph_node": "model"})),
        ("updates", {"model": {"messages": [final]}}),
    ]), [HumanMessage("hi")]))
    assert [e["type"] for e in ev] == ["token", "tool_start", "progress", "tool_end", "token", "done"]
    assert ev[1]["name"] == "bank_overview" and ev[3]["summary"] == "13760 questions" and ev[3]["findings"] == [finding]
    assert ev[3]["content"] == "13760 questions\nmore"
    assert ev[-1]["messages"] == [call_msg, tool_msg, final]


def test_stream_turn_error_keeps_history():
    ev = list(stream_turn(Fake(exc=RuntimeError("503 upstream")), [HumanMessage("hi")]))
    assert ev[-1]["type"] == "error" and "503" in ev[-1]["message"] and not any(e["type"] == "done" for e in ev)
    ev = list(stream_turn(Fake(exc=GraphRecursionError("limit")), [HumanMessage("hi")]))
    assert "too many steps" in ev[-1]["message"]
