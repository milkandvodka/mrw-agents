import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from qbank import agent
from qbank import server


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_overview():
    d = TestClient(server.app).get("/api/overview").json()
    assert d["questions"] == 13760 and len(d["planted"]) == 10 and "has_key" in d
    assert len(d["suggestions"]) >= 8 and all(s["label"] and s["prompt"] for s in d["suggestions"])
    assert "example" not in d  # the first screen is one headline and one button, no sample card


def test_overview_offers_several_different_examples_to_paste():
    ex = TestClient(server.app).get("/api/overview").json()["examples"]  # the "Use an example" button cycles through these
    assert len(ex) >= 4 and len(set(ex)) == len(ex) and all("Answer:" in e for e in ex)
    assert "PH-" not in "".join(ex)  # pasted as a faculty member would, without bank ids


def test_no_suggestion_starts_a_minutes_long_ai_review_of_a_topic():
    prompts = [s["prompt"].lower() for s in TestClient(server.app).get("/api/overview").json()["suggestions"]]
    assert not any("ai review" in p or "nervous system" in p for p in prompts)


def test_overview_suggests_several_different_questions_to_paste():
    pastes = [s["prompt"] for s in TestClient(server.app).get("/api/overview").json()["suggestions"] if "Answer:" in s["prompt"]]
    assert len(pastes) >= 4 and len(set(pastes)) == len(pastes)
    assert "PH-" not in "".join(pastes)  # pasted as a faculty member would, without bank ids


def test_conflict_card_marks_the_other_copys_key():
    # the card marks the answer the other copy is keyed to, located by its text (copies may order options differently)
    f = next(f for f in server.get_bank().findings if f["check"] == "key_conflict" and f["question_id"] == "PH-07208")
    ex = server._enrich(f)
    other = server.get_bank().by_id[ex["related_ids"][0]]
    assert ex["question"]["options"][ex["suggested"]] == other["options"][other["answer"]] != ex["question"]["options"][ex["question"]["answer"]]


def test_a_pasted_question_card_keeps_its_own_question():
    src = server.get_bank().by_id["PH-07208"]
    q = {"id": "NEW-ABC12", "question": src["question"], "options": src["options"], "answer": 1, "explanation": "", "topic": "Pasted question"}
    f = {"key": "key_conflict:NEW-ABC12", "check": "key_conflict", "severity": "high", "source": "exact", "question_id": "NEW-ABC12",
         "related_ids": ["PH-07208"], "title": "t", "detail": "d", "question": q}
    out = server._enrich(f)
    assert out["question"]["id"] == "NEW-ABC12" and out["suggested"] == 0  # marks the bank copy's key, (A)


def test_chat_streams_events_enriches_findings_and_keeps_history(monkeypatch):
    tm = agent.run_checks.invoke({"type": "tool_call", "id": "c1", "name": "run_checks", "args": {"check": "duplicate_options"}})

    class Fake:
        def stream(self, *_a, **_k):
            yield ("updates", {"model": {"messages": [AIMessage("", tool_calls=[{"name": "run_checks", "args": {}, "id": "c1"}])]}})
            yield ("updates", {"tools": {"messages": [tm]}})
            yield ("messages", (AIMessageChunk("Done."), {"langgraph_node": "model"}))
            yield ("updates", {"model": {"messages": [AIMessage("Done.")]}})

    monkeypatch.setattr(agent, "get_agent", lambda: Fake())
    ev = events(TestClient(server.app).post("/api/chat", json={"session_id": "s1", "message": "find mismatches"}))
    assert [e["type"] for e in ev] == ["tool_start", "tool_end", "token", "done"]
    f = ev[1]["findings"][0]
    assert f["question"]["options"] and f["question"]["id"] == f["question_id"]
    assert "messages" not in ev[-1] and len(server.SESSIONS["s1"]) == 4


def test_chat_rejects_bad_input():
    c = TestClient(server.app)
    assert c.post("/api/chat", json={"session_id": "s", "message": "   "}).status_code == 422
    assert c.post("/api/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
    assert c.post("/api/chat", json={"session_id": "s", "message": "x" * 4001}).status_code == 422


def test_chat_without_key_streams_a_readable_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    agent.get_agent.cache_clear()
    ev = events(TestClient(server.app).post("/api/chat", json={"session_id": "s2", "message": "hi"}))
    agent.get_agent.cache_clear()
    assert ev[-1]["type"] == "error" and "OLLAMA_API_KEY" in ev[-1]["message"]


def test_sessions_are_capped(monkeypatch):
    monkeypatch.setattr(server, "MAX_SESSIONS", 2)

    class Quiet:
        def stream(self, *_a, **_k):
            yield ("updates", {"model": {"messages": [AIMessage("ok")]}})

    monkeypatch.setattr(agent, "get_agent", lambda: Quiet())
    c = TestClient(server.app)
    for sid in ("a1", "a2", "a3"):
        events(c.post("/api/chat", json={"session_id": sid, "message": "hi"}))
    assert "a1" not in server.SESSIONS and {"a2", "a3"} <= set(server.SESSIONS)
    assert c.delete("/api/session/a3").json() == {"ok": True} and "a3" not in server.SESSIONS
