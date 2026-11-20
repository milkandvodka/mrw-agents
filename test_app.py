import threading
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app import app
from mentor import agent as mentor_agent, server as mentor_server
from qbank import agent as qbank_agent, server as qbank_server
from qbank.bank import get_bank

client = TestClient(app)
FAKE_KEY = "f00dfeed0123456789abcdef.FakeKeyForTests-only"
AGENTS = [("qbank", qbank_agent), ("mentor", mentor_agent)]
SERVERS = [("qbank", qbank_server), ("mentor", mentor_server)]


@pytest.mark.parametrize("prefix,agent", AGENTS)
def test_a_turn_stops_streaming_once_cancelled(prefix, agent):
    cancel, out = threading.Event(), []

    class Endless:
        def stream(self, *_a, **_k):
            while True:
                yield ("messages", (AIMessageChunk("x"), {"langgraph_node": "model"}))

    for ev in agent.stream_turn(Endless(), [], cancel):
        out.append(ev)
        if len(out) == 3:
            cancel.set()
    assert [e["type"] for e in out] == ["token"] * 3  # no "done": a cancelled turn isn't saved as finished


@pytest.mark.parametrize("prefix,server", SERVERS)
def test_going_back_cancels_the_run_still_going_on_the_server(prefix, server):
    running = threading.Event()
    server.RUNNING["going-back"] = running
    assert client.delete(f"/{prefix}/api/session/going-back").json() == {"ok": True}
    assert running.is_set() and "going-back" not in server.RUNNING


def test_a_review_stops_calling_the_model_once_the_user_goes_back(monkeypatch):
    cancel, calls = threading.Event(), []

    class Reviewer:
        def invoke(self, *_a):
            calls.append(1)
            cancel.set()  # the user goes back while the first batches are out
            time.sleep(0.05)
            return AIMessage("[]")

    monkeypatch.setattr(qbank_agent, "make_model", lambda **_: Reviewer())
    monkeypatch.setattr(qbank_agent, "_cancel_event", lambda: cancel)
    qs = list(get_bank().by_id.values())[:12 * 10]  # 10 batches
    qbank_agent._review(qs, False, "ai_review", "t1")
    time.sleep(0.3)  # calls already out may finish
    assert 1 <= len(calls) <= 4  # at most the ones already holding a slot; the other batches never call the model


def test_a_cancelled_review_frees_its_slot_without_calling_the_model():
    cancel, calls = threading.Event(), []
    cancel.set()

    class Spy:
        def invoke(self, *_a):
            calls.append(1)
            return AIMessage("[]")

    assert qbank_agent.review_batch(Spy(), [get_bank().by_id["PH-00143"]], cancel=cancel) is None and calls == []


@pytest.mark.parametrize("prefix,agent", AGENTS)
def test_a_pasted_key_with_a_trailing_newline_still_makes_a_valid_header(monkeypatch, prefix, agent):
    monkeypatch.setenv("OLLAMA_API_KEY", FAKE_KEY + "\n")
    assert agent.make_model().client_kwargs["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"


@pytest.mark.parametrize("prefix,agent", AGENTS)
def test_the_key_never_reaches_the_browser_even_inside_an_error(monkeypatch, prefix, agent):
    monkeypatch.setenv("OLLAMA_API_KEY", FAKE_KEY)
    monkeypatch.setattr(agent, "get_agent", lambda: None)
    monkeypatch.setattr(agent, "stream_turn", lambda *_a: iter(
        [{"type": "error", "message": f"Illegal header value b'Bearer {FAKE_KEY}\\n'"}]))
    body = client.post(f"/{prefix}/api/chat", json={"session_id": "leak-check", "message": "hi"}).text
    assert "Illegal header value" in body and FAKE_KEY not in body


def test_both_agents_answer_under_their_own_prefix():
    q = client.get("/qbank/api/overview")
    m = client.get("/mentor/api/overview")
    assert q.status_code == m.status_code == 200
    assert q.json()["questions"] > 13000 and m.json()["students"] == 300


def test_an_unknown_student_is_a_404_not_the_page():
    assert client.get("/mentor/api/students/nobody").status_code == 404
