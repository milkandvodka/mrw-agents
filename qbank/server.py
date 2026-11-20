"""FastAPI server: streams the auditor agent over SSE and serves the built React app.

Run locally:  uvicorn app:app --reload  (from the repo root)   (and `npm run dev` in frontend/ for hot reload)
"""
import json
import os
import threading
from collections import OrderedDict
from functools import cache
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import AfterValidator, BaseModel, StringConstraints

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent / ".env")

from . import agent  # noqa: E402  (after .env so the key is visible)
from .bank import get_bank  # noqa: E402
from .checks import L, norm  # noqa: E402

DIST = ROOT / "frontend" / "dist"
PLANTED = json.loads((ROOT / "data" / "planted.json").read_text(encoding="utf-8"))
# Chips on the first screen, all answered in seconds. (A topic's AI review takes minutes, so it isn't offered: ask for one.)
AUDIT = ("Audit the demo bank", "Audit the demo bank: run the exact checks on every question and tell me what to fix first.")
ASKS = [(q, q) for q in ("Which questions in the demo bank appear twice with different answer keys?",
                         "Which questions in the demo bank list the same option twice?",
                         "What topics does the demo bank cover?",
                         "Find questions on beta blockers in the demo bank")]
PASTE_LABELS = [  # one per examples() entry, in the same order
    "Check a question the bank has with another key",
    "Check a question whose explanation disagrees with its key",
    "Check a question with a repeated option",
    "Check a question with a wrong key",
    "Check two questions with no problems",
]
HELDOUT = ROOT / "data" / "heldout.json"
MAX_SESSIONS = 500
SESSIONS: OrderedDict[str, list] = OrderedDict()  # ponytail: in-memory, one process; move to Redis if you run several workers
RUNNING: dict[str, threading.Event] = {}  # session -> cancel flag of the turn it's running

app = FastAPI(title="QBank Auditor")


def _session_id(s: str) -> str:
    if not 1 <= len(s) <= 64 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in s):
        raise ValueError("session_id: 1-64 letters, digits, - or _")
    return s


class ChatIn(BaseModel):
    session_id: Annotated[str, AfterValidator(_session_id)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


def _enrich(f: dict) -> dict:
    """Attach the question so the UI can render the card without another request. Pasted questions bring their own."""
    b = get_bank()
    q = f.get("question") or b.by_id[f["question_id"]]
    out = {**f, "question": {**q, "explanation": q["explanation"][:400]}}
    if f["check"] == "key_conflict" and f["related_ids"] and f["related_ids"][0] in b.by_id:  # mark the other copy's key, matched by text
        other = b.by_id[f["related_ids"][0]]
        want = norm(other["options"][other["answer"]])
        out["suggested"] = next((i for i, o in enumerate(q["options"]) if norm(o) == want and i != q["answer"]), None)
    return out


def _paste(q: dict, key: int | None = None, explanation: bool = False) -> str:
    """A question written out the way a faculty member would paste it."""
    text = q["question"].strip() + "".join(f"\n{L[i]}. {o}" for i, o in enumerate(q["options"]))
    text += f"\nAnswer: {L[q['answer'] if key is None else key]}"
    return text + (f"\nExplanation: {q['explanation'].strip()}" if explanation and q["explanation"] else "")


@cache
def examples() -> list[str]:
    """Real questions to try, each showing a different kind of problem, and one pair with none."""
    bank = get_bank().by_id
    past = {q["id"]: q for q in json.loads(HELDOUT.read_text(encoding="utf-8"))}
    return [
        _paste(bank["PH-11986"]),  # keyed metronidazole; its copy PH-07208 in the bank says oral vancomycin
        _paste(bank["PH-04618"], explanation=True),  # its own explanation says competitive, the key says non-competitive
        _paste(bank["PH-00143"]),  # simvastatin listed twice
        _paste(past["PYQ-0067"], key=0),  # a past-paper question not in the bank, pasted with a wrong key
        _paste(past["PYQ-0106"]) + "\n\n" + _paste(past["PYQ-0109"]),  # two correct past-paper questions at once
    ]


@cache
def suggestions() -> list[dict]:
    chips = [AUDIT, *zip(PASTE_LABELS, examples()), *ASKS]
    return [{"label": label, "prompt": prompt} for label, prompt in chips]


@app.get("/api/overview")
def overview() -> dict:
    o = get_bank().overview()
    return {"questions": o["questions"], "topics": o["topics"], "counts": o["counts"], "suggestions": suggestions(),
            "examples": examples(), "planted": PLANTED, "has_key": bool(os.getenv("OLLAMA_API_KEY"))}


def _sse(ev: dict) -> str:
    data = json.dumps(ev, ensure_ascii=False)
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if len(key) >= 16:  # an error can quote the request's auth header: everything bound for a browser passes here
        data = data.replace(key, "[key hidden]")
    return f"data: {data}\n\n"


@app.post("/api/chat")
def chat(body: ChatIn) -> StreamingResponse:
    history = SESSIONS.get(body.session_id, [])
    human = HumanMessage(body.message)
    cancel = threading.Event()
    if old := RUNNING.get(body.session_id):
        old.set()  # a new message replaces whatever this chat was still running
    RUNNING[body.session_id] = cancel

    def events():
        try:
            ag = agent.get_agent()
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return
        try:
            for ev in agent.stream_turn(ag, history + [human], cancel):
                if ev["type"] == "done":
                    SESSIONS[body.session_id] = history + [human, *ev["messages"]]
                    SESSIONS.move_to_end(body.session_id)
                    while len(SESSIONS) > MAX_SESSIONS:
                        SESSIONS.popitem(last=False)
                    yield _sse({"type": "done"})
                elif ev["type"] == "tool_end":
                    yield _sse({**ev, "findings": [_enrich(f) for f in ev["findings"]]})
                else:
                    yield _sse(ev)
        finally:  # also runs when the browser stops listening (Stop, a closed tab): the agent stops working for nobody
            cancel.set()
            if RUNNING.get(body.session_id) is cancel:
                del RUNNING[body.session_id]

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str) -> dict:
    """Back to the first screen: forget the chat and stop anything it's still running."""
    SESSIONS.pop(session_id, None)
    if running := RUNNING.pop(session_id, None):
        running.set()
    return {"ok": True}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="app")
