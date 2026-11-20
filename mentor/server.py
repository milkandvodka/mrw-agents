"""FastAPI server: streams the Mentor Desk agent over SSE and serves the built React app.

Run locally:  uvicorn app:app --reload  (from the repo root)   (and `npm run dev` in frontend/ for hot reload)
"""
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import AfterValidator, BaseModel, StringConstraints

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent / ".env")

from . import agent  # noqa: E402  (after .env so the key is visible)
from . import tools  # noqa: E402
from .generate import AS_OF  # noqa: E402

DIST = ROOT / "frontend" / "dist"
MAX_SESSIONS = 500
SESSIONS: OrderedDict[str, list] = OrderedDict()  # ponytail: in-memory, one process; move to Redis if you run several workers
RUNNING: dict[str, threading.Event] = {}  # session -> cancel flag of the turn it's running
_n = tools.student_name
EXAMPLES = [  # what a mentor might ask first; each leads to a planted story
    f"How is {_n('S042')} doing?",
    f"Is {_n('S199')} improving?",
    f"When was {_n('S017')} last on the app?",
    f"Has {_n('S105')} ever cheated?",
    f"Does {_n('S063')} finish the lectures they start?",
    f"What's {_n('S011')}'s weakest subject?",
    f"Anything suspicious about {_n('S150')}'s tests?",
    f"How has {_n('S222')} been doing lately?",
    f"Is {_n('S088')} following the course order?",
    "Who in Batch A has gone quiet lately?",
    "Has anyone cheated in Grand Test 3?",
    "How is Priya Sharma doing?",
]

app = FastAPI(title="Mentor Desk")


def _session_id(s: str) -> str:
    if not 1 <= len(s) <= 64 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in s):
        raise ValueError("session_id: 1-64 letters, digits, - or _")
    return s


class ChatIn(BaseModel):
    session_id: Annotated[str, AfterValidator(_session_id)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


@app.get("/api/overview")
def overview() -> dict:
    batches = tools.q("SELECT batch, count(*) FROM students GROUP BY batch ORDER BY batch")
    return {"students": sum(n for _b, n in batches), "batches": [b for b, _n in batches], "as_of": AS_OF.isoformat(),
            "examples": EXAMPLES, "has_key": bool(os.getenv("OLLAMA_API_KEY"))}


@app.get("/api/students")
def students() -> list[dict]:
    return tools.roster()


@app.get("/api/students/{sid}")
def student(sid: str) -> dict:
    d = tools.student_detail(sid)
    if d is None:
        raise HTTPException(404, "No such student")
    return d


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
