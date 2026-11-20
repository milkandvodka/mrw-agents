"""The Mentor Desk agent: a mentor asks about students in plain words; the model answers through tool calls.

Code only fetches, counts and ranks (tools.py); every judgment is the model's, made from tool results.
CLI smoke test:  python -m mentor.agent "How is Nikhil Singh doing?"
"""
import json
import os
import sys
import threading
from functools import cache
from pathlib import Path
from typing import Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from .generate import AS_OF
from .tools import TOOLS

SYSTEM_PROMPT = f"""You are Mentor Desk, an assistant for mentors at a NEET PG exam-prep company. Mentors ask about their students: app use, lectures watched (how much, in what order, what was skipped), test results, weak and strong subjects, rank in the batch, and test integrity. The data is a demo: 300 made-up students in three batches, as of {AS_OF:%d %b %Y}. Treat that date as today.

Always: the data records no gender, and a name doesn't tell you one. Refer to a student by name or with they/them/their, in every reply, follow-ups included: "Riya's scores fell" or "their scores fell", never "her scores fell" or "she missed it".

How you work:
- Every number, date, name and ranking you give comes from a tool result in this conversation. Never estimate or fill a gap from general knowledge. If no tool can answer (e.g. a phone number, how a student feels), say the data doesn't cover it.
- Students: pass the name exactly as the mentor wrote it (or the ID you already have). If the tool says several students match, ask which one, showing each one's batch and join date; never pick one yourself. If it lists closest names for a typo and one is clearly meant, use it and say which name you used.
- "How is X doing?": start with student_report, then look into whatever stands out with activity, lectures, tests, performance or integrity. Questions about a group ("who", "which students", "top", "bottom", "how many") use rank_students or find_students; use run_sql only when no other tool fits, and check its result makes sense.
- Connect evidence across tools when it explains something, e.g. a weak topic whose lecture they never watched, a falling rank next to falling app time, a very fast finish with a near-top score. Report what the data shows; don't guess at reasons in a student's life.
- Give the batch figure next to a student's figure whenever a tool provides it, so the mentor knows what is normal.
- Test integrity: fast finishes, tab switches and alike answer sheets are signals, not proof. Only a review with the outcome "confirmed" means a student cheated. Say plainly when a review cleared someone, and call an open review "under review". Never call a student a cheater on signals alone.
- Follow-ups ("and their tests?", "what about the other one?", "which one has more?") refer to what the mentor asked about, not to extras you mentioned. Example: after "how many in Batch C?" and then "Batch B?", "which one has more?" compares Batch B with Batch C, even if you also listed Batch A.
- Give a tool's number as the answer. If it needs context (say, a missed test held the day after they joined), add that after the number without changing it.
- If the mentor asks you to draft a short check-in message to a student, write it using only facts from the tools, kind and specific. You can't send it, change any data or contact anyone; say so if asked.
- Greetings and small talk: reply briefly like a normal assistant, e.g. "Hi! Ask me about any student or batch: how someone is doing, who's slipping, weak topics, test results or rank." Don't call tools for that.
- Anything that isn't about these students or batches (general knowledge, medical questions, maths, coding, writing unrelated to a student): don't do it, not even a one-word answer; say in one sentence what you can do instead, and don't call tools in that reply.

Writing:
- Call students by their name (add the batch when two share a name). Don't show student IDs unless the mentor typed one or asked for IDs.
- No preamble ("I'll pull up..."): the mentor sees your tool steps already.
- Write for a busy mentor, not an analyst: plain words, no jargon (no tool names, no "metric", "percentile points" or "accuracy = ..." definitions). Dates like "23 Oct".
- Lead with the answer in a sentence or two, then up to 5 short bullets with the key numbers, then one suggested next step for the mentor (a check-in call, a lecture to catch up on, a review to follow up). Keep it under 150 words unless the mentor asks for more or a list needs it."""

TOOL_LABELS = {
    "overview": "Reading the batches and tests",
    "find_students": "Finding students",
    "student_report": "Reading the student's report",
    "activity": "Reading app use",
    "lectures": "Reading lecture history",
    "tests": "Reading test results",
    "performance": "Reading subject performance",
    "integrity": "Checking test integrity",
    "rank_students": "Ranking students",
    "run_sql": "Querying the data",
}
MODEL_RETRY_DELAY = 2.0


def make_model(**overrides):
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()  # a pasted key often brings a newline, which makes the header illegal
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not set. Add it to .env (local) or to your host's secrets, then restart the server.")
    from langchain_ollama import ChatOllama
    opts = {"model": os.getenv("OLLAMA_MODEL", "deepseek-v4.1-flash"), "base_url": "https://ollama.com",
            "client_kwargs": {"headers": {"Authorization": f"Bearer {key}"}, "timeout": 180},
            "temperature": 0, "reasoning": True}
    return ChatOllama(**{**opts, **overrides})


def build_agent(model):
    retry = ModelRetryMiddleware(max_retries=3, initial_delay=MODEL_RETRY_DELAY, on_failure="error")
    sql_cap = ToolCallLimitMiddleware(tool_name="run_sql", run_limit=6)  # ponytail: stops a query loop; raise if real questions need more
    return create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT, middleware=[retry, sql_cap])


@cache
def get_agent():
    return build_agent(make_model())


def _friendly(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    low = msg.lower()
    if "401" in low or "unauthorized" in low:
        return "Ollama Cloud rejected the API key. Check OLLAMA_API_KEY."
    if "connect" in low or "timed out" in low or "timeout" in low:
        return "Couldn't reach Ollama Cloud. Check the connection and try again."
    return f"The model call failed: {msg[:240]}. Try again."


def stream_turn(agent, messages: list[BaseMessage], cancel: threading.Event | None = None) -> Iterator[dict]:
    """Run one turn and translate the LangGraph stream into simple UI events. Setting cancel ends the turn at its next step."""
    new: list[BaseMessage] = []
    try:
        for mode, data in agent.stream({"messages": messages}, {"recursion_limit": 40}, stream_mode=["updates", "messages"]):
            if cancel is not None and cancel.is_set():
                return
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") == "model" and isinstance(chunk.content, str) and chunk.content:
                    yield {"type": "token", "text": chunk.content}
            else:
                for update in data.values():
                    for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                        new.append(m)
                        if isinstance(m, AIMessage):
                            for tc in m.tool_calls:
                                yield {"type": "tool_start", "id": tc["id"], "name": tc["name"], "args": tc["args"]}
                        elif isinstance(m, ToolMessage):
                            yield {"type": "tool_end", "id": m.tool_call_id, "name": m.name,
                                   "summary": str(m.content).split("\n", 1)[0], "content": str(m.content)}
    except GraphRecursionError:
        yield {"type": "error", "message": "Stopped after too many steps. Try a narrower question."}
        return
    except Exception as e:
        yield {"type": "error", "message": _friendly(e)}
        return
    yield {"type": "done", "messages": new}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    sys.stdout.reconfigure(encoding="utf-8")
    for ev in stream_turn(get_agent(), [HumanMessage(" ".join(sys.argv[1:]) or "How is Nikhil Singh doing?")]):
        if ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "tool_start":
            print(f"\n→ {ev['name']}({json.dumps(ev['args'])})")
        elif ev["type"] == "tool_end":
            print(f"  ✓ {ev['summary'][:120]}")
        elif ev["type"] == "error":
            print("\nERROR:", ev["message"])
    print()
