"""The QBank Auditor agent: tools over the bank, an AI reviewer, and a UI-friendly event stream.

Code only fetches data and compares text exactly; every judgment is made by the model, through tool calls.
CLI smoke test:  python -m qbank.agent "Run a full quality audit of the Pharmacology bank"
"""
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches
from functools import cache
from pathlib import Path
from typing import Annotated, Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from .bank import _key, get_bank
from .checks import CHECKS, L, norm, run_all_checks

MAX_CHARS = 4000
REVIEW_BATCH = 12  # with thinking on: ~3 s per question per call; smaller batches + more workers = faster wall clock
REVIEW_WORKERS = 8
# Ollama Cloud rejects bursts with 429 "too many concurrent requests"; the limit depends on your plan.
# One cap shared by every AI review in this process (calibration knob: REVIEW_CONCURRENCY).
REVIEW_SLOTS = threading.BoundedSemaphore(int(os.getenv("REVIEW_CONCURRENCY", "4")))
RETRY_DELAY = 2.0  # reviewer backoff after a failed call: 2 s, then 4 s
MODEL_RETRY_DELAY = 2.0  # agent model calls: up to 3 retries with exponential backoff

TOOL_LABELS = {
    "check_questions": "Checking your questions",
    "bank_overview": "Reading the bank overview",
    "list_topics": "Listing topics",
    "search_questions": "Searching questions",
    "get_questions": "Reading questions",
    "run_checks": "Running exact checks",
    "find_similar": "Looking for similar questions",
    "ai_review": "AI reviewing questions",
    "flag_question": "Adding to the review queue",
}

SYSTEM_PROMPT = """You are QBank Auditor, an assistant for the faculty and content team of a medical exam-prep company (NEET PG style MCQs).
You put the problems you find in a review queue, and a faculty member confirms or dismisses every item. You can do two things:
1. Check MCQs the user pastes. Call check_questions with each question split into its stem, options and key letter, copied exactly as written (don't fix typos or reword). If the user didn't say which option is the answer, ask for the key before checking. If they only ask whether a pasted question is already in the bank, find_similar is enough.
2. Audit the loaded demo bank: 13,760 real pharmacology questions from past NEET PG and AIIMS papers (public MedMCQA data), with the other tools. Call it "the demo bank" when you talk about it.
If the user greets you, thanks you or makes small talk, reply briefly like a normal assistant, e.g. "Hi! Paste MCQs with their answer keys and I'll check them, or I can audit the demo bank of past NEET PG questions. What would you like to do?" Don't call tools for that, and don't list checks or bank statistics nobody asked for.

How you work:
- Use tools for everything about the bank. Never invent questions, ids or counts.
- Two kinds of evidence. run_checks compares text exactly: the same question saved with different keys, an option listed twice, an empty option. ai_review has an AI reviewer read each question with its explanation and judge it: a wrong key, an explanation that backs a different option, two defensible answers, and (if asked) badly worded items. When you report a judgment, say it is the AI's opinion ("the AI thinks…"); text comparisons you can state as facts. Anything you flag yourself after reading is an AI opinion, however obvious it looks.
- For an audit or check request, start broad (bank_overview or run_checks), then dig into specifics with get_questions and find_similar.
- ai_review is slow (about a minute per 100 questions). Run it when the user asks for a review of keys, explanations or wording, asks about a few specific questions, or asks you to check or audit one named topic (then run the exact checks and the AI review on that topic); otherwise offer it as the next step. Review one topic (up to 300 questions) or a list of ids at a time: only one AI review runs per message, so offer the next topic as a follow-up. A request to audit or check the whole bank means run_checks across all questions, not an AI review; finish with the exact-check results and offer the AI review for a named topic. Never AI-review "Unspecified": it is the 6,800-question catch-all and a review would cover only a sliver of it.
- run_checks and ai_review add their findings to the review queue automatically. Use flag_question only for problems you find yourself by reading questions, and cite the evidence (ids, option text) in the reason.
- Check every topic the user names with list_topics before using it, and every drug or word with search_questions. If it isn't in the bank exactly (ignoring case and punctuation), your first sentence must say so, even when you go on to use the closest match. Example: asked to "check the Neuro-ABC topic", start with "There is no topic called Neuro-ABC; the closest is Central Nervous system, so I checked that." Never say a word or spelling isn't in the bank when a tool found questions containing it; misspellings in the bank are real bank text.
- Follow-ups ("and in X?", "which of those two?", "was the first one ok?") refer to what the user asked about or pasted earlier in the conversation, not to extras you mentioned in passing. Example: the user asks how many questions are in Anesthesia, then in Hematology, then "which of those two has more?": compare Anesthesia with Hematology, even if you also mentioned Anaesthesia or Hematology and CVS.
- When a topic exists under the name the user gave, answer for that topic only. Don't mention or offer look-alike topics (e.g. C.N.S next to Central Nervous system) unless the name the user gave doesn't exist or they ask about similar topics.
- Do what was asked and no more: a question about how many gets the number, without running checks or reviews; offer those as the next step instead.
- If the user states something about the bank that a tool result contradicts (a question's topic, key or number of options, a count), correct it in your first sentence using the tool result, e.g. "That question is in Endocrinology, not Kidney." Never repeat the user's version as fact.
- Stick to what is in the bank. Don't propose new content or coverage gaps, and don't state medical facts of your own; when a medical judgment is needed, show the evidence and leave the call to faculty.
- If asked a general medical or pharmacology question, don't answer it. You may list the bank questions on it (their text and keys as the bank words them), but don't summarise them into a conclusion or add facts, comparisons or explanations of your own.
- After check_questions, say for each pasted question what is wrong, if anything, and whether the bank already has it, without answering the question yourself. Mention only problems, not the checks that found nothing; if nothing is wrong, say so in one line.
- You only check exam questions: MCQs the user pastes, on any subject, and the bank's. If asked for anything else (answering a general-knowledge question, maths, translation, writing, coding, personal advice), don't do it, not even a one-word answer; say in one sentence what you can do instead. Don't call tools or carry on earlier work in that reply; wait for the user's next request.
- When the bank disagrees with itself (two copies with different keys, an explanation contradicting its key), quote both sides and leave the medical call to faculty.
- If asked for clinical or patient advice (doses, treatment choices), decline in one or two sentences and say what you can do instead; don't search the bank for it.
- You do not write or rewrite questions, explanations or answer keys, and you do not give clinical advice. If asked, say faculty should make that change and offer to pull together the evidence they need.

Question IDs: if the user typed a question ID anywhere in the conversation or asked for IDs, give the ID of every bank question you mention (including copies and matches you found), copied exactly from a tool result, in the order the tool returned them. Otherwise leave IDs out, because they mean nothing to most users: name a question by a few words of its text in quotes, e.g. "the pseudomembranous colitis question". Never show the ids of pasted questions (NEW-…): call them "your question", or "question 2" in the order they were pasted.
Style: write for a busy reader who is not technical. Short, plain words, no jargon ("exact match", "similarity", tool names). Lead with the answer, then at most 3-5 items, then one suggested next step. Markdown bullets, no preamble.
After run_checks, ai_review or check_questions, keep the reply under 120 words: the review queue already shows every problem with its question."""

ISSUES = {  # what the AI reviewer may report -> (severity, card title)
    "wrong_key": ("high", "AI thinks the key is wrong"),
    "explanation_conflict": ("high", "Explanation backs a different option"),
    "ambiguous": ("medium", "Two defensible answers"),
    "flawed_wording": ("low", "Badly worded item"),
}
REVIEW_PROMPT = """You are helping pharmacology faculty audit exam MCQs. For each question, read the stem, the options, the key and the
bank's explanation (if any), and report at most one problem, the most serious:
- "wrong_key": you are confident the keyed option is wrong. suggested = the option you think is right.
- "explanation_conflict": the bank's own explanation supports a different option than the key. suggested = that option.
- "ambiguous": two options are equally defensible. suggested = the other one.{wording}
Answer "ok" unless you are confident. Reply with ONLY a JSON array, one object per question:
[{{"id": "...", "verdict": "ok" | "problem", "issue": {issues} | null, "suggested": "a" | "b" | "c" | "d" | null, "reason": "one short sentence"}}]

Questions:
"""
WORDING = ('\n- "flawed_wording": the item is badly written, e.g. an "all/none of the above" option, a negative stem that isn\'t '
           "emphasised, a stem that doesn't ask anything, or a clue that gives the answer away. suggested = null.")


# ---------- helpers ----------

PLAIN = ("Reply in plain words: leave these ids out unless the user typed a question ID or asked for IDs. "
         "The review queue already shows each problem with its question.")


def _cap(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 40].rstrip() + "\n… (truncated)"


def _short(s: str, n: int = 140) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _key_text(q: dict) -> str:
    return f"({L[q['answer']]}) {q['options'][q['answer']]}"


def _topic_miss(topic: str) -> str:
    names = [t for t, _ in get_bank().topics()]
    close = get_close_matches(topic, names, n=5, cutoff=0.3)
    return (f"No topic matches '{topic}'. Closest topics: {', '.join(close) or 'none'}. Call bank_overview to see all topics. "
            f"Your reply must start by telling the user there is no topic called '{topic}', and name any closest topic you use instead.")


def _covered(topic: str) -> str:
    return "topic " + ", ".join(f"{t} ({n})" for t, n in get_bank().topics(topic))


def _cancel_event() -> threading.Event | None:
    """The cancel flag stream_turn puts in the run's config (set when the user goes back or stops); None outside a run."""
    try:
        from langgraph.config import get_config
        return get_config().get("configurable", {}).get("cancel")
    except Exception:
        return None


def _progress(data: dict) -> None:
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()(data)
    except Exception:
        pass  # not running inside the graph (CLI/tests)


def make_model(**overrides):
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()  # a pasted key often brings a newline, which makes the header illegal
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not set. Add it to .env (local) or to your host's secrets, then restart the server.")
    from langchain_ollama import ChatOllama
    opts = {"model": os.getenv("OLLAMA_MODEL", "deepseek-v4.1-flash"), "base_url": "https://ollama.com",
            "client_kwargs": {"headers": {"Authorization": f"Bearer {key}"}, "timeout": 180},
            "temperature": 0, "reasoning": True}
    return ChatOllama(**{**opts, **overrides})


# ---------- tools ----------

@tool(response_format="content_and_artifact")
def bank_overview() -> tuple[str, list]:
    """Size of the bank, its biggest topics, and how many problems the exact checks found. Call this first for broad requests."""
    o = get_bank().overview()
    c = o["counts"]
    lines = [f"Bank: {o['questions']:,} Pharmacology questions across {o['topics']} topics.",
             "Top topics: " + ", ".join(f"{t} ({n})" for t, n in o["top_topics"]),
             "This is not every topic: call list_topics to check any topic the user names.",
             "Exact check results (not in the queue until run_checks is called):"]
    lines += [f"- {k}: {c.get(k, 0)} ({sev})" for k, sev in CHECKS.items()]
    lines.append(f"Info only: {c.get('exact_duplicate_groups', 0):,} groups of exact duplicates that agree on the key.")
    lines.append("Wrong keys, explanations that back another option, and badly worded items need ai_review (judgment, per topic).")
    return _cap("\n".join(lines)), []


@tool(response_format="content_and_artifact")
def list_topics(match: str = "") -> tuple[str, list]:
    """Every topic name in the bank with its question count, or every name containing match (ignoring case/punctuation).
    Use it to check that a topic the user names exists, and to get its exact name and size. Pass the name as the user
    wrote it, without shortening it."""
    ts = [(t, n) for t, n in get_bank().topics() if _key(match) in _key(t)]
    if not ts:
        return _topic_miss(match), []
    out = f"{len(ts)} topic{'s' if len(ts) != 1 else ''}: " + ", ".join(f"{t} ({n})" for t, n in ts)
    if match and len(ts) > 1:  # look-alikes: the user's own words pick one; the search text may have been shortened
        out += (". These are separate topics. Answer about the one whose name is what the user wrote, ignoring case and "
                "punctuation, and don't mention the others unless the user asks about similar topics or none of them is "
                "what the user wrote.")
    return _cap(out), []


@tool(response_format="content_and_artifact")
def search_questions(query: str = "", topic: str = "", limit: int = 10) -> tuple[str, list]:
    """Find questions by keyword/phrase (ranked by relevance) and/or topic (exact name, or partial e.g. 'autonomic'). limit ≤ 30.
    The first line says how many questions in total contain the words, even when fewer are shown."""
    b = get_bank()
    if topic and not b.topics(topic):
        return _topic_miss(topic), []
    hits = b.search(query, topic, max(1, min(limit, 30)))
    where = f" in {_covered(topic)}" if topic else ""
    if not hits:
        return f"0 questions contain '{query}'{where}." if query.strip() else f"No questions{where}.", []
    total = len(b.search(query, topic, 100_000))
    if query.strip():
        words = "" if len(query.split()) == 1 else "at least one word of "
        head = f"{total} questions contain {words}'{query}'{where}"
    else:
        head = f"{total} questions{where}"
    head += f"; showing the top {len(hits)}:" if total > len(hits) else ":"
    return _cap("\n".join([head] + [f"{q['id']} [{q['topic']}] {_short(q['question'])} | key {_key_text(q)}" for q in hits])), []


@tool(response_format="content_and_artifact")
def get_questions(ids: list[str]) -> tuple[str, list]:
    """Read up to 10 questions in full: stem, options, key and the start of the explanation."""
    b = get_bank()
    found = [b.by_id[i] for i in ids[:10] if i in b.by_id]
    missing = [i for i in ids[:10] if i not in b.by_id]
    blocks = []
    for q in found:
        opts = "\n".join(f"  {L[i]}) {o or '(blank)'}" for i, o in enumerate(q["options"]))
        blocks.append(f"{q['id']} · {q['topic']}\nQ: {_short(q['question'], 400)}\n{opts}\n  Key: {L[q['answer']]}"
                      f"\n  Explanation: {_short(q['explanation'], 220) or '(none)'}")
    if missing:
        blocks.append("Not found: " + ", ".join(missing))
    return _cap("\n\n".join(blocks) or "No ids given."), []


@tool(response_format="content_and_artifact")
def run_checks(check: str = "all", topic: str = "") -> tuple[str, list]:
    """Run the exact checks (text comparison, no judgment) and add their findings to the review queue:
    key_conflict (the same question saved with different keys), duplicate_options, blank_option. check: 'all' or one of those."""
    b = get_bank()
    if check != "all" and check not in CHECKS:
        return f"Unknown check '{check}'. Use 'all' or one of: {', '.join(CHECKS)}.", []
    names = None
    if topic:
        names = {t for t, _ in b.topics(topic)}
        if not names:
            return _topic_miss(topic), []
    fs = [f for f in b.findings
          if (check == "all" or f["check"] == check) and (names is None or b.by_id[f["question_id"]]["topic"] in names)]
    fs.sort(key=lambda f: f["severity"] != "high")
    counts = {}
    for f in fs:
        counts[f["check"]] = counts.get(f["check"], 0) + 1
    scope = _covered(topic) if topic else f"all {len(b.items):,} questions"
    lines = [f"Checked {scope}: {len(fs)} findings added to the review queue (exact match)."]
    lines += [f"- {k} ({CHECKS[k]}): {n}" for k, n in counts.items()]
    if fs:
        lines.append("Examples:")
        lines += [f"{f['question_id']}" + (f" (copies: {', '.join(f['related_ids'])})" if f["related_ids"] else "")
                  + f" · {f['title']} — {_short(f['detail'], 200)}" for f in fs[:12]]
        lines.append(PLAIN)
    return _cap("\n".join(lines)), fs


@tool(response_format="content_and_artifact")
def find_similar(question_id: str = "", text: str = "", limit: int = 5) -> tuple[str, list]:
    """Find questions most similar to a question id or to free text. Useful for spotting duplicates and checking whether near-copies share the same key."""
    b = get_bank()
    if question_id and question_id not in b.by_id:
        return f"Question {question_id} not found.", []
    hits = b.similar(question_id, text, max(1, min(limit, 15)))
    if not hits:
        return "Nothing similar found (give a question_id or some text).", []
    lines = []
    if question_id:
        src = b.by_id[question_id]
        lines.append(f"Source {question_id}: {_short(src['question'])} | key {_key_text(src)}")
    lines += [f"{q['id']} (similarity {s:.2f}) [{q['topic']}] {_short(q['question'])} | key {_key_text(q)}" for q, s in hits]
    return _cap("\n".join(lines)), []


def review_batch(model, questions: list[dict], wording: bool = False, cancel: threading.Event | None = None) -> list[dict] | None:
    """Ask the model to review a batch of questions. Returns its verdicts (problems with an unknown issue dropped), or None
    if the reply can't be parsed after retries or the run was cancelled."""
    stopped = lambda: cancel is not None and cancel.is_set()  # noqa: E731
    allowed = [k for k in ISSUES if wording or k != "flawed_wording"]
    prompt = REVIEW_PROMPT.format(wording=WORDING if wording else "", issues=" | ".join(f'"{k}"' for k in allowed))
    body = "\n\n".join(
        f"id: {q['id']}\nQ: {q['question']}\n" + "\n".join(f"{L[i].lower()}) {o}" for i, o in enumerate(q["options"]))
        + f"\nkey: {L[q['answer']].lower()}\nexplanation: {_short(q['explanation'], 500) or '(none)'}" for q in questions)
    ids = {q["id"] for q in questions}
    for attempt in range(3):  # retries cover rate limits, dropped connections and unreadable replies
        if stopped():
            return None
        try:
            with REVIEW_SLOTS:
                if stopped():  # waited for a slot behind a run that's since been cancelled: pass the slot on
                    return None
                text = model.invoke(prompt + body).content
        except Exception as e:
            err = e
            time.sleep(RETRY_DELAY * 2 ** attempt)
            continue
        try:
            text = text if isinstance(text, str) else json.dumps(text)
            data = json.loads(text[text.index("["): text.rindex("]") + 1])
            return [d for d in data if isinstance(d, dict) and d.get("id") in ids
                    and (d.get("verdict") == "ok" or (d.get("verdict") == "problem" and d.get("issue") in allowed))]
        except Exception as e:
            err = e
    print(f"[ai_review] batch of {len(questions)} skipped: {type(err).__name__}: {str(err)[:200]}", file=sys.stderr)
    return None


def _review(qs: list[dict], wording: bool, tool_name: str, tool_call_id: str) -> tuple[list[dict], int, int]:
    """Run the AI reviewer over qs in parallel batches. Returns (verdicts, batches skipped, questions reviewed).
    Raises RuntimeError when there is no API key."""
    model, cancel = make_model(), _cancel_event()
    batches = [qs[i: i + REVIEW_BATCH] for i in range(0, len(qs), REVIEW_BATCH)]
    verdicts, skipped, done, reviewed = [], 0, 0, 0
    pool = ThreadPoolExecutor(REVIEW_WORKERS)
    futures = {pool.submit(review_batch, model, batch, wording, cancel): batch for batch in batches}
    try:
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                skipped += 1
            else:
                verdicts += res
                reviewed += len(futures[fut])
            done += len(futures[fut])
            _progress({"tool": tool_name, "id": tool_call_id, "done": done, "total": len(qs)})
            if cancel is not None and cancel.is_set():
                break  # nobody is waiting for this any more
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # queued batches never start; running ones stop at their next check
    return verdicts, skipped, reviewed


def _ai_findings(verdicts: list[dict], by_id: dict[str, dict]) -> list[dict]:
    findings = []
    for v in verdicts:
        if v["verdict"] != "problem":
            continue
        q = by_id[v["id"]]
        sev, title = ISSUES[v["issue"]]
        s = str(v.get("suggested") or "").strip().upper()[:1]
        suggested = L.index(s) if s and s in L[:len(q["options"])] and L.index(s) != q["answer"] else None
        alt = f" points to ({s}) {q['options'][suggested]}" if suggested is not None else ""
        reason = _short(v.get("reason", ""), 200)
        findings.append({"key": f"ai:{q['id']}", "check": f"ai_{v['issue']}", "severity": sev, "source": "ai",
                         "question_id": q["id"], "related_ids": [], "suggested": suggested, "title": title, "reason": reason,
                         "detail": f"Key is {_key_text(q)}; AI opinion{alt}: {reason} Verify before changing."})
    findings.sort(key=lambda f: ("high", "medium", "low").index(f["severity"]))
    return findings


@tool(response_format="content_and_artifact")
def ai_review(topic: str = "", ids: list[str] | None = None, include_wording: bool = False, max_questions: int = 300,
              tool_call_id: Annotated[str, InjectedToolCallId] = "") -> tuple[str, list]:
    """Have an AI reviewer read every question in a topic (or a list of ids) with its explanation, and judge it: a wrong key,
    an explanation that backs a different option, or two defensible answers. include_wording=True also flags badly worded
    items. Problems go to the review queue as 'AI opinion'. About 1 minute per 100 questions; max_questions ≤ 300."""
    b = get_bank()
    if ids:
        qs = [b.by_id[i] for i in ids if i in b.by_id]
        if not qs:
            return "None of those ids were found.", []
    elif topic:
        if not b.topics(topic):
            return _topic_miss(topic), []
        qs = b.search("", topic, 100_000)
    else:
        return "Give a topic or a list of ids to review.", []
    qs = qs[: max(1, min(max_questions, 300))]
    try:
        verdicts, skipped, reviewed = _review(qs, include_wording, "ai_review", tool_call_id)
    except RuntimeError as e:
        return str(e), []
    findings = _ai_findings(verdicts, b.by_id)
    scope = _covered(topic) if topic and not ids else f"{len(qs)} selected questions"
    n = len(findings)
    by_issue = {}
    for f in findings:
        by_issue[f["title"]] = by_issue.get(f["title"], 0) + 1
    lines = [f"AI reviewed {reviewed} questions in {scope}: {n} problem{'s' if n != 1 else ''} added to the queue as AI opinions."]
    lines += [f"- {t}: {k}" for t, k in by_issue.items()]
    if skipped:
        lines.append(f"{skipped} batch{'es' if skipped != 1 else ''} skipped (the reviewer's reply was unreadable) — run again to retry them.")
    lines += [f"{f['question_id']} · {f['title']}: {_short(f['detail'], 220)}" for f in findings[:15]]
    if findings:
        lines.append(PLAIN)
    return _cap("\n".join(lines)), findings


@tool(response_format="content_and_artifact")
def flag_question(question_id: str, issue: str, reason: str, severity: str = "medium",
                  related_ids: list[str] | None = None) -> tuple[str, list]:
    """Add a problem you found yourself (by reading questions) to the review queue. reason must cite evidence. severity: high | medium | low."""
    b = get_bank()
    if question_id not in b.by_id:
        return f"Question {question_id} not found; nothing flagged.", []
    sev = severity if severity in ("high", "medium", "low") else "medium"
    f = {"key": f"agent_flag:{question_id}", "check": "agent_flag", "severity": sev, "source": "agent",
         "question_id": question_id, "related_ids": [i for i in (related_ids or []) if i in b.by_id],
         "title": _short(issue, 80), "detail": _short(reason, 400)}
    return f"Added {question_id} to the review queue ({sev}).", [f]


class PastedQuestion(BaseModel):
    question: str = Field(description="The stem, exactly as the user wrote it")
    options: list[str] = Field(description="The options in order, without their letters (2 to 4)")
    key: str = Field(description="Letter of the option the user marked as the answer, e.g. 'B'. If they wrote the answer as "
                                 "text (e.g. 'key: Ethambutol'), pass that option's letter.")
    explanation: str = Field("", description="The user's explanation, if they gave one")


def pasted_id(stem: str) -> str:
    """The same question always gets the same id, so pasting it twice doesn't duplicate it in the queue."""
    return "NEW-" + hashlib.sha1(norm(stem).encode()).hexdigest()[:5].upper()


@cache
def _bank_by_stem() -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for q in get_bank().items:
        idx.setdefault(norm(q["question"]), []).append(q)
    return idx


@tool(response_format="content_and_artifact")
def check_questions(questions: list[PastedQuestion], tool_call_id: Annotated[str, InjectedToolCallId] = "") -> tuple[str, list]:
    """Check MCQs the user pasted (not bank questions). Pass each one once, split into stem, options and the key letter,
    copied exactly as written. Runs the exact checks (repeated or empty options), looks for the same question in the bank and
    compares keys, and has the AI reviewer judge the key, explanation and wording. Problems go to the review queue. ≤ 30 questions."""
    qs, errors = [], []
    for n, p in enumerate(questions[:30], 1):
        key = p.key.strip(" ().").upper()
        if not 2 <= len(p.options) <= len(L):
            errors.append(f"Question {n} has {len(p.options)} options; only 2 to {len(L)} are supported.")
        elif len(key) != 1 or key not in L[:len(p.options)]:
            errors.append(f"Question {n}: key '{p.key}' is not one of its option letters. Pass the letter, or ask the user for the key.")
        else:
            qs.append({"id": pasted_id(p.question), "question": p.question.strip(), "options": [o.strip() for o in p.options],
                       "answer": L.index(key), "explanation": p.explanation.strip(), "topic": "Pasted question"})
    if not qs:
        return "\n".join(errors) or "No questions given.", []
    by_id = {q["id"]: q for q in qs}
    findings, _ = run_all_checks(qs)  # repeated/empty options, and two pasted copies with different keys
    notes = {q["id"]: [] for q in qs}
    for q in qs:
        copies = _bank_by_stem().get(norm(q["question"]), [])
        if copies:
            keys = Counter(_key_text(c) for c in copies)
            notes[q["id"]].append(f"In the bank word for word: {len(copies)} cop{'ies' if len(copies) > 1 else 'y'}, keyed "
                                  + "; ".join(f"{k}" + (f" ({n} copies)" if n > 1 else "") for k, n in keys.items()))
            mine, opts = norm(q["options"][q["answer"]]), {norm(o) for o in q["options"]}
            other = [c["id"] for c in copies if norm(c["options"][c["answer"]]) != mine and norm(c["options"][c["answer"]]) in opts]
            if other:
                findings.append({"key": f"key_conflict:{q['id']}", "check": "key_conflict", "severity": "high", "source": "exact",
                                 "question_id": q["id"], "related_ids": other, "title": "The bank has this question with a different key",
                                 "detail": "The bank has it keyed " + " and ".join(
                                     sorted({_key_text(c) for c in copies if c["id"] in other})) + "."})
        else:
            close = get_bank().similar(text=q["question"] + " " + " ".join(q["options"]), limit=3)
            notes[q["id"]].append("Not in the bank word for word. Closest bank questions (judge whether any is the same question): "
                                  + "; ".join(f'"{_short(c["question"], 90)}" keyed {_key_text(c)} (similarity {s:.2f})' for c, s in close))
    try:
        verdicts, skipped, _ = _review(qs, True, "check_questions", tool_call_id)
    except RuntimeError as e:
        verdicts, skipped = [], 0
        errors.append(f"AI review not run: {e}")
    ai = _ai_findings(verdicts, by_id)
    findings += ai
    for f in findings:
        f["question"] = by_id[f["question_id"]]
    lines = [f"Checked {len(qs)} pasted question{'s' if len(qs) != 1 else ''}: {len(findings)} problem{'s' if len(findings) != 1 else ''} "
             "added to the review queue."] + errors
    if skipped:
        lines.append("The AI review of some questions failed (unreadable reply); run the check again for them.")
    for n, q in enumerate(qs, 1):
        lines.append(f"\nQuestion {n}: {_short(q['question'], 100)} | keyed {_key_text(q)}")
        lines += [f"- {n}" for n in notes[q["id"]]]
        mine = [f for f in findings if f["question_id"] == q["id"]]
        lines += [f"- {'Exact match' if f['source'] == 'exact' else 'AI opinion'} · {f['title']}: {_short(f['detail'], 220)}" for f in mine]
        if not any(f["source"] == "ai" for f in mine) and any(v["id"] == q["id"] for v in verdicts):
            lines.append("- AI review: no problem found.")
    lines.append("\nIf the user asks for the ID of a bank question above, look it up with find_similar.")
    return _cap("\n".join(lines)), findings


TOOLS = [check_questions, bank_overview, list_topics, search_questions, get_questions, run_checks, find_similar, ai_review, flag_question]


def build_agent(model):
    retry = ModelRetryMiddleware(max_retries=3, initial_delay=MODEL_RETRY_DELAY, on_failure="error")
    one_review = ToolCallLimitMiddleware(tool_name="ai_review", run_limit=1)  # one AI review per turn: minutes, not an hour
    return create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT, middleware=[retry, one_review])


@cache
def get_agent():
    return build_agent(make_model())


# ---------- streaming for the UI ----------

def _friendly(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    low = msg.lower()
    if "401" in low or "unauthorized" in low:
        return f"Ollama Cloud rejected the API key. Check OLLAMA_API_KEY. ({_short(msg, 160)})"
    if "connect" in low or "timed out" in low or "timeout" in low:
        return f"Couldn't reach Ollama Cloud. Check the connection and try again. ({_short(msg, 160)})"
    return f"The model call failed: {_short(msg, 240)}. Try again."


def stream_turn(agent, messages: list[BaseMessage], cancel: threading.Event | None = None) -> Iterator[dict]:
    """Run one turn and translate the LangGraph stream into simple UI events. Setting cancel ends the turn at its next
    step, and tools that run long (the AI review) stop too."""
    new: list[BaseMessage] = []
    try:
        for mode, data in agent.stream({"messages": messages}, {"recursion_limit": 40, "configurable": {"cancel": cancel}},
                                       stream_mode=["updates", "messages", "custom"]):
            if cancel is not None and cancel.is_set():
                return
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") == "model" and isinstance(chunk.content, str) and chunk.content:
                    yield {"type": "token", "text": chunk.content}
            elif mode == "custom":
                yield {"type": "progress", **data}
            elif mode == "updates":
                for update in data.values():
                    for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                        new.append(m)
                        if isinstance(m, AIMessage):
                            for tc in m.tool_calls:
                                yield {"type": "tool_start", "id": tc["id"], "name": tc["name"], "args": tc["args"]}
                        elif isinstance(m, ToolMessage):
                            yield {"type": "tool_end", "id": m.tool_call_id, "name": m.name,
                                   "summary": str(m.content).split("\n", 1)[0], "content": str(m.content),
                                   "findings": m.artifact or []}
    except GraphRecursionError:
        yield {"type": "error", "message": "Stopped after too many steps. Try a narrower request (one topic or one check)."}
        return
    except Exception as e:
        yield {"type": "error", "message": _friendly(e)}
        return
    yield {"type": "done", "messages": new}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    for ev in stream_turn(get_agent(), [HumanMessage(" ".join(sys.argv[1:]) or "Give me an overview of the bank.")]):
        t = ev["type"]
        if t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "tool_start":
            print(f"\n→ {ev['name']}({json.dumps(ev['args'])})")
        elif t == "tool_end":
            print(f"  ✓ {ev['summary']}  [{len(ev['findings'])} findings]")
        elif t == "progress":
            print(f"  … {ev['done']}/{ev['total']}")
        elif t == "error":
            print("\nERROR:", ev["message"])
    print()
