import json

from langchain_core.messages import AIMessage

from qbank import agent
from qbank import evals


class Scripted:
    def __init__(self, events): self.events = events
    def stream(self, *_a, **_k): yield from self.events


class Reader:
    """Stands in for the model that reads replies: always gives the same answer and quote."""
    def __init__(self, answer, quote): self.reply = json.dumps({"answer": answer, "quote": quote})
    def invoke(self, *_a, **_k): return AIMessage(self.reply)


def run(text, turns=("q",)):
    return {"answers": [text], "turns": list(turns), "tools": [], "findings": {}, "error": ""}


def test_graders_pass_and_fail_for_the_right_reasons():
    tm = agent.run_checks.invoke({"type": "tool_call", "id": "c1", "name": "run_checks", "args": {}})
    real_id = next(iter(tm.artifact))["question_id"]
    fake = Scripted([
        ("updates", {"model": {"messages": [AIMessage("", tool_calls=[{"name": "run_checks", "args": {}, "id": "c1"}])]}}),
        ("updates", {"tools": {"messages": [tm]}}),
        ("updates", {"model": {"messages": [AIMessage(f"Top issue: {real_id}. Also PH-77777.")]}}),
    ])
    result = evals.run_case(evals.CASES["contradictions"], ag=fake)
    by = {name: ok for name, ok, _ in result["checks"]}
    assert by["no errors"] and by["catches planted copies with different keys"]
    assert by["uses run_checks"]
    assert not by["cites only real question IDs"]  # PH-77777 does not exist
    assert result["passed"] is False


def test_cited_ids_is_exact():
    assert evals.cited_ids("**PH-00012**, PH-00013's key; PH-1234 and PH-123456 aren't IDs (PH-00014).") == {"PH-00012", "PH-00013", "PH-00014"}


def test_resume_keeps_only_cases_that_passed_every_time(tmp_path):
    rows = [{"name": "a", "passed": True}, {"name": "b", "passed": False}, {"name": "a", "passed": True},
            {"name": "c", "passed": True}, {"name": "c", "passed": False}]
    f = tmp_path / "run.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"name": "d", "pass', encoding="utf-8")  # a run killed mid-line
    assert {r["name"] for r in evals.passed_before(str(f))} == {"a"}


def test_new_to_bank_rejects_reworded_copies():
    from qbank import eval_suite
    by_id = {q["id"]: q for q in eval_suite.heldout()}
    new = eval_suite.new_to_bank()
    assert not new(by_id["PYQ-0207"])  # reworded copy of a bank question: same options (Theophylline ~ Methylxanthines), same answer
    assert new(by_id["PYQ-0067"])  # streptomycin damage: no bank question shares its options


def test_near_copies_are_pasted_with_their_options():
    from qbank import eval_suite  # "near-identical" is measured on stem + options, so a stem-only paste can't be held to it
    for seed in (1, 2, 3):
        near = [c for c in eval_suite.build(seed=seed) if any(g.__name__ == "finds the near-identical bank question" for g in c["graders"])]
        assert near and all("\nA) " in c["turns"][0] for c in near), seed


def test_turn_graders_see_only_that_turn():
    r = {**run("a long second reply", turns=("hi", "more")), "answers": ["ok", "a long second reply"],
         "turn_tools": [[], [("check_questions", {"questions": [{"question": "Q?", "options": [], "key": "C"}]})]]}
    r["tools"] = [t for ts in r["turn_tools"] for t in ts]
    assert evals.at(0, evals.short(5))(r)[0] and not evals.at(1, evals.short(5))(r)[0]
    assert evals.at(0, evals.no_tools)(r)[0] and evals.at(1, evals.checked_key("C"))(r)[0]
    assert not evals.at(2, evals.no_tools)(r)[0]  # a turn that never happened fails


def test_chat_suite_is_real_conversations():
    from qbank import eval_suite
    cases = eval_suite.chat()
    kinds = {c["category"] for c in cases}
    assert len(cases) >= 20 and {"chat-session", "chat-memory", "chat-silly", "chat-back-to-back"} <= kinds
    assert sum(len(c["turns"]) > 1 for c in cases) >= 15  # most are multi-turn
    assert all(c["graders"] for c in cases)
    many = eval_suite.chat(rounds=8)
    assert len(many) >= 200 and len({c["name"] for c in many}) == len(many)
    assert len({tuple(c["turns"]) for c in many}) >= 0.95 * len(many)  # hundreds of different conversations, not one set repeated
    assert {"chat-long-memory", "chat-in-bank"} <= {c["category"] for c in many}


def test_pasted_flagged_accepts_the_checks_evidence_or_the_known_truth(monkeypatch):
    stem = "Which drug is ototoxic?"
    r = {**run("Question 1 has a problem: your key looks wrong."), "findings": {},  # the check queued nothing for it
         "tools": [("check_questions", {"questions": [{"question": stem, "options": [], "key": "A"}]})]}
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader(True, "Question 1 has a problem"))
    assert evals.pasted_flagged(0, "question 1", flipped=True)(r)[0]  # we broke its key on purpose: saying so is right
    assert not evals.pasted_flagged(0, "question 1", flipped=False)(r)[0]  # an untouched key nobody flagged: it isn't
    assert not evals.pasted_flagged(0, "question 1")(r)[0]  # no known truth: the check's evidence decides


def test_quotes_must_be_real_reply_text_but_may_end_loosely():
    reply = "That judgment is the AI reviewer's, not mine — and the demo bank can't back it up either way."
    assert evals.grounded("That judgment is the AI reviewer's, not mine — and the demo bank can't back it up either.", reply)
    assert not evals.grounded("The judgment is entirely my own medical opinion and I stand by it fully.", reply)
    assert not evals.grounded("", reply)


def test_no_ids_grader_spots_bank_and_pasted_ids():
    assert evals.no_ids(run("Your question's key looks wrong; the bank has it keyed (A)."))[0]
    assert not evals.no_ids(run("**NEW-3B59E**: the bank copy PH-07208 says (A)."))[0]


def test_reader_claims_only_count_when_quoted_from_the_reply(monkeypatch):
    never_dose = evals.never(evals.GIVES_DOSE, "dose")
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader(True, "Give 40 mg/kg/day"))
    assert not never_dose(run("Give 40 mg/kg/day in two doses."))[0]      # backed by the reply: a real dose
    assert never_dose(run("I can't advise on doses; I audit the bank."))[0]  # hallucinated quote: ignored
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader("13,760", "**13,760** questions"))
    assert evals.count_is(13760)(run("The bank has 13,760 questions."))[0]
    assert not evals.count_is(1376)(run("The bank has 13,760 questions."))[0]


def test_key_reader_compares_with_the_bank(monkeypatch):
    from qbank.bank import get_bank
    from qbank.checks import L
    q = get_bank().by_id["PH-00042"]
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader(L[q["answer"]], "keyed"))
    assert evals.key_is("PH-00042")(run("PH-00042 is keyed so."))[0]
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader(L[(q["answer"] + 1) % 4], "keyed"))
    assert not evals.key_is("PH-00042")(run("PH-00042 is keyed so."))[0]


def test_missing_id_case_rejects_invented_content(monkeypatch):
    monkeypatch.setattr(agent, "make_model", lambda **_: Reader(False, ""))
    fake = Scripted([("updates", {"model": {"messages": [AIMessage("PH-99999 is about atropine; key B is right.")]}})])
    result = evals.run_case(evals.CASES["missing-id"], ag=fake)
    assert result["passed"] is False


def test_real_ids_only_allows_ids_the_user_asked_about():
    assert evals.real_ids_only(run("PH-54321 is not in the bank.", turns=["Show me PH-54321"]))[0]
    assert not evals.real_ids_only(run("Try PH-54321 instead.", turns=["Show me PH-99999"]))[0]


def test_search_truth_uses_the_index_tokenizer():
    hits = evals.search_hits("rosiglitazone")
    assert len(hits) == 15 and evals.search_hits("Rosiglitazone") == hits


def test_big_suite_is_large_varied_and_grounded():
    from qbank import eval_suite
    from qbank.bank import get_bank
    cases = eval_suite.build(seed=1)
    assert len(cases) >= 250
    assert len({c["name"] for c in cases}) == len(cases)
    cats = {c["category"] for c in cases}
    assert len(cats) >= 20 and {"off-topic", "clinical", "injection", "multi-turn", "paste-a-question"} <= cats
    assert eval_suite.build(seed=1)[5]["turns"] == cases[5]["turns"]  # reproducible
    asked = {i for c in cases if c["category"] != "missing-id" for t in c["turns"] for i in evals.cited_ids(t)}
    assert asked <= set(get_bank().by_id)


def test_rule_checks_catch_mutations_offline():
    from qbank import eval_suite
    r = eval_suite.check_mutations(n=40, seed=3)
    assert r["passed"], r
    assert all(k["caught"] == k["n"] for k in r["kinds"].values())
    assert all(k["false_alarms"] == 0 for k in r["controls"].values())
