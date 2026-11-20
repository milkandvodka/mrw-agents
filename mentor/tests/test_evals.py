from mentor import evals as E
from mentor import tools as T


def run(*answers, turns=None, tools=()):
    turns = list(turns or [f"q{i}" for i in range(len(answers))])
    return {"turns": turns, "answers": list(answers), "tools": list(tools), "turn_tools": [list(tools)] * len(answers), "error": ""}


def test_names_and_no_ids_read_the_reply_exactly():
    name = T.student_name("S017")
    assert E.names("S017")(run(f"{name} hasn't opened the app in 23 days."))[0]
    assert not E.names("S017")(run("Someone hasn't opened the app."))[0]
    assert E.no_ids(run(f"{name} is fine."))[0]
    assert not E.no_ids(run(f"**{name}** (S017) is fine."))[0]
    assert E.no_ids(run("S017 is fine.", turns=["How is S017?"]))[0]  # the mentor typed it


def test_turn_graders_see_only_that_turn():
    r = run("Hi!", "A long answer about tests.", tools=[])
    r["turn_tools"] = [[], [("tests", {"student": "x"})]]
    assert E.at(0, E.no_tools)(r)[0] and not E.at(1, E.no_tools)(r)[0]
    assert not E.at(2, E.no_tools)(r)[0]  # a turn that never happened fails


def test_quotes_must_be_real_reply_text():
    reply = "Nikhil Singh was last active on 23 Oct, 23 days ago, and missed Grand Test 5 entirely as a result."
    assert E.grounded("last active on 23 Oct", reply)
    assert E.grounded("Nikhil Singh was last active on 23 Oct, 23 days ago, and missed Grand Test 5 entirely.", reply)
    assert not E.grounded("Nikhil Singh is doing great", reply) and not E.grounded("", reply)


def test_cases_are_grounded_varied_and_unique():
    fixed = E.fixed()
    assert len(fixed) >= 25 and all(c["graders"] for c in fixed)
    gen = E.generated(seed=7, rounds=10)
    names = [c["name"] for c in fixed + gen]
    assert len(names) == len(set(names)) and len(gen) == 90
    assert len({tuple(c["turns"]) for c in gen}) >= 0.9 * len(gen)


def test_chat_suite_is_real_conversations_covering_every_kind():
    cases = E.chat(seed=3, rounds=10)
    kinds = {c["category"] for c in cases}
    assert {"chat-session", "chat-memory", "chat-long-memory", "chat-correction", "chat-multi", "chat-interrupted",
            "chat-silly", "chat-back-to-back", "chat-twins"} <= kinds
    assert len(cases) >= 110 and len({c["name"] for c in cases}) == len(cases)
    assert sum(len(c["turns"]) > 1 for c in cases) >= 0.6 * len(cases)
    # batches (3) and the twins (2) have few combinations, so those repeat across rounds: a consistency check
    assert len({tuple(c["turns"]) for c in cases}) >= 0.8 * len(cases)


def test_about_grader_resolves_names_and_ids():
    r = run("x")
    r["tools"] = [("student_report", {"student": T.student_name("S042")})]
    assert E.about("S042")(r)[0] and not E.about("S017")(r)[0]
    r["tools"] = [("tests", {"student": "S017"})]
    assert E.about("S017")(r)[0]


def test_no_gendered_checks_every_turn_by_whole_word():
    assert E.no_gendered(run("Riya's scores fell.", "Their rank is 4."))[0]
    assert not E.no_gendered(run("Fine.", "Her scores fell."))[0]  # any turn counts
    assert E.no_gendered(run("The theme here: hesitant shell scores."))[0]  # whole words only
