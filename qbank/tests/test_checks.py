import inspect

from qbank import checks
from qbank.checks import norm, run_all_checks


def q(i, stem, opts, ans, exp="", topic="T"):
    return dict(id=i, question=stem, options=opts, answer=ans, explanation=exp, topic=topic)


def test_norm_ignores_case_punctuation_and_spacing():
    assert norm("  Atropine. ") == norm("ATROPINE") == "atropine"
    assert norm("Folic acid & B12") == "folic acid b12" and norm(None) == ""


def test_exact_checks_find_each_problem():
    bank = [
        q("A", "Drug of choice for X is", ["p", "q", "r", "s"], 0),
        q("B", "Drug of choice for X is", ["q", "p", "s", "r"], 0),   # same question re-entered, keyed 'q' instead of 'p'
        q("C", "Drug of choice for X is", ["p", "q", "r", "s"], 0),   # a third copy that agrees with A
        q("E", "Stem long enough here", ["a", "b", "c", "A."], 1),   # same option twice, written differently
        q("F", "Stem long enough here 2", ["a", " ", "c", "d"], 0),  # an empty option
        q("G", "Which is NOT used in Y", ["m", "n", "o", "p"], 0, exp="Ans. is 'c' i.e., o"),  # judgment: left to the AI
    ]
    f, counts = run_all_checks(bank)
    by = {(x["check"], x["question_id"]) for x in f}
    conflict = [x for x in f if x["check"] == "key_conflict"]
    assert len(conflict) == 1 and {conflict[0]["question_id"], *conflict[0]["related_ids"]} == {"A", "B", "C"}
    assert conflict[0]["detail"] == "Saved 3 times with different keys: (A) p in 2 copies; (A) q in 1."  # plain words, no ids
    assert by - {("key_conflict", "A")} == {("duplicate_options", "E"), ("blank_option", "F")}
    assert counts == {"key_conflict": 1, "duplicate_options": 1, "blank_option": 1, "exact_duplicate_groups": 0}
    assert all(x["key"] == f'{x["check"]}:{x["question_id"]}' and x["source"] == "exact" for x in f)


def test_copies_that_agree_are_counted_not_flagged():
    bank = [q("A", "Stem", ["p", "q", "r", "s"], 0), q("B", "stem.", ["s", "r", "q", "P"], 3)]
    f, counts = run_all_checks(bank)
    assert f == [] and counts["exact_duplicate_groups"] == 1


def test_checks_compare_text_and_never_pattern_match():
    assert "import re" not in inspect.getsource(checks)
