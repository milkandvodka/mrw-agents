import os

import pytest

pd = pytest.importorskip("pandas")  # data-prep extra (requirements-dev.txt); the app itself doesn't need it

from qbank.checks import run_all_checks  # noqa: E402
from qbank.prepare_data import AI_TOPIC, plant_errors, to_items  # noqa: E402

PARQUET = os.getenv("MEDMCQA_PARQUET")


@pytest.fixture(scope="module")
def items():
    if not PARQUET:
        pytest.skip("set MEDMCQA_PARQUET to the MedMCQA train parquet")
    return to_items(pd.read_parquet(PARQUET))


def test_items_shape(items):
    assert len(items) == 13758
    assert all(len(x["options"]) == 4 and x["answer"] in range(4) and x["topic"] for x in items)
    assert max(len(x["explanation"]) for x in items) <= 700


def test_plants_split_between_exact_checks_and_ai_review(items):
    bank, planted = plant_errors(items)
    assert len(planted) == 10 and len(bank) == len(items) + 2
    assert len({b["id"] for b in bank}) == len(bank) and bank[0]["id"] == "PH-00001"
    found = {(f["check"], i) for f in run_all_checks(bank)[0] for i in [f["question_id"], *f["related_ids"]]}
    flagged = {i for _, i in found}
    by_id = {b["id"]: b for b in bank}
    types = {}
    for p in planted:
        types[p["type"]] = types.get(p["type"], 0) + 1
        if p["type"] in ("explanation_conflict", "ai_only_wrong_key"):
            assert not flagged & set(p["ids"]) and p["topic"] == AI_TOPIC  # only the AI review can see these
        else:
            assert any((p["type"], i) in found for i in p["ids"]), p
        if p["type"] == "explanation_conflict":
            q = by_id[p["ids"][0]]
            assert q["explanation"] and q["answer"] != p["original"]
    assert types == {"key_conflict": 2, "duplicate_options": 2, "blank_option": 1, "explanation_conflict": 2, "ai_only_wrong_key": 3}


def test_planting_is_deterministic_and_leaves_input_alone(items):
    before = items[0]["options"][:]
    a, pa = plant_errors(items)
    b, pb = plant_errors(items)
    assert pa == pb and a == b and items[0]["options"] == before
