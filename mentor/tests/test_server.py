from fastapi.testclient import TestClient

from mentor import server, tools

client = TestClient(server.app)


def test_overview_has_the_cohort_and_examples_that_name_real_students():
    o = client.get("/api/overview").json()
    assert o["students"] == 300 and o["batches"] == ["Batch A", "Batch B", "Batch C"] and o["as_of"] == "2026-11-15"
    assert len(o["examples"]) >= 4
    assert tools.student_name("S042") in " ".join(o["examples"])  # examples point at real (planted) students


def test_students_roster_shows_the_raw_data_for_everyone():
    rows = client.get("/api/students").json()
    assert len(rows) == 300
    by = {r["id"]: r for r in rows}
    assert by["S017"]["last_active"] == "2026-10-23" and by["S017"]["active_30"] <= 7  # the dropout
    assert by["S275"]["tests_taken"] == 0 and by["S275"]["lectures_full"] > 100  # watches everything, never tests
    assert by["S105"]["integrity"] == "confirmed" and by["S131"]["integrity"] == "cleared" and by["S150"]["integrity"] == "open"
    assert by["S288"]["latest_gt"] is None and by["S199"]["latest_gt"]["test"] == "Grand Test 5"


def test_one_students_records():
    d = client.get("/api/students/S042").json()
    assert d["name"] == tools.student_name("S042") and d["mentor"].startswith("Dr.")
    pharm = next(s for s in d["subjects"] if s["subject"] == "Pharmacology")
    assert all(lec["pct"] is None for lec in pharm["lectures"]) and len(pharm["lectures"]) == 6  # never watched
    assert pharm["accuracy"] < pharm["batch_accuracy"]
    assert len(d["weeks"]) == 12 and d["tests"] and all("rank" in t for t in d["tests"])
    assert client.get("/api/students/S999").status_code == 404


def test_chat_rejects_bad_session_ids_and_empty_messages():
    assert client.post("/api/chat", json={"session_id": "../x", "message": "hi"}).status_code == 422
    assert client.post("/api/chat", json={"session_id": "abc", "message": "   "}).status_code == 422
