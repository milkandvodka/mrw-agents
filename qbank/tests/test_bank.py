from qbank.bank import get_bank


def test_search_ranks_keyword_hits():
    b = get_bank()
    hits = b.search("rosiglitazone")
    assert hits
    assert all("rosiglitazone" in (h["question"] + " " + " ".join(h["options"])).lower() for h in hits[:3])


def test_similar_excludes_self_and_scores_in_range():
    b = get_bank()
    first = b.search("rosiglitazone")[0]["id"]
    sim = b.similar(question_id=first)
    assert sim and sim[0][0]["id"] != first and 0 < sim[0][1] <= 1.0001
    assert b.similar(text="drug of choice for absence seizures")


def test_topics_and_bad_input():
    b = get_bank()
    assert b.topics("autonomic") and b.topics("cns")  # "cns" matches "C.N.S"
    assert b.search(topic="no-such-topic") == []
    assert b.similar(question_id="PH-99999") == [] and b.similar() == []
    t = b.search(topic="autonomic", limit=500)
    assert t and all("autonomic" in x["topic"].lower() for x in t)


def test_exact_topic_name_wins_over_partial_matches():
    b = get_bank()
    assert b.topics("Hematology") == [("Hematology", 195)]        # not also "Hematology and CVS"
    assert b.topics("hematology and cvs") == [("Hematology and CVS", 16)]
    assert {t for t, _ in b.topics("hemato")} == {"Hematology", "Hematology and CVS"}  # partial still works
    assert {t for t, _ in b.topics("cns")} == {"C.N.S", "CNS"}    # same name up to punctuation: both


def test_overview_and_findings_loaded():
    b = get_bank()
    o = b.overview()
    assert o["questions"] == 13760 and o["counts"]["key_conflict"] >= 26  # 24 real groups + 2 planted
    assert len(b.by_id) == 13760 and b.findings


def test_numeric_libraries_stay_single_threaded():
    # each BLAS thread reserves a large buffer: 16 threads cost ~1 GB of committed memory per process for nothing
    from qbank import bank  # noqa: F401  (sets the limit before numpy loads)
    from threadpoolctl import threadpool_info
    blas = [p for p in threadpool_info() if p["internal_api"] == "openblas"]
    assert blas and all(p["num_threads"] == 1 for p in blas)
