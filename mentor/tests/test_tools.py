"""Every tool answers from the database, and the planted stories are visible in what the tools return."""
import pytest

from mentor import tools as T


def call(t, **args):
    return t.invoke(args)


def test_student_lookup_by_name_id_typo_and_twins():
    assert "Batch A" in call(T.student_report, student="S017")
    name = T.student_name("S017")
    assert T.student_name(name.lower()) == name
    typo = call(T.student_report, student=name + name[-1])  # not resolved in code: candidates, closest first, for the model
    assert typo.split("closest first: ", 1)[1].startswith(name)
    twins = call(T.student_report, student="Priya Sharma")
    assert "Batch B" in twins and "Batch C" in twins and "which" in twins.lower()  # ask, don't guess
    assert "no student" in call(T.student_report, student="Zzyzx Qwerty").lower()


def test_report_shows_the_dropout_and_the_new_joiner():
    r = call(T.student_report, student="S017")
    assert "last active 23 Oct 2026 (23 days ago)" in r and "Grand Test 5" in r  # missed it
    new = call(T.student_report, student="S288")
    assert "joined 11 Nov 2026 (4 days ago)" in new and "no tests" in new.lower()


def test_activity_weeks_and_gaps():
    r = call(T.activity, student="S017", weeks=6)
    assert "0 min" in r.split("\n")[1] and "longest gap" in r.lower()
    fading = call(T.activity, student="S222", weeks=12)
    assert "batch median" in fading.lower()


def test_lectures_show_skips_partial_watching_and_order():
    assert "Pharmacology: 0 of 6" in call(T.lectures, student="S042")
    assert "average 25%" in call(T.lectures, student="S063")
    rev = call(T.lectures, student="S088", subject="Anatomy")
    assert rev.index("Upper Limb") < rev.index("Neuroanatomy") and "out of course order" in rev.lower()
    assert "never" in call(T.lectures, student="S042", subject="pharmacology").lower()


def test_tests_ranks_and_one_test_detail():
    r = call(T.tests, student="S199")  # the improver: bottom third in the first grand test, top tenth in the latest
    gts = [t[0] for t in T._tests() if t[2] == "Grand test"]
    (r1, n1), (r5, n5) = T._test_ranks(gts[0])["S199"][:2], T._test_ranks(gts[-1])["S199"][:2]
    assert r1 > 2 * n1 / 3 and r5 <= n5 / 10
    assert f"rank {r1} of {n1} in Batch A" in r and f"rank {r5} of {n5} in Batch A" in r
    assert "never taken a test" in call(T.tests, student="S275").lower()
    took = T.q("SELECT t.name FROM attempts a JOIN tests t ON t.id = a.test_id WHERE a.student_id = 'S042' AND t.kind = 'Grand test'")[0][0]
    one = call(T.tests, student="S042", test=took)
    assert "By subject" in one and "% of the batch got it right" in one
    missed = [t[1] for t in T._held_since(T._students()["S042"][2]) if t[1] != took and not T.q(
        "SELECT 1 FROM attempts WHERE student_id = 'S042' AND test_id = ?", t[0])]
    if missed:
        assert "did not take" in call(T.tests, student="S042", test=missed[0])


def test_performance_finds_the_weak_subject():
    r = call(T.performance, student="S011")
    assert r.index("Microbiology") < r.index("Anatomy")  # weakest first
    p = call(T.performance, student="S042", subject="Pharmacology")
    assert "not watched" in p.lower()


def test_integrity_separates_confirmed_cleared_and_open():
    assert "confirmed" in call(T.integrity, student="S105")
    cleared = call(T.integrity, student="S177")
    assert "cleared" in cleared and "confirmed" not in cleared
    assert "open" in call(T.integrity, student="S150")
    alike = call(T.integrity, test="Grand Test 3").split("\n\n")[3].split("\n")[1]  # the most alike pair
    assert T.student_name("S105") in alike and T.student_name("S131") in alike  # the copiers, not two weak students
    gt5 = call(T.integrity, test="Grand Test 5")
    assert T.student_name("S150") in gt5.split("\n\n")[1]  # fastest finish


def test_rank_students():
    gone = call(T.rank_students, metric="days_since_active", order="top", limit=10)
    assert T.student_name("S017") in gone  # a few low-engagement students have longer natural gaps; that's realistic
    climb = call(T.rank_students, metric="grand_test_rank_change", batch="Batch A", order="top", limit=3)
    assert T.student_name("S199") in climb
    with pytest.raises(Exception):
        call(T.rank_students, metric="bogus")


def test_sql_is_read_only_at_the_engine():
    assert "300" in call(T.run_sql, query="SELECT count(*) FROM students")
    for bad in ["DELETE FROM students", "UPDATE students SET name='x'", "ATTACH DATABASE 'x.db' AS x",
                "CREATE TABLE x (a)", "PRAGMA query_only=0"]:
        assert "not allowed" in call(T.run_sql, query=bad).lower(), bad
    assert call(T.run_sql, query="SELECT count(*) FROM students").count("\n") >= 1


def test_student_tools_say_pronouns_are_not_recorded():
    for t in (T.student_report, T.activity, T.lectures, T.tests, T.performance, T.integrity):
        out = call(t, student="S017")  # the model guesses he/she from names otherwise
        assert f"{T.student_name('S017').split()[0]}'s scores or their scores" in out, t.name
    assert "watched_pct >= 90" in T.run_sql.description  # "watched fully" means the same in SQL as in the other tools


def test_overview_names_the_snapshot_date_and_batches():
    r = call(T.overview)
    assert "15 Nov 2026" in r and "Batch A (107 students" in r and "Grand Test 5" in r
