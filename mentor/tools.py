"""The mentor's tools. Each reads the database (read-only, enforced by SQLite) and returns plain facts with the batch
alongside for comparison. Code only fetches, counts and ranks; the model decides what the facts mean."""
import math
import sqlite3
import time
from collections import Counter
from contextlib import closing
from itertools import combinations
from datetime import date, timedelta
from difflib import get_close_matches
from functools import cache, wraps
from statistics import median
from typing import Literal

from langchain_core.tools import tool

from .generate import AS_OF, DB

L = "ABCD"
FULL = 90  # a lecture counts as watched fully at 90% or more
PRONOUNS = "(Pronouns aren't recorded: write {who}'s scores or their scores, never his or her; {who} or they, never he or she.)"
READ_ONLY = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorize(action, *_):
    return sqlite3.SQLITE_OK if action in READ_ONLY else sqlite3.SQLITE_DENY


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    con.set_authorizer(_authorize)  # the engine refuses anything but reading: no writes, ATTACH or PRAGMA
    return con


def q(sql: str, *args) -> list[tuple]:
    with closing(_db()) as con:
        return con.execute(sql, args).fetchall()


def _date(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(x)


def _day(x) -> str:
    d = _date(x)
    return f"{d.day} {d:%b %Y}"


def _ago(x) -> str:
    n = (AS_OF - _date(x)).days
    return "today" if n == 0 else "yesterday" if n == 1 else f"{n} days ago"


def _hm(minutes: float) -> str:
    h, m = divmod(round(minutes), 60)
    return f"{h}h {m}m" if h else f"{m} min"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{round(100 * x)}%"


# ---------- names ----------

@cache
def _students() -> dict[str, tuple[str, str, str]]:
    return {i: (n, b, j) for i, n, b, j in q("SELECT id, name, batch, joined_on FROM students")}


def _who(sid: str) -> str:
    n, b, j = _students()[sid]
    return f"{n} ({b}, joined {_day(j)}, ID {sid})"


def _resolve(student: str) -> tuple[str | None, str]:
    """(id, "") when exactly one student matches; otherwise (None, what to tell the mentor)."""
    s = " ".join(student.split())
    st = _students()
    if s.upper() in st:
        return s.upper(), ""
    low = s.lower()
    hits = [i for i, (n, *_r) in st.items() if n.lower() == low] or [i for i, (n, *_r) in st.items() if low and low in n.lower()]
    if len(hits) == 1:
        return hits[0], ""
    if hits:
        return None, (f"{len(hits)} students match '{s}': " + "; ".join(_who(i) for i in hits[:12])
                      + ". Ask the mentor which one they mean; don't pick one yourself.")
    near = get_close_matches(low, [n.lower() for n, *_r in st.values()], n=5, cutoff=0.6)  # closest first
    if not near:
        return None, f"No student matches '{s}'."
    by_name = {n.lower(): i for i, (n, *_r) in st.items()}
    return None, (f"No student is named '{s}'. Closest names, closest first: " + "; ".join(_who(by_name[n]) for n in near)
                  + ". If one differs only by a likely typo (a letter or two), use them and tell the mentor which name you used. A "
                    "different first or last name is a different student: say there's no student by that name and offer "
                    "the closest names; "
                    "otherwise ask.")


def student_name(student: str) -> str:
    sid, msg = _resolve(student)
    if not sid:
        raise LookupError(msg)
    return _students()[sid][0]


@cache
def _subjects() -> list[str]:
    return [s for s, in q("SELECT subject FROM lectures GROUP BY subject ORDER BY min(course_order)")]


def _subject(name: str) -> tuple[str | None, str]:
    by = {s.lower(): s for s in _subjects()}
    low = name.strip().lower()
    hit = low if low in by else next(iter(get_close_matches(low, by, n=1, cutoff=0.7)), None)
    if hit:
        return by[hit], ""
    return None, f"No subject '{name}'. Subjects: {', '.join(_subjects())}."


@cache
def _tests() -> list[tuple]:  # (id, name, kind, subject, held_on, questions, minutes), by date
    return q("SELECT id, name, kind, subject, held_on, questions, minutes FROM tests ORDER BY held_on")


def _test(name: str) -> tuple[tuple | None, str]:
    by = {t[1].lower(): t for t in _tests()} | {t[0].lower(): t for t in _tests()}
    hit = by.get(name.strip().lower()) or next(iter(get_close_matches(name.strip().lower(), by, n=1, cutoff=0.75)), None)
    if isinstance(hit, str):
        hit = by[hit]
    return (hit, "") if hit else (None, f"No test '{name}'. Tests: {', '.join(t[1] for t in _tests())}.")


def _held_since(joined: str) -> list[tuple]:
    return [t for t in _tests() if t[4] >= joined]


@cache
def _test_ranks(test_id: str) -> dict[str, tuple[int, int, int, int]]:
    """student -> (rank in batch, batch takers, rank overall, all takers). Ties share a rank."""
    rows = q("""SELECT a.student_id,
                       rank() OVER (PARTITION BY s.batch ORDER BY a.score DESC), count(*) OVER (PARTITION BY s.batch),
                       rank() OVER (ORDER BY a.score DESC), count(*) OVER ()
                FROM attempts a JOIN students s ON s.id = a.student_id WHERE a.test_id = ?""", test_id)
    return {r[0]: r[1:] for r in rows}


@cache
def _median_minutes(test_id: str) -> float:
    return median(m for m, in q("SELECT minutes_taken FROM attempts WHERE test_id = ?", test_id))


MIN_WRONG = 10  # fewer wrong answers than this are too few to compare answer sheets


@cache
def _overlaps(test_id: str) -> tuple[dict, list, float]:
    """How alike two takers' wrong answers are: the same wrong option on the same question, as a share of their wrong
    answers (shared / sqrt(wrong_a * wrong_b)), for takers with at least MIN_WRONG wrong answers.
    Returns (student -> (closest other, shared, own wrong, share)), the 5 most alike pairs, and the typical taker's
    closest share."""
    groups: dict[tuple, list[str]] = {}
    for s, qid, c in q("""SELECT a.student_id, a.question_id, a.chosen FROM answers a JOIN questions q ON q.id = a.question_id
                           WHERE q.test_id = ? AND a.correct = 0""", test_id):
        groups.setdefault((qid, c), []).append(s)
    wrong = {s: w for s, w in q("SELECT student_id, wrong FROM attempts WHERE test_id = ?", test_id) if w >= MIN_WRONG}
    shared = Counter(pair for g in groups.values() for pair in combinations(sorted(x for x in g if x in wrong), 2))
    scored = sorted(((n / math.sqrt(wrong[a] * wrong[b]), a, b, n) for (a, b), n in shared.items()), reverse=True)
    closest: dict[str, tuple] = {}
    for r, a, b, n in scored:  # best first, so the first pair seen for a student is their closest
        closest.setdefault(a, (b, n, wrong[a], r))
        closest.setdefault(b, (a, n, wrong[b], r))
    return closest, scored[:5], median([c[3] for c in closest.values()] or [0])


def _they(fn):
    """Per-student tools end with the pronoun note: from a name alone the model would guess he or she."""
    @wraps(fn)
    def w(*a, **k):
        sid, _msg = _resolve(str(k.get("student") or ""))
        return fn(*a, **k) + "\n" + PRONOUNS.format(who=_students()[sid][0].split()[0] if sid else "the student")
    return w


# ---------- tools ----------

@tool
def overview() -> str:
    """The snapshot date, the batches (size, mentor, start), the course (subjects in teaching order) and every test held
    (date, questions, takers). Call it to get exact batch, subject and test names."""
    lines = [f"Data as of {_day(AS_OF)}: treat this as today."]
    rows = q("""SELECT b.name, b.mentor, b.starts_on, count(s.id) FROM batches b LEFT JOIN students s ON s.batch = b.name
                GROUP BY b.name ORDER BY b.name""")
    lines.append("Batches: " + "; ".join(f"{n} ({c} students, mentor {m}, started {_day(st)})" for n, m, st, c in rows))
    n_lec = q("SELECT count(*) FROM lectures")[0][0]
    lines.append(f"Course: {len(_subjects())} subjects, {n_lec} lectures (6 per subject), taught in this order: {', '.join(_subjects())}.")
    takers = dict(q("SELECT test_id, count(*) FROM attempts GROUP BY test_id"))
    lines.append("Tests held (date, questions, takers): " + "; ".join(
        f"{t[1]} ({_day(t[4])}, {t[5]} questions, {takers.get(t[0], 0)} takers)" for t in _tests()))
    lines.append("Scoring: +4 for a right answer, -1 for a wrong one, 0 for a skipped one.")
    return "\n".join(lines)


@tool
@_they
def find_students(name: str = "", batch: str = "") -> str:
    """Students whose name contains the given text (or close to it, for typos), optionally in one batch. With no name,
    lists the batch. Use it when the mentor gives a partial name or wants a list."""
    st = _students()
    low = " ".join(name.split()).lower()
    hits = [i for i, (n, b, _j) in st.items() if low in n.lower() and (not batch or b.lower() == batch.strip().lower())]
    if not hits and low:
        close = set(get_close_matches(low, [n.lower() for n, *_r in st.values()], n=8, cutoff=0.7))
        hits = [i for i, (n, *_r) in st.items() if n.lower() in close]
    if not hits:
        return f"No student matches '{name}'" + (f" in {batch}" if batch else "") + "."
    last = dict(q("SELECT student_id, max(day) FROM activity GROUP BY student_id"))
    out = [f"{len(hits)} student{'s' if len(hits) != 1 else ''}" + (" (showing 40)" if len(hits) > 40 else "") + ":"]
    out += [f"- {_who(i)}, last active {_day(last[i]) if i in last else 'never'}" for i in hits[:40]]
    return "\n".join(out)


@tool
@_they
def student_report(student: str) -> str:
    """One student at a glance: batch, join date, recent app use, lectures, tests, latest grand test rank, accuracy by
    subject (strongest/weakest), and test-integrity reviews, each next to the batch. student: a name or an ID."""
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, joined = _students()[sid]
    mentor = q("SELECT mentor FROM batches WHERE name = ?", batch)[0][0]
    out = [f"{name}, {batch} (mentor {mentor}, ID {sid}). They joined {_day(joined)} ({_ago(joined)})."]

    since = (AS_OF - timedelta(days=30)).isoformat()
    week = (AS_OF - timedelta(days=7)).isoformat()
    batch30 = q("""SELECT s.id, count(a.day), coalesce(sum(a.minutes), 0) FROM students s
                   LEFT JOIN activity a ON a.student_id = s.id AND a.day > ? WHERE s.batch = ? GROUP BY s.id""", since, batch)
    mine = {s: (d, m) for s, d, m in batch30}[sid]
    per_day = [m / d for _s, d, m in batch30 if d]
    last = q("SELECT max(day), sum(minutes) FROM activity WHERE student_id = ?", sid)[0]
    d7 = q("SELECT count(*) FROM activity WHERE student_id = ? AND day > ?", sid, week)[0][0]
    out.append(f"App use: they were last active {_day(last[0])} ({_ago(last[0])}). " if last[0] else "App use: they never opened the app. ")
    out[-1] += (f"Active {d7} of the last 7 days and {mine[0]} of the last 30 (batch median {median(d for _s, d, _m in batch30):g}). "
                f"{_hm(mine[1] / mine[0]) if mine[0] else '0 min'} per active day in the last 30 days "
                f"(batch median {_hm(median(per_day)) if per_day else '0 min'}). {_hm(last[1] or 0)} in total since joining.")

    w = q("SELECT count(*), sum(watched_pct >= ?), avg(watched_pct) FROM watch WHERE student_id = ?", FULL, sid)[0]
    b_avg = q("SELECT avg(w.watched_pct) FROM watch w JOIN students s ON s.id = w.student_id WHERE s.batch = ?", batch)[0][0]
    n_lec = q("SELECT count(*) FROM lectures")[0][0]
    out.append(f"Lectures: they started {w[0]} of {n_lec}, {w[1] or 0} watched fully" + (
        f", on average {round(w[2])}% of each started lecture watched (batch {round(b_avg)}%)." if w[0] else "."))

    held = _held_since(joined)
    taken = {t for t, in q("SELECT test_id FROM attempts WHERE student_id = ?", sid)}
    if not held:
        out.append("Tests: no tests held since they joined.")
    elif not taken:
        out.append(f"Tests: has taken no tests ({len(held)} held since joining).")
    else:
        missed = [t[1] for t in reversed(held) if t[0] not in taken]
        out.append(f"Tests: they took {len(taken)} of {len(held)} held since joining." + (
            f" Missed (latest first): {', '.join(missed[:6])}{' and more' if len(missed) > 6 else ''}." if missed else ""))
        gts = [t for t in held if t[2] == "Grand test" and t[0] in taken]
        if gts:
            t = gts[-1]
            r = _test_ranks(t[0])[sid]
            sc = q("SELECT score FROM attempts WHERE student_id = ? AND test_id = ?", sid, t[0])[0][0]
            out.append(f"Latest grand test taken: {t[1]} ({_day(t[4])}), score {sc}/{4 * t[5]}, rank {r[0]} of {r[1]} in {batch}, "
                       f"{r[2]} of {r[3]} overall.")
        acc = q("""SELECT q.subject, avg(a.correct), count(a.correct) FROM answers a JOIN questions q ON q.id = a.question_id
                   WHERE a.student_id = ? AND a.correct IS NOT NULL GROUP BY q.subject HAVING count(a.correct) >= 10
                   ORDER BY avg(a.correct)""", sid)
        overall = q("SELECT avg(correct) FROM answers WHERE student_id = ? AND correct IS NOT NULL", sid)[0][0]
        b_overall = q("""SELECT avg(a.correct) FROM answers a JOIN students s ON s.id = a.student_id
                         WHERE s.batch = ? AND a.correct IS NOT NULL""", batch)[0][0]
        out.append(f"Accuracy: {_pct(overall)} of attempted questions right (batch {_pct(b_overall)})."
                   + (f" Weakest subject: {acc[0][0]} ({_pct(acc[0][1])}); strongest: {acc[-1][0]} ({_pct(acc[-1][1])})." if len(acc) > 1 else ""))

    reviews = q("SELECT outcome, count(*) FROM integrity_reviews WHERE student_id = ? GROUP BY outcome", sid)
    out.append("Test integrity: " + ("; ".join(f"{n} {o} review{'s' if n > 1 else ''}" for o, n in reviews)
                                     + " (details: integrity tool)." if reviews else "no reviews."))
    return "\n".join(out)


@tool
@_they
def activity(student: str, weeks: int = 8) -> str:
    """Week-by-week app use for the last few weeks (days active, time spent) next to the batch median, plus the current
    streak, the longest gap since joining and the trend. student: a name or an ID."""
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, joined = _students()[sid]
    weeks = max(1, min(weeks, 52))
    days = dict(q("SELECT day, minutes FROM activity WHERE student_id = ?", sid))
    bdays = q("""SELECT a.student_id, a.day, a.minutes FROM activity a JOIN students s ON s.id = a.student_id
                 WHERE s.batch = ? AND a.day > ?""", batch, (AS_OF - timedelta(days=7 * weeks)).isoformat())
    members = [i for i, (_n, b, _j) in _students().items() if b == batch]
    out = [f"{name} ({batch}): their app use by week, most recent first:"]
    totals = []
    for k in range(weeks):
        end = AS_OF - timedelta(days=7 * k)
        start = end - timedelta(days=6)
        if end < _date(joined):
            break
        span = [(start + timedelta(days=i)).isoformat() for i in range(7)]
        mins = sum(days.get(d, 0) for d in span)
        act = sum(d in days for d in span)
        per = {m: 0 for m in members}
        for s, d, m in bdays:
            if start.isoformat() <= d <= end.isoformat():
                per[s] += m
        totals.append(mins)
        out.append(f"- week {_day(start)} to {_day(end)}: {act} day{"s" if act != 1 else ""} active, {_hm(mins)} (batch median {_hm(median(per.values()))})")
    active = sorted(_date(d) for d in days)
    if not active:
        return out[0] + " never opened the app."
    streak = 0
    while (AS_OF - timedelta(days=streak)) in set(active):
        streak += 1
    gaps = [((b - a).days - 1, a, b) for a, b in zip([_date(joined) - timedelta(days=1)] + active, active + [AS_OF + timedelta(days=1)])]
    g, a, b = max(gaps)
    out.append(f"Last active {_day(active[-1])} ({_ago(active[-1])}); current streak {streak} day{'s' if streak != 1 else ''}. "
               + (f"Longest gap since joining: {g} days ({_day(a + timedelta(days=1))} to {_day(b - timedelta(days=1))}"
                  + (", still going" if b > AS_OF else "") + ")." if g > 0 else "No gaps since joining."))
    if len(totals) >= 8:
        recent, before = sum(totals[:4]), sum(totals[4:8])
        out.append(f"Last 4 weeks: {_hm(recent)} vs {_hm(before)} in the 4 weeks before"
                   + (f" ({round(100 * (recent - before) / before):+d}%)." if before else "."))
    return "\n".join(out)


@tool
@_they
def lectures(student: str, subject: str = "") -> str:
    """Lecture watching. Without a subject: per subject, how many lectures were started and watched fully, average share
    watched, lectures skipped while later ones were watched, subjects watched out of course order, and the latest
    lectures watched. With a subject: each lecture in course order with when it was first watched, how much, and in
    what order. student: a name or an ID."""
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, joined = _students()[sid]
    rows = q("""SELECT l.id, l.subject, l.title, l.course_order, l.minutes, w.first_watched, w.last_watched, w.watched_pct
                FROM lectures l LEFT JOIN watch w ON w.lecture_id = l.id AND w.student_id = ? ORDER BY l.course_order""", sid)
    by_subj: dict[str, list] = {}
    for r in rows:
        by_subj.setdefault(r[1], []).append(r)

    def out_of_order(rs):
        seen = sorted((r[5], r[3]) for r in rs if r[5])
        return any(b < a for (_d1, a), (_d2, b) in zip(seen, seen[1:]))

    if subject:
        subj, msg = _subject(subject)
        if not subj:
            return msg
        rs = by_subj[subj]
        order = {r[0]: k for k, r in enumerate(sorted((r for r in rs if r[5]), key=lambda r: (r[5], r[3])), 1)}
        out = [f"{name} ({batch}), {subj} lectures in course order:"]
        for k, r in enumerate(rs, 1):
            if r[5]:
                out.append(f"{k}. {r[2]} ({r[4]} min): first watched {_day(r[5])} ({(_date(r[5]) - _date(joined)).days} days after "
                           f"joining), {r[7]}% watched, the no. {order[r[0]]} lecture they opened in this subject"
                           + (f", last opened {_day(r[6])}" if r[6] != r[5] else ""))
            else:
                out.append(f"{k}. {r[2]} ({r[4]} min): never watched")
        if order:
            out.append("Watched out of course order: they opened later lectures before earlier ones." if out_of_order(rs)
                       else "Watched in course order.")
        return "\n".join(out)

    started = [r for r in rows if r[5]]
    b_avg = q("SELECT avg(w.watched_pct) FROM watch w JOIN students s ON s.id = w.student_id WHERE s.batch = ?", batch)[0][0]
    out = [f"{name} ({batch}): they started {len(started)} of {len(rows)} lectures, {sum(r[7] >= FULL for r in started)} watched "
           f"fully ({FULL}% or more)" + (f", average {round(sum(r[7] for r in started) / len(started))}% of each started "
                                          f"lecture watched (batch average {round(b_avg)}%)." if started else ".")]
    for subj, rs in by_subj.items():
        st = [r for r in rs if r[5]]
        out.append(f"- {subj}: {len(st)} of {len(rs)} started" + (
            f", {sum(r[7] >= FULL for r in st)} fully, average {round(sum(r[7] for r in st) / len(st))}% watched" if st else ""))
    skipped = [f"{r[1]}: {r[2]}" for rs in by_subj.values() for k, r in enumerate(rs)
               if not r[5] and any(x[5] for x in rs[k + 1:])]
    if skipped:
        out.append(f"Skipped while watching later lectures of the same subject ({len(skipped)}): {'; '.join(skipped[:15])}"
                   + (" and more" if len(skipped) > 15 else "") + ".")
    disorder = [s for s, rs in by_subj.items() if out_of_order(rs)]
    out.append(f"Subjects watched out of course order: {', '.join(disorder)}." if disorder else "Every subject watched in course order.")
    recent = sorted(started, key=lambda r: r[6], reverse=True)[:3]
    if recent:
        out.append("Most recently opened: " + "; ".join(f"{r[2]} ({r[1]}) on {_day(r[6])}, {r[7]}% watched" for r in recent) + ".")
    return "\n".join(out)


@tool
@_they
def tests(student: str, test: str = "") -> str:
    """Test results. Without a test: every test taken (score, right/wrong/skipped, rank in batch and overall, time taken
    vs the median, tab switches) and the tests missed. With a test name: that attempt in detail, by subject, with each
    wrong answer and how much of the batch got that question right. student: a name or an ID."""
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, joined = _students()[sid]
    att = {r[0]: r[1:] for r in q("""SELECT test_id, minutes_taken, tab_switches, correct, wrong, skipped, score
                                       FROM attempts WHERE student_id = ?""", sid)}
    held = _held_since(joined)
    if test:
        t, msg = _test(test)
        if not t:
            return msg
        if t[0] not in att:
            return f"{name} did not take {t[1]} ({_day(t[4])})." + ("" if t[4] >= joined else " It was held before they joined.")
        return _attempt_detail(sid, t, att[t[0]])
    if not att:
        return f"{name} ({batch}) has never taken a test ({len(held)} held since joining)."
    out = [f"{name} ({batch}): they took {len(att)} of the {len(held)} tests held since they joined."]
    for t in held:
        if t[0] in att:
            mins, tabs, c, w, s, sc = att[t[0]]
            r = _test_ranks(t[0])[sid]
            out.append(f"- {t[1]} ({_day(t[4])}): score {sc}/{4 * t[5]} ({c} right, {w} wrong, {s} skipped), rank {r[0]} of {r[1]} "
                       f"in {batch}, {r[2]} of {r[3]} overall; {mins} of {t[6]} min (median {_median_minutes(t[0]):g})"
                       + (f"; {tabs} tab switches" if tabs else ""))
    missed = [t[1] for t in held if t[0] not in att]
    if missed:
        out.append(f"Missed: {', '.join(missed)}.")
    return "\n".join(out)


def _attempt_detail(sid: str, t: tuple, a: tuple) -> str:
    name, batch, _j = _students()[sid]
    mins, tabs, c, w, s, sc = a
    r = _test_ranks(t[0])[sid]
    out = [f"{name} in {t[1]} ({_day(t[4])}): their score {sc}/{4 * t[5]} ({c} right, {w} wrong, {s} skipped), rank {r[0]} of {r[1]} in "
           f"{batch}, {r[2]} of {r[3]} overall; {mins} of {t[6]} min (median {_median_minutes(t[0]):g}); {tabs} tab switches."]
    rows = q("""SELECT q.number, q.subject, q.topic, q.kind, q.difficulty, q.answer, a.chosen, a.correct,
                       (SELECT avg(b.correct IS 1) FROM answers b JOIN students s ON s.id = b.student_id
                        WHERE b.question_id = q.id AND s.batch = ?)
                FROM answers a JOIN questions q ON q.id = a.question_id WHERE a.student_id = ? AND q.test_id = ?
                ORDER BY q.number""", batch, sid, t[0])
    if t[2] == "Grand test":
        subj: dict[str, list[int]] = {}
        for row in rows:
            v = subj.setdefault(row[1], [0, 0])
            v[0] += row[7] == 1
            v[1] += 1
        out.append("By subject (right of asked): " + "; ".join(f"{k} {c_}/{n}" for k, (c_, n) in sorted(subj.items(), key=lambda x: x[1][0] / x[1][1])))
    wrong = [row for row in rows if row[7] == 0]
    out.append(f"Wrong answers ({len(wrong)}), with the share of the batch who got each right:")
    out += [f"- Q{row[0]} {row[1]}: {row[2]} ({row[3]}, {row[4]}): chose {L[row[6]]}, answer {L[row[5]]}; {_pct(row[8])} of the batch got it right"
            for row in wrong[:40]]
    skipped = [str(row[0]) for row in rows if row[7] is None]
    if skipped:
        out.append(f"Skipped: Q{', Q'.join(skipped)}.")
    return "\n".join(out)


@tool
@_they
def performance(student: str, subject: str = "") -> str:
    """Accuracy over every test taken, next to the batch: by subject (weakest first), by question kind (recall, clinical
    vignette, image-based) and by difficulty. With a subject: by topic within it, and whether they watched each topic's
    lecture. student: a name or an ID."""
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, _j = _students()[sid]

    def acc(col: str, where: str = "", *args):
        mine = q(f"""SELECT q.{col}, avg(a.correct), count(a.correct), avg(a.correct IS NULL) FROM answers a
                     JOIN questions q ON q.id = a.question_id WHERE a.student_id = ? {where} GROUP BY q.{col}""", sid, *args)
        theirs = dict(q(f"""SELECT q.{col}, avg(a.correct) FROM answers a JOIN questions q ON q.id = a.question_id
                            JOIN students s ON s.id = a.student_id WHERE s.batch = ? AND a.correct IS NOT NULL {where}
                            GROUP BY q.{col}""", batch, *args))
        return sorted(((k, c, n, sk, theirs.get(k)) for k, c, n, sk in mine if n), key=lambda x: x[1])

    if not q("SELECT 1 FROM attempts WHERE student_id = ? LIMIT 1", sid):
        return f"{name} ({batch}) has not taken any test, so there are no answers to analyse."
    head = f"{name} ({batch}): their accuracy, i.e. right answers out of attempted (skipped excluded), batch in brackets."
    if subject:
        subj, msg = _subject(subject)
        if not subj:
            return msg
        watched = {t: (f, p) for t, f, p in q("""SELECT l.title, w.first_watched, w.watched_pct FROM lectures l
                                                  JOIN watch w ON w.lecture_id = l.id WHERE w.student_id = ? AND l.subject = ?""", sid, subj)}
        rows = acc("topic", "AND q.subject = ?", subj)
        if not rows:
            return f"{head}\nNo {subj} questions answered yet."
        out = [head, f"{subj} by topic, weakest first:"]
        for k, c, n, sk, b in rows:
            lec = f"lecture watched {watched[k][1]} on {_day(watched[k][0])}" if k in watched else "lecture not watched"
            lec = lec.replace(f"watched {watched[k][1]} on", f"{watched[k][1]}% watched, first on") if k in watched else lec
            out.append(f"- {k}: {_pct(c)} of {n} ({_pct(b)}); {lec}")
        return "\n".join(out)
    out = [head, "By subject, weakest first:"]
    out += [f"- {k}: {_pct(c)} of {n} ({_pct(b)}), skipped {_pct(sk)}" for k, c, n, sk, b in acc("subject")]
    out.append("By question kind: " + "; ".join(f"{k} {_pct(c)} of {n} ({_pct(b)})" for k, c, n, _sk, b in acc("kind")))
    out.append("By difficulty: " + "; ".join(f"{k} {_pct(c)} of {n} ({_pct(b)})" for k, c, n, _sk, b in acc("difficulty")))
    return "\n".join(out)


@tool
@_they
def integrity(student: str = "", test: str = "") -> str:
    """Test-integrity evidence. For a student: their integrity reviews (signal, outcome: confirmed, cleared or open) and,
    for every test they took, time taken vs the median, tab switches, and the other student whose wrong answers match
    theirs most, next to what is typical. For a test: the fastest finishes, the most tab switches, the most similar
    answer sheets, and its reviews. Facts only: a match or a fast finish is not proof; only a confirmed review is."""
    if test and not student:
        t, msg = _test(test)
        if not t:
            return msg
        return _test_integrity(t)
    sid, msg = _resolve(student)
    if not sid:
        return msg
    name, batch, _j = _students()[sid]
    out = [f"{name} ({batch})."]
    revs = q("""SELECT t.name, r.opened_on, r.signal, r.outcome, r.decided_on, r.note FROM integrity_reviews r
                JOIN tests t ON t.id = r.test_id WHERE r.student_id = ? ORDER BY r.opened_on""", sid)
    out.append("Integrity reviews: " + ("none." if not revs else ""))
    out += [f"- {tn}: '{sig}' (opened {_day(op)}). Outcome: {oc}" + (f" on {_day(dec)}" if dec else ", awaiting a decision")
            + f". Note: {note}" for tn, op, sig, oc, dec, note in revs]
    att = q("SELECT test_id, minutes_taken, tab_switches, wrong FROM attempts WHERE student_id = ?", sid)
    chosen = [t for t in _tests() if t[0] in {a[0] for a in att} and (not test or _test(test)[0] == t)]
    if chosen:
        out.append("Per test (time taken vs median; tab switches; the most alike answer sheet: the same wrong option on the "
                   "same questions, as a share of their wrong answers, vs the typical taker's closest share):")
    a_by = {a[0]: a for a in att}
    for t in chosen:
        _t, mins, tabs, wrong = a_by[t[0]]
        closest, _pairs, typical = _overlaps(t[0])
        o = closest.get(sid)
        match = (f"{o[1]} of their {wrong} wrong answers match {_students()[o[0]][0]}'s ({_pct(o[3])}; typical {_pct(typical)})"
                 if o else f"too few wrong answers ({wrong}) to compare")
        out.append(f"- {t[1]}: {mins} of {t[6]} min (median {_median_minutes(t[0]):g}); {tabs} tab switches; {match}")
    return "\n".join(out)


def _test_integrity(t: tuple) -> str:
    rows = q("""SELECT a.student_id, a.minutes_taken, a.tab_switches, a.score FROM attempts a WHERE a.test_id = ?""", t[0])
    ranks = _test_ranks(t[0])
    nm = lambda s: f"{_students()[s][0]} ({_students()[s][1]})"
    out = [f"{t[1]} ({_day(t[4])}): {len(rows)} takers, {t[6]} minutes allowed, median time {_median_minutes(t[0]):g} min."]
    fast = sorted(rows, key=lambda r: r[1])[:5]
    out.append("Fastest finishes:\n" + "\n".join(f"- {nm(s)}: {m} min, score {sc}/{4 * t[5]} (rank {ranks[s][2]} of {ranks[s][3]})"
                                                for s, m, _tb, sc in fast))
    tabs = [r for r in sorted(rows, key=lambda r: -r[2]) if r[2] > 0][:5]
    out.append("Most tab switches:\n" + ("\n".join(f"- {nm(s)}: {tb}" for s, _m, tb, _sc in tabs) if tabs else "- none"))
    _closest, pairs, typical = _overlaps(t[0])
    out.append(f"Most alike answer sheets (same wrong option on the same questions, as a share of their wrong answers; the "
               f"typical taker's closest share is {_pct(typical)}):\n"
               + "\n".join(f"- {nm(a)} and {nm(b)}: {_pct(r)} ({n} shared)" for r, a, b, n in pairs))
    revs = q("""SELECT student_id, signal, outcome, decided_on FROM integrity_reviews WHERE test_id = ?""", t[0])
    out.append("Integrity reviews for this test:\n" + ("\n".join(f"- {nm(s)}: '{sig}', {oc}" + (f" on {_day(d)}" if d else "")
                                                             for s, sig, oc, d in revs) if revs else "- none"))
    return "\n\n".join(out)


METRICS = Literal["test_score", "accuracy", "grand_test_average", "active_days_30", "minutes_30", "days_since_active",
                  "lectures_completed", "tests_missed", "grand_test_rank_change"]


@tool
@_they
def rank_students(metric: METRICS, batch: str = "", subject: str = "", test: str = "",
                  order: Literal["top", "bottom"] = "top", limit: int = 10) -> str:
    """Rank students by one measure, highest first ("top") or lowest first ("bottom"), optionally within a batch.
    metric: test_score (needs test), accuracy (optionally in one subject), grand_test_average (average score % over grand
    tests taken), active_days_30 and minutes_30 (last 30 days), days_since_active, lectures_completed (watched 90%+),
    tests_missed (held since joining but not taken), grand_test_rank_change (batch percentile, latest grand test minus
    first; positive = climbing)."""
    st = {i: v for i, v in _students().items() if not batch or v[1].lower() == batch.strip().lower()}
    if not st:
        return f"No batch '{batch}'. Batches: {', '.join(sorted({v[1] for v in _students().values()}))}."
    since = (AS_OF - timedelta(days=30)).isoformat()
    label, fmt = metric.replace("_", " "), str
    if metric == "test_score":
        t, msg = _test(test) if test else (None, "test_score needs a test name.")
        if not t:
            return msg
        vals = dict(q("SELECT student_id, score FROM attempts WHERE test_id = ?", t[0]))
        label, fmt = f"score in {t[1]} (out of {4 * t[5]})", str
    elif metric == "accuracy":
        subj, msg = _subject(subject) if subject else ("", "")
        if subject and not subj:
            return msg
        vals = dict(q("""SELECT a.student_id, avg(a.correct) FROM answers a JOIN questions q ON q.id = a.question_id
                         WHERE a.correct IS NOT NULL AND (? = '' OR q.subject = ?) GROUP BY a.student_id
                         HAVING count(a.correct) >= 20""", subj, subj))
        label, fmt = f"accuracy{' in ' + subj if subj else ''} (at least 20 answered)", _pct
    elif metric == "grand_test_average":
        vals = dict(q("""SELECT a.student_id, avg(100.0 * a.score / (4 * t.questions)) FROM attempts a JOIN tests t ON t.id = a.test_id
                         WHERE t.kind = 'Grand test' GROUP BY a.student_id"""))
        label, fmt = "average grand test score", lambda v: f"{v:.0f}%"
    elif metric in ("active_days_30", "minutes_30"):
        rows = q("SELECT student_id, count(*), sum(minutes) FROM activity WHERE day > ? GROUP BY student_id", since)
        vals = {i: 0 for i in st} | {s: (d if metric == "active_days_30" else m) for s, d, m in rows}
        label, fmt = ("days active in the last 30", str) if metric == "active_days_30" else ("time in the app in the last 30 days", _hm)
    elif metric == "days_since_active":
        last = dict(q("SELECT student_id, max(day) FROM activity GROUP BY student_id"))
        vals = {i: (AS_OF - _date(last.get(i, v[2]))).days for i, v in st.items()}
        label, fmt = "days since last active", str
    elif metric == "lectures_completed":
        vals = {i: 0 for i in st} | dict(q("SELECT student_id, count(*) FROM watch WHERE watched_pct >= ? GROUP BY student_id", FULL))
        label = f"lectures watched fully ({FULL}%+)"
    elif metric == "tests_missed":
        taken: dict[str, set] = {}
        for s, t_ in q("SELECT student_id, test_id FROM attempts"):
            taken.setdefault(s, set()).add(t_)
        vals = {i: sum(t[0] not in taken.get(i, set()) for t in _held_since(v[2])) for i, v in st.items()}
        label = "tests missed since joining"
    else:  # grand_test_rank_change
        vals = {}
        for i in st:
            gts = [t for t in _tests() if t[2] == "Grand test" and i in _test_ranks(t[0])]
            if len(gts) >= 2:
                pct = [100 * (1 - (_test_ranks(t[0])[i][0] - 1) / _test_ranks(t[0])[i][1]) for t in (gts[0], gts[-1])]
                vals[i] = pct[1] - pct[0]
        label, fmt = "change in batch percentile, first to latest grand test", lambda v: f"{v:+.0f} points"
    vals = {i: v for i, v in vals.items() if i in st and v is not None}
    ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=(order == "top"))[:max(1, min(limit, 50))]
    out = [f"{'Highest' if order == 'top' else 'Lowest'} {label}" + (f", {batch}" if batch else ", all batches")
           + f" ({len(vals)} students with a value):"]
    out += [f"{k}. {_who(i)}: {fmt(v)}" for k, (i, v) in enumerate(ranked, 1)]
    return "\n".join(out)


# ---------- raw data for the UI's student browser (structured, not for the model) ----------

def _latest_gt(sid: str) -> dict | None:
    for t in reversed([t for t in _tests() if t[2] == "Grand test"]):
        if sid in _test_ranks(t[0]):
            r = _test_ranks(t[0])[sid]
            return {"test": t[1], "rank": r[0], "of": r[1]}
    return None


def roster() -> list[dict]:
    """One row per student: the numbers a mentor scans first."""
    since = (AS_OF - timedelta(days=30)).isoformat()
    last = dict(q("SELECT student_id, max(day) FROM activity GROUP BY student_id"))
    act30 = dict(q("SELECT student_id, count(*) FROM activity WHERE day > ? GROUP BY student_id", since))
    lec = {s: (n, f) for s, n, f in q("SELECT student_id, count(*), sum(watched_pct >= ?) FROM watch GROUP BY student_id", FULL)}
    taken = dict(q("SELECT student_id, count(*) FROM attempts GROUP BY student_id"))
    acc = dict(q("SELECT student_id, avg(correct) FROM answers WHERE correct IS NOT NULL GROUP BY student_id"))
    worst = {"confirmed": 3, "open": 2, "cleared": 1}
    integ: dict[str, str] = {}
    for s, o in q("SELECT student_id, outcome FROM integrity_reviews"):
        integ[s] = max(integ.get(s, o), o, key=worst.get)
    return [{"id": i, "name": n, "batch": b, "joined": j, "last_active": last.get(i), "active_30": act30.get(i, 0),
             "lectures_started": lec.get(i, (0, 0))[0], "lectures_full": lec.get(i, (0, 0))[1] or 0,
             "tests_taken": taken.get(i, 0), "tests_held": len(_held_since(j)), "latest_gt": _latest_gt(i),
             "accuracy": acc.get(i), "integrity": integ.get(i)} for i, (n, b, j) in _students().items()]


def student_detail(sid: str) -> dict | None:
    """Everything recorded about one student, for the UI."""
    if sid not in _students():
        return None
    name, batch, joined = _students()[sid]
    days = dict(q("SELECT day, minutes FROM activity WHERE student_id = ?", sid))
    weeks = []
    for k in range(12):
        end = AS_OF - timedelta(days=7 * k)
        span = [(end - timedelta(days=i)).isoformat() for i in range(7)]
        weeks.append({"start": span[-1], "end": span[0], "days": sum(d in days for d in span), "minutes": sum(days.get(d, 0) for d in span)})
    att = {r[0]: r for r in q("SELECT test_id, minutes_taken, tab_switches, correct, wrong, skipped, score FROM attempts WHERE student_id = ?", sid)}
    tests_, missed = [], []
    for t in _held_since(joined):
        if t[0] not in att:
            missed.append(t[1])
            continue
        _id, mins, tabs, c, w, s, sc = att[t[0]]
        r = _test_ranks(t[0])[sid]
        tests_.append({"test": t[1], "date": t[4], "kind": t[2], "score": sc, "max": 4 * t[5], "right": c, "wrong": w,
                       "skipped": s, "rank": r[0], "of": r[1], "minutes": mins, "limit": t[6], "tabs": tabs})
    mine = {s: (a, n) for s, a, n in q("""SELECT q.subject, avg(a.correct), count(a.correct) FROM answers a
                                           JOIN questions q ON q.id = a.question_id WHERE a.student_id = ? GROUP BY q.subject""", sid)}
    theirs = dict(q("""SELECT q.subject, avg(a.correct) FROM answers a JOIN questions q ON q.id = a.question_id
                       JOIN students s ON s.id = a.student_id WHERE s.batch = ? AND a.correct IS NOT NULL GROUP BY q.subject""", batch))
    lec = q("""SELECT l.subject, l.title, w.watched_pct, w.first_watched FROM lectures l
               LEFT JOIN watch w ON w.lecture_id = l.id AND w.student_id = ? ORDER BY l.course_order""", sid)
    subjects = [{"subject": s, "accuracy": mine.get(s, (None, 0))[0], "answered": mine.get(s, (None, 0))[1],
                 "batch_accuracy": theirs.get(s),
                 "lectures": [{"title": t, "pct": p, "first": f} for subj, t, p, f in lec if subj == s]} for s in _subjects()]
    reviews = [{"test": t, "signal": sg, "outcome": o, "opened": op, "decided": d, "note": nt} for t, sg, o, op, d, nt in q(
        """SELECT t.name, r.signal, r.outcome, r.opened_on, r.decided_on, r.note FROM integrity_reviews r
           JOIN tests t ON t.id = r.test_id WHERE r.student_id = ? ORDER BY r.opened_on""", sid)]
    return {"id": sid, "name": name, "batch": batch, "joined": joined,
            "mentor": q("SELECT mentor FROM batches WHERE name = ?", batch)[0][0], "last_active": max(days) if days else None,
            "weeks": weeks, "tests": tests_, "missed": missed, "subjects": subjects, "reviews": reviews}


SCHEMA_DOC = """Tables (dates are 'YYYY-MM-DD' text; today is the snapshot date from overview):
students(id, name, batch, joined_on); batches(name, mentor, starts_on)
lectures(id, subject, title, course_order, minutes)
watch(student_id, lecture_id, first_watched, last_watched, watched_pct 0-100): one row per student and lecture opened
activity(student_id, day, minutes): one row per day the student used the app
tests(id, name, kind 'Subject test'|'Grand test', subject, held_on, questions, minutes)
questions(id, test_id, number, subject, topic, kind 'Recall'|'Clinical vignette'|'Image-based', difficulty 'Easy'|'Medium'|'Hard', answer 0-3)
attempts(student_id, test_id, minutes_taken, tab_switches, correct, wrong, skipped, score)  score = 4*correct - wrong
answers(student_id, question_id, chosen 0-3 or NULL if skipped, correct 1/0 or NULL if skipped)
integrity_reviews(student_id, test_id, opened_on, signal, outcome 'confirmed'|'cleared'|'open', decided_on, note)
A lecture counts as watched fully at watched_pct >= 90 (the other tools use the same rule)."""


@tool
def run_sql(query: str) -> str:
    """Run one read-only SQLite SELECT for questions the other tools don't answer (custom lists, counts, comparisons).
    Returns up to 50 rows. The database refuses anything that isn't a read.
    """
    try:
        with closing(_db()) as con:
            deadline = time.monotonic() + 10
            con.set_progress_handler(lambda: time.monotonic() > deadline, 20000)
            cur = con.execute(query)
            rows = cur.fetchmany(51)
            cols = [d[0] for d in cur.description or []]
    except sqlite3.DatabaseError as e:
        if getattr(e, "sqlite_errorcode", None) in (sqlite3.SQLITE_AUTH, sqlite3.SQLITE_READONLY):
            return "Not allowed: the database is read-only; only SELECT queries run."
        return f"SQL error: {e}. Fix the query and try again."
    except sqlite3.Error as e:
        return f"SQL error: {e}. Fix the query and try again."
    cell = lambda v: f"{v:.2f}" if isinstance(v, float) else "NULL" if v is None else str(v)
    out = [" | ".join(cols)] + [" | ".join(cell(v) for v in r) for r in rows[:50]]
    return "\n".join(out) + ("\n(more rows not shown: add a LIMIT or aggregate)" if len(rows) > 50 else "")


run_sql.description += "\n" + SCHEMA_DOC

TOOLS = [overview, find_students, student_report, activity, lectures, tests, performance, integrity, rank_students, run_sql]
