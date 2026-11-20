"""Exact checks over an MCQ bank. They only compare text (ignoring case, punctuation and spacing); they never
interpret it. Every judgment (a wrong key, an explanation backing another option, a badly worded item) is made by
the AI reviewer in agent.py and labelled AI opinion."""
from collections import Counter, defaultdict

L = "ABCD"
CHECKS = {  # check -> severity
    "key_conflict": "high",
    "duplicate_options": "medium",
    "blank_option": "medium",
}
CHECK_TITLES = {
    "key_conflict": "Same question, different answer keys",
    "duplicate_options": "Same option appears twice",
    "blank_option": "Empty option",
}


def norm(s) -> str:
    """'  Atropine. ' and 'ATROPINE' compare equal."""
    return " ".join("".join(c if c.isalnum() else " " for c in str(s or "").lower()).split())


def _finding(check: str, q: dict, detail: str, related: list[str] | None = None) -> dict:
    return {"key": f"{check}:{q['id']}", "check": check, "severity": CHECKS[check], "source": "exact",
            "question_id": q["id"], "related_ids": related or [], "title": CHECK_TITLES[check], "detail": detail}


def _opt(q: dict, i: int) -> str:
    return f"({L[i]}) {q['options'][i]}"


def run_all_checks(bank: list[dict]) -> tuple[list[dict], dict[str, int]]:
    findings: list[dict] = []
    groups = defaultdict(list)
    for q in bank:
        on = [norm(o) for o in q["options"]]
        blanks = [i for i, o in enumerate(on) if not o]
        if blanks:
            findings.append(_finding("blank_option", q, f"Option {', '.join(L[i] for i in blanks)} is blank."))
        seen = {}
        for i, o in enumerate(on):
            if o and o in seen:
                findings.append(_finding("duplicate_options", q, f"Options {L[seen[o]]} and {L[i]} are both '{q['options'][i].strip()}'."))
                break
            seen.setdefault(o, i)
        if norm(q["question"]):
            groups[(norm(q["question"]), tuple(sorted(on)))].append(q)

    counts = defaultdict(int)
    for qs in groups.values():
        if len(qs) < 2:
            continue
        if len({norm(q["options"][q["answer"]]) for q in qs}) > 1:
            keys = Counter(_opt(q, q["answer"]) for q in qs)
            detail = (f"Saved {len(qs)} times with different keys: "
                      + "; ".join(f"{k} in {n} cop{'ies' if n > 1 else 'y'}" if i == 0 else f"{k} in {n}"
                                  for i, (k, n) in enumerate(keys.items())) + ".")
            findings.append(_finding("key_conflict", qs[0], detail, [q["id"] for q in qs[1:]]))
        else:
            counts["exact_duplicate_groups"] += 1
    for f in findings:
        counts[f["check"]] += 1
    return findings, {**{c: 0 for c in CHECKS}, "exact_duplicate_groups": 0, **counts}
