"""Build data/bank.json (MedMCQA Pharmacology) with 10 planted errors, and data/planted.json (the answer sheet).
`--heldout` builds data/heldout.json (past-year questions from the validation split, used only by evals).

Usage: python prepare_data.py [path/to/medmcqa-train.parquet]   (downloads from Hugging Face if no path)
"""
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .checks import norm, run_all_checks

URL = "https://huggingface.co/api/datasets/openlifescienceai/medmcqa/parquet/default/train/0.parquet"
AI_TOPIC = "Autonomic nervous system"


def to_items(df: pd.DataFrame) -> list[dict]:
    df = df[df.subject_name == "Pharmacology"]
    s = lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
    return [
        {"id": r.id, "question": s(r.question), "options": [s(r.opa), s(r.opb), s(r.opc), s(r.opd)],
         "answer": int(r.cop), "explanation": s(r.exp)[:700], "topic": s(r.topic_name) or "Unspecified"}
        for r in df.itertuples()
    ]


def names_only_its_key(x: dict) -> bool:
    """Test-data filter: the explanation contains the keyed option's text and no other option's text."""
    exp, opts = f" {norm(x['explanation'])} ", [f" {norm(o)} " for o in x["options"]]
    return all(len(o) > 5 and (o in exp) == (i == x["answer"]) for i, o in enumerate(opts))


def plant_errors(items: list[dict], seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Plant 10 errors: 5 the exact checks can see, 5 (all in AI_TOPIC) that only the AI review can catch."""
    rng = random.Random(seed)
    bank = [dict(x, options=list(x["options"])) for x in items]
    findings, _ = run_all_checks(bank)
    flagged = {i for f in findings for i in [f["question_id"], *f["related_ids"]]}
    stems = Counter(norm(x["question"]) for x in bank)
    pool = [x for x in bank if x["id"] not in flagged and stems[norm(x["question"])] == 1
            and all(norm(o) for o in x["options"]) and len({norm(o) for o in x["options"]}) == 4]
    rng.shuffle(pool)
    used, planted = set(), []

    def take(pred=lambda x: True):
        x = next(x for x in pool if x["id"] not in used and pred(x))
        used.add(x["id"])
        return x

    def wrong(x):
        return rng.choice([i for i in range(4) if i != x["answer"]])

    elsewhere = lambda x: x["topic"] != AI_TOPIC
    for _ in range(2):  # the same question re-entered with a different key
        x = take(elsewhere)
        opts = x["options"][:]
        rng.shuffle(opts)
        c = dict(x, id=x["id"] + "-copy", options=opts, answer=opts.index(x["options"][wrong(x)]))
        bank.insert(rng.randrange(len(bank)), c)
        planted.append({"type": "key_conflict", "ids": [x["id"], c["id"]], "topic": x["topic"]})
    for _ in range(2):  # a distractor duplicated
        x = take(elsewhere)
        j, k = rng.sample([i for i in range(4) if i != x["answer"]], 2)
        x["options"][j] = x["options"][k]
        planted.append({"type": "duplicate_options", "ids": [x["id"]], "topic": x["topic"]})
    x = take(elsewhere)  # a distractor blanked
    x["options"][wrong(x)] = ""
    planted.append({"type": "blank_option", "ids": [x["id"]], "topic": x["topic"]})
    for _ in range(2):  # key changed; the explanation still gives the original answer
        x = take(lambda x: x["topic"] == AI_TOPIC and names_only_its_key(x))
        original, x["answer"] = x["answer"], wrong(x)
        planted.append({"type": "explanation_conflict", "ids": [x["id"]], "topic": x["topic"], "original": original})
    for _ in range(3):  # silent wrong key, no explanation left
        x = take(lambda x: x["topic"] == AI_TOPIC and max(len(o) for o in x["options"]) <= 40)
        original, x["answer"], x["explanation"] = x["answer"], wrong(x), ""
        planted.append({"type": "ai_only_wrong_key", "ids": [x["id"]], "topic": x["topic"], "original": original})

    new_id = {}
    for n, x in enumerate(bank, 1):
        new_id[x["id"]] = x["id"] = f"PH-{n:05d}"
    for p in planted:
        p["ids"] = [new_id[i] for i in p["ids"]]
    return bank, planted


def _read(path: str | None, url: str) -> pd.DataFrame:
    if path:
        return pd.read_parquet(path)
    import requests
    print(f"Downloading {url.rsplit('/', 2)[-2]} split...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def main(path: str | None = None) -> None:
    bank, planted = plant_errors(to_items(_read(path, URL)))
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    (out / "bank.json").write_text(json.dumps(bank, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "planted.json").write_text(json.dumps(planted, indent=1), encoding="utf-8")
    print(f"{len(bank)} questions, {len(planted)} planted")


def heldout(path: str | None = None) -> None:
    """data/heldout.json: Pharmacology past-year questions from MedMCQA's validation split (not in the bank), for evals."""
    items = to_items(_read(path, URL.replace("/train/", "/validation/")))
    for i, x in enumerate(items):
        x["id"] = f"PYQ-{i:04d}"
    (Path(__file__).parent / "data" / "heldout.json").write_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(items)} held-out questions")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--heldout"]:
        heldout(args[1] if len(args) > 1 else None)
    else:
        main(args[0] if args else None)
