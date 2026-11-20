"""The question bank: loading, precomputed checks, and TF-IDF search."""
import json
import os

# One BLAS thread: TF-IDF on sparse text gains nothing from more, and each extra thread reserves a buffer
# (16 threads cost ~1 GB of committed memory per process). Must be set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from collections import Counter  # noqa: E402
from functools import cache  # noqa: E402
from pathlib import Path  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from .checks import run_all_checks  # noqa: E402

DATA = Path(__file__).parent / "data"


def _key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


class Bank:
    def __init__(self, items: list[dict]):
        self.items = items
        self.by_id = {x["id"]: x for x in items}
        self._pos = {x["id"]: i for i, x in enumerate(items)}
        self.findings, self.counts = run_all_checks(items)
        self._topic_counts = Counter(x["topic"] for x in items)
        # no stop-word removal: "not" / "except" change what a question asks
        self._vec = TfidfVectorizer(sublinear_tf=True)
        self._X = self._vec.fit_transform(x["question"] + " " + " ".join(x["options"]) for x in items)

    @classmethod
    def load(cls, path: Path = DATA / "bank.json") -> "Bank":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def topics(self, match: str = "") -> list[tuple[str, int]]:
        """Topics named match (ignoring case/punctuation); if none is, every topic whose name contains it."""
        k, ranked = _key(match), self._topic_counts.most_common()
        return [(t, n) for t, n in ranked if _key(t) == k] or [(t, n) for t, n in ranked if k in _key(t)]

    def _rank(self, vec, idx: list[int] | None, limit: int, exclude: int | None = None) -> list[tuple[int, float]]:
        X = self._X if idx is None else self._X[idx]
        scores = (X @ vec.T).toarray().ravel()
        order = [i for i in scores.argsort()[::-1] if scores[i] > 0]
        pos = order if idx is None else [idx[i] for i in order]
        return [(p, float(scores[i])) for p, i in zip(pos, order) if p != exclude][:limit]

    def search(self, query: str = "", topic: str = "", limit: int = 10) -> list[dict]:
        idx = None
        if topic:
            names = {t for t, _ in self.topics(topic)}
            idx = [i for i, x in enumerate(self.items) if x["topic"] in names]
            if not idx:
                return []
        if query.strip():
            return [self.items[p] for p, _ in self._rank(self._vec.transform([query]), idx, limit)]
        return [self.items[i] for i in (idx if idx is not None else range(len(self.items)))][:limit]

    def similar(self, question_id: str = "", text: str = "", limit: int = 5) -> list[tuple[dict, float]]:
        if question_id:
            if question_id not in self._pos:
                return []
            p = self._pos[question_id]
            hits = self._rank(self._X[p], None, limit, exclude=p)
        elif text.strip():
            hits = self._rank(self._vec.transform([text]), None, limit)
        else:
            return []
        return [(self.items[p], s) for p, s in hits]

    def overview(self) -> dict:
        return {"questions": len(self.items), "topics": len(self._topic_counts),
                "top_topics": self.topics()[:15], "counts": self.counts}


@cache
def get_bank() -> Bank:
    return Bank.load()
