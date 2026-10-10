"""Synthetic corpus and update-stream generation."""

from datetime import datetime, timezone
from typing import Any

import numpy as np

from types_ import Document


# Fixed vocabulary used throughout synthetic experiments.
VOCAB: list[str] = [f"kw{i}" for i in range(10)]

# Base epoch timestamps – one per interval, spaced one day apart.
_EPOCH_BASES: list[datetime] = [
    datetime(2024, 1, d, tzinfo=timezone.utc) for d in (1, 2, 3)
]


def _zipf_probs(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a Zipf-like probability vector over *n* items.

    Weights are proportional to 1/rank with a small random perturbation so
    that frequencies are unequal and, at the scale used here (100 docs,
    10 keywords), some keywords end up with unique counts.

    Args:
        n:   Number of items.
        rng: Seeded numpy Generator.

    Returns:
        1-D float array of length *n* summing to 1.
    """
    ranks = np.arange(1, n + 1, dtype=float)
    # Permute ranks so the "popular" keyword is not always kw0.
    rng.shuffle(ranks)
    weights = 1.0 / ranks
    # Small jitter to break exact ties between nearby ranks.
    weights += rng.uniform(0.0, 0.05, size=n)
    return weights / weights.sum()


def make_synthetic(seed: int) -> list[Document]:
    """Create a reproducible synthetic corpus of 100 documents.

    Documents are distributed across 3 time intervals (epochs). Each document
    receives 2–5 keywords drawn without replacement from a 10-keyword
    vocabulary. Keyword selection probabilities follow a Zipf-like distribution
    so that frequencies across documents are unequal and distinguishable.

    Args:
        seed: Master RNG seed for full reproducibility.

    Returns:
        List of 100 :class:`~types_.Document` objects whose timestamps fall in
        one of three distinct epochs.
    """
    rng = np.random.default_rng(seed)

    # One Zipf weight vector per interval keeps per-interval frequencies distinct.
    interval_probs: list[np.ndarray] = [_zipf_probs(len(VOCAB), rng) for _ in range(3)]

    docs: list[Document] = []
    n_docs = 100

    for doc_id in range(n_docs):
        # Round-robin interval assignment (≈ 33 docs per interval).
        interval = doc_id % 3
        probs = interval_probs[interval]

        # Draw 2–5 keywords without replacement.
        n_kw = int(rng.integers(2, 6))          # uniform in {2, 3, 4, 5}
        chosen = rng.choice(VOCAB, size=n_kw, replace=False, p=probs)
        keywords = frozenset(chosen)

        # Assign a timestamp within the interval's epoch day.
        offset_seconds = int(rng.integers(0, 86_400))
        ts = _EPOCH_BASES[interval].replace(
            hour=offset_seconds // 3600,
            minute=(offset_seconds % 3600) // 60,
            second=offset_seconds % 60,
        )

        docs.append(Document(doc_id=doc_id, keywords=keywords, timestamp=ts))

    return docs


def make_obvious_query_dist(
    vocab: list[str],
    tau: int,
) -> dict[int, dict[str, float]]:
    """Return a clearly different query distribution for each interval.

    In interval *t* the probability mass is concentrated on keywords whose
    index falls in the third ``[t * |vocab| // tau, (t+1) * |vocab| // tau)``
    of the vocabulary.  The remaining keywords receive a tiny uniform
    background probability so that every keyword has non-zero probability.

    This deliberately non-uniform, interval-varying distribution makes it easy
    for tests (and attacks) to detect per-interval structure.

    Args:
        vocab: Ordered list of keyword strings.
        tau:   Number of intervals (must satisfy 1 <= tau <= len(vocab)).

    Returns:
        Mapping ``interval -> {keyword: probability}`` for intervals
        ``0 .. tau-1``.  Each inner dict sums to 1.0.
    """
    n = len(vocab)
    # Background probability for non-focal keywords.
    background = 0.001
    result: dict[int, dict[str, float]] = {}

    for t in range(tau):
        # Keywords that are "hot" in interval t.
        lo = (t * n) // tau
        hi = ((t + 1) * n) // tau
        hot_kws = set(vocab[lo:hi])

        raw: dict[str, float] = {}
        for i, kw in enumerate(vocab):
            if kw in hot_kws:
                if tau > 1:
                    raw[kw] = float(2.5 ** (i - lo + 1))
                else:
                    raw[kw] = 1.0
            else:
                raw[kw] = background

        total = sum(raw.values())
        result[t] = {kw: v / total for kw, v in raw.items()}

    return result
