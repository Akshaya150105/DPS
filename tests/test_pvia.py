"""Tests for attacks/pvia.py: Partial Volumetric Inference Attack."""

from __future__ import annotations

from collections import Counter
import pathlib
import numpy as np
import pytest

import attacks.pvia as pvia
from config import Config
from equality import build_equality
from evaluate import accuracy
from simulator import make_aux_known_docs, simulate
from synthetic import VOCAB, make_synthetic
from types_ import Aux, Event, LeakageRecord


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _run_pvia(
    seed: int = 1,
    tau: int = 3,
    lam: float = 200.0,
    gamma: float = 0.0,
    alpha: float = 0.8,
    equality_mode: str = "T1",
) -> tuple[dict[int, str | None], dict[int, str], list[int]]:
    """Simulate and run PVIA for a given seed, gamma, and alpha.

    Returns:
        (preds, gt, linked_qids)
    """
    rng = np.random.default_rng(seed)
    docs = make_synthetic(seed=seed)
    cfg = Config(
        seed=seed,
        tau=tau,
        lambda_queries=lam,
        gamma=gamma,
        equality=equality_mode,
    )
    records, gt, events = simulate(docs, cfg)
    eq = build_equality(records, mode=equality_mode)
    known_ids = make_aux_known_docs(events, alpha=alpha, rng=rng)
    aux = Aux(known_docs=known_ids)

    preds = pvia.run(records, aux, eq, events, alpha=alpha)
    linked_qids = [
        qid for qid, next_qid in eq.linked_pairs.items() if next_qid is not None
    ]
    return preds, gt, linked_qids


def _mean_accuracy_over_seeds(
    seeds: list[int],
    alpha: float,
    gamma: float,
) -> float:
    """Compute mean accuracy on linked queries across given seeds."""
    accs: list[float] = []
    for s in seeds:
        preds, gt, linked_qids = _run_pvia(seed=s, alpha=alpha, gamma=gamma)
        sub_preds = {q: preds[q] for q in linked_qids}
        sub_gt = {q: gt[q] for q in linked_qids}
        accs.append(accuracy(sub_preds, sub_gt))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# 1. Interface / validation tests
# ---------------------------------------------------------------------------

class TestPVIAInterface:
    def test_returns_dict(self) -> None:
        """run() must return a dict."""
        preds, _, _ = _run_pvia(seed=0, lam=50.0, alpha=0.5)
        assert isinstance(preds, dict)

    def test_all_query_ids_present_as_keys(self) -> None:
        """Every query_id in records must be a key in the returned dict."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, events = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)
        preds = pvia.run(records, aux, eq, events, alpha=0.5)
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_predicted_values_are_keywords_or_none(self) -> None:
        """Predicted values must be keyword strings or None."""
        preds, _, _ = _run_pvia(seed=1, lam=80.0, alpha=0.7)
        vocab_set = set(VOCAB)
        for qid, pred in preds.items():
            assert pred is None or pred in vocab_set, (
                f"Query {qid}: prediction {pred!r} not a valid keyword."
            )

    def test_no_ground_truth_imported(self) -> None:
        """attacks/pvia.py must not contain 'ground_truth' as a token."""
        src = pathlib.Path(__file__).parent.parent / "attacks" / "pvia.py"
        text = src.read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            "pvia.py must not reference 'ground_truth'."
        )

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records → empty prediction dict."""
        from equality import EqualityResult
        eq = EqualityResult()
        aux = Aux(known_docs=[])
        assert pvia.run([], aux, eq, [], alpha=0.5) == {}

    def test_none_known_docs_gives_all_none(self) -> None:
        """When aux.known_docs is None, all predictions must be None."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, events = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=None)
        preds = pvia.run(records, aux, eq, events, alpha=0.5)
        assert all(v is None for v in preds.values())

    def test_empty_events_gives_all_none(self) -> None:
        """When events is empty, all predictions must be None."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)
        preds = pvia.run(records, aux, eq, [], alpha=0.5)
        assert all(v is None for v in preds.values())


class TestPVIAValidation:
    def test_alpha_zero_raises(self) -> None:
        """alpha=0 must raise ValueError (open interval)."""
        with pytest.raises(ValueError, match="open interval"):
            pvia.run([], Aux(known_docs=[]), None, [], alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        """alpha=1 must raise ValueError (open interval)."""
        with pytest.raises(ValueError, match="open interval"):
            pvia.run([], Aux(known_docs=[]), None, [], alpha=1.0)

    def test_alpha_negative_raises(self) -> None:
        """Negative alpha must raise ValueError."""
        with pytest.raises(ValueError, match="open interval"):
            pvia.run([], Aux(known_docs=[]), None, [], alpha=-0.1)

    def test_alpha_greater_than_one_raises(self) -> None:
        """alpha > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="open interval"):
            pvia.run([], Aux(known_docs=[]), None, [], alpha=1.2)


# ---------------------------------------------------------------------------
# 2. Accuracy trends (prompt requirements)
# ---------------------------------------------------------------------------

class TestPVIATrends:
    SEEDS = [1, 2, 3, 4, 5]

    def test_accuracy_decreases_when_alpha_decreases(self) -> None:
        """Averaged over 5 seeds, accuracy decreases when alpha decreases."""
        acc_high = _mean_accuracy_over_seeds(self.SEEDS, alpha=0.8, gamma=0.0)
        acc_mid = _mean_accuracy_over_seeds(self.SEEDS, alpha=0.5, gamma=0.0)
        acc_low = _mean_accuracy_over_seeds(self.SEEDS, alpha=0.2, gamma=0.0)

        assert acc_high > acc_mid, (
            f"Expected acc(alpha=0.8)={acc_high:.4f} > acc(alpha=0.5)={acc_mid:.4f}"
        )
        assert acc_mid > acc_low, (
            f"Expected acc(alpha=0.5)={acc_mid:.4f} > acc(alpha=0.2)={acc_low:.4f}"
        )

    def test_accuracy_decreases_when_gamma_increases(self) -> None:
        """Averaged over 5 seeds, accuracy decreases when deletion rate gamma increases."""
        acc_no_del = _mean_accuracy_over_seeds(self.SEEDS, alpha=0.8, gamma=0.0)
        acc_with_del = _mean_accuracy_over_seeds(self.SEEDS, alpha=0.8, gamma=0.2)

        assert acc_no_del > acc_with_del, (
            f"Expected acc(gamma=0.0)={acc_no_del:.4f} > acc(gamma=0.2)={acc_with_del:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Linked pair & tie-breaking behavior
# ---------------------------------------------------------------------------

class TestPVIALinkedPairBehavior:
    def test_queries_without_linked_pair_are_none(self) -> None:
        """Queries without a linked successor (linked_pairs[qid] is None) must be mapped to None."""
        preds, _, _ = _run_pvia(seed=42, lam=150.0, alpha=0.7)
        docs = make_synthetic(seed=42)
        cfg = Config(seed=42, tau=3, lambda_queries=150.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")

        unlinked_qids = [
            qid for qid, next_qid in eq.linked_pairs.items() if next_qid is None
        ]
        assert len(unlinked_qids) > 0
        for qid in unlinked_qids:
            assert preds[qid] is None, (
                f"Unlinked query {qid} should be None, got {preds[qid]!r}."
            )

    def test_tie_breaking_by_distance_to_window(self) -> None:
        """When multiple candidates have identical multiset overlap, break ties by distance to window."""
        from equality import EqualityResult
        from types_ import Document
        from datetime import datetime, timezone

        # Two documents: doc1 with kwA (vol 2), doc2 with kwB (vol 2)
        # Both have same volume multiset overlap with diff_fvol {2: 1}.
        # delta_a = 4. alpha = 0.5. Window is [2, 4].
        # kwA appears in 1 known doc (L[kwA] = 1, outside window by 1).
        # kwB appears in 2 known docs (L[kwB] = 2, inside window, distance 0).
        doc1 = Document(doc_id=1, keywords=frozenset({"kwA"}), timestamp=datetime.now(timezone.utc))
        doc2 = Document(doc_id=2, keywords=frozenset({"kwB"}), timestamp=datetime.now(timezone.utc))
        doc3 = Document(doc_id=3, keywords=frozenset({"kwB"}), timestamp=datetime.now(timezone.utc))

        events = [
            Event(mu=10, op="add", keyword="kwA", doc_id=1),
            Event(mu=11, op="add", keyword="kwB", doc_id=2),
            Event(mu=12, op="add", keyword="kwB", doc_id=3),
        ]

        q0 = LeakageRecord(query_id=0, token="tok0", interval=0, timestamp=5, ulen=0, rlen=0, fvol=Counter())
        q1 = LeakageRecord(query_id=1, token="tok1", interval=1, timestamp=20, ulen=4, rlen=4, fvol=Counter({2: 1}))

        eq = EqualityResult(
            groups={0: [0, 1]},
            group_of={0: 0, 1: 0},
            linked_pairs={0: 1, 1: None},
        )
        aux = Aux(known_docs=[doc1, doc2, doc3])

        # delta_a = 4. alpha=0.5 -> window [2, 4].
        # L[kwA] = 1 (outside window). L[kwB] = 2 (in window).
        # Both have overlap with {2: 1} (if volumes match).
        preds = pvia.run([q0, q1], aux, eq, events, alpha=0.5)
        # kwB is in the candidate window [2, 4], while kwA is outside.
        assert preds[0] == "kwB"
