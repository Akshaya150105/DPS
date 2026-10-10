"""Tests for attacks/via.py: Volumetric Inference Attack."""

from __future__ import annotations

from collections import Counter
import pathlib
import pytest

import attacks.via as via
from config import Config
from equality import build_equality
from evaluate import accuracy
from simulator import simulate
from synthetic import VOCAB, make_synthetic
from types_ import Aux, Event, LeakageRecord


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _setup(
    seed: int = 42,
    tau: int = 3,
    lam: float = 200.0,
    gamma: float = 0.0,
    equality_mode: str = "T1",
    known_docs_mode: str = "documents",  # 'documents' | 'doc_ids' | 'none'
) -> tuple[dict[int, str | None], dict[int, str], list[int]]:
    """Run simulation → build_equality → VIA with full knowledge.

    Returns:
        (preds, gt, linked_qids)
    """
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

    if known_docs_mode == "documents":
        aux = Aux(known_docs=docs)
    elif known_docs_mode == "doc_ids":
        aux = Aux(known_docs=[d.doc_id for d in docs])
    else:
        aux = Aux(known_docs=None)

    preds = via.run(records, aux, eq, events)
    linked_qids = [
        qid for qid, next_qid in eq.linked_pairs.items() if next_qid is not None
    ]
    return preds, gt, linked_qids


# ---------------------------------------------------------------------------
# 1. Interface / contract tests
# ---------------------------------------------------------------------------

class TestVIAInterface:
    def test_returns_dict(self) -> None:
        """run() must return a dict."""
        preds, _, _ = _setup(seed=0, lam=50.0)
        assert isinstance(preds, dict)

    def test_all_query_ids_present_as_keys(self) -> None:
        """Every query_id in records must be a key in the returned dict."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, events = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)
        preds = via.run(records, aux, eq, events)
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_predicted_values_are_keywords_or_none(self) -> None:
        """Predicted values must be keyword strings or None."""
        preds, _, _ = _setup(seed=1, lam=80.0)
        vocab_set = set(VOCAB)
        for qid, pred in preds.items():
            assert pred is None or pred in vocab_set, (
                f"Query {qid}: prediction {pred!r} not a valid keyword."
            )

    def test_no_ground_truth_imported(self) -> None:
        """attacks/via.py must not contain 'ground_truth' as a token."""
        src = pathlib.Path(__file__).parent.parent / "attacks" / "via.py"
        text = src.read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            "via.py must not reference 'ground_truth'."
        )

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records → empty prediction dict."""
        from equality import EqualityResult
        eq = EqualityResult()
        aux = Aux(known_docs=[])
        assert via.run([], aux, eq, []) == {}

    def test_none_known_docs_gives_all_none(self) -> None:
        """When aux.known_docs is None, all predictions must be None."""
        preds, _, _ = _setup(seed=0, lam=50.0, known_docs_mode="none")
        assert all(v is None for v in preds.values())

    def test_empty_events_gives_all_none(self) -> None:
        """When events is empty, all predictions must be None."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)
        preds = via.run(records, aux, eq, [])
        assert all(v is None for v in preds.values())


# ---------------------------------------------------------------------------
# 2. Accuracy requirements (prompt requirement: > 0.9 with gamma=0, full knowledge)
# ---------------------------------------------------------------------------

class TestVIAAccuracy:
    def test_accuracy_above_0_9_on_linked_queries(self) -> None:
        """With gamma=0 and full knowledge, accuracy on queries with a linked pair must exceed 0.9."""
        preds, gt, linked_qids = _setup(seed=42, tau=3, lam=200.0, gamma=0.0)
        assert len(linked_qids) > 0, "No linked queries found."
        sub_preds = {qid: preds[qid] for qid in linked_qids}
        sub_gt = {qid: gt[qid] for qid in linked_qids}
        acc = accuracy(sub_preds, sub_gt)
        assert acc >= 0.9, f"VIA accuracy={acc:.4f} < 0.9 on linked queries."

    @pytest.mark.parametrize("seed", [1, 7, 123])
    def test_accuracy_above_0_9_across_seeds(self, seed: int) -> None:
        """Accuracy requirement must hold across different random seeds."""
        preds, gt, linked_qids = _setup(seed=seed, tau=3, lam=200.0, gamma=0.0)
        sub_preds = {qid: preds[qid] for qid in linked_qids}
        sub_gt = {qid: gt[qid] for qid in linked_qids}
        acc = accuracy(sub_preds, sub_gt)
        assert acc >= 0.9, f"seed={seed}: VIA accuracy={acc:.4f} < 0.9."


# ---------------------------------------------------------------------------
# 3. Documented behavior for queries without a linked pair
# ---------------------------------------------------------------------------

class TestVIALinkedPairBehavior:
    def test_queries_without_linked_pair_are_none(self) -> None:
        """Queries without a linked successor (linked_pairs[qid] is None) must be mapped to None."""
        docs = make_synthetic(seed=42)
        cfg = Config(seed=42, tau=3, lambda_queries=150.0, gamma=0.0)
        records, _, events = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)
        preds = via.run(records, aux, eq, events)

        unlinked_qids = [
            qid for qid, next_qid in eq.linked_pairs.items() if next_qid is None
        ]
        assert len(unlinked_qids) > 0, "Expected at least one unlinked query."
        for qid in unlinked_qids:
            assert preds[qid] is None, (
                f"Unlinked query {qid} should be None, got {preds[qid]!r}."
            )


# ---------------------------------------------------------------------------
# 4. Input representations (Document objects vs doc_id ints)
# ---------------------------------------------------------------------------

class TestVIAInputFormats:
    def test_supports_doc_ids_as_ints(self) -> None:
        """VIA must work identically when aux.known_docs contains integer IDs."""
        preds, gt, linked_qids = _setup(
            seed=42, tau=3, lam=200.0, gamma=0.0, known_docs_mode="doc_ids"
        )
        sub_preds = {qid: preds[qid] for qid in linked_qids}
        sub_gt = {qid: gt[qid] for qid in linked_qids}
        acc = accuracy(sub_preds, sub_gt)
        assert acc >= 0.9, f"doc_ids format accuracy={acc:.4f} < 0.9."


# ---------------------------------------------------------------------------
# 5. Volumetric filtering (fvol multiset difference disambiguation)
# ---------------------------------------------------------------------------

class TestVIAVolumeDisambiguation:
    def test_fvol_difference_filters_ties(self) -> None:
        """When two keywords have identical count L[w] == delta_a, multiset fvol difference eliminates the wrong one."""
        from equality import EqualityResult
        from types_ import Document
        from datetime import datetime, timezone

        # Doc 1 (vol 2): contains kwA and dummy
        # Doc 2 (vol 3): contains kwB and dummy
        # Doc 3 (vol 1): contains dummy
        # Between q0 and q1:
        # L[dummy] = 3 (!= 1)
        # L[kwA] = 1, L[kwB] = 1 (tie!)
        doc1 = Document(doc_id=1, keywords=frozenset({"kwA", "dummy"}), timestamp=datetime.now(timezone.utc))
        doc2 = Document(doc_id=2, keywords=frozenset({"kwB", "dummy", "extra"}), timestamp=datetime.now(timezone.utc))
        doc3 = Document(doc_id=3, keywords=frozenset({"dummy", "extra"}), timestamp=datetime.now(timezone.utc))

        events = [
            Event(mu=10, op="add", keyword="kwA", doc_id=1),
            Event(mu=10, op="add", keyword="dummy", doc_id=1),
            Event(mu=11, op="add", keyword="kwB", doc_id=2),
            Event(mu=11, op="add", keyword="dummy", doc_id=2),
            Event(mu=11, op="add", keyword="extra", doc_id=2),
            Event(mu=12, op="add", keyword="dummy", doc_id=3),
            Event(mu=12, op="add", keyword="extra", doc_id=3),
        ]

        # Query q0 at mu=5, Query q1 at mu=20
        # delta_a = 1: both kwA and kwB have L[w] == 1
        q0 = LeakageRecord(
            query_id=0, token="tok0", interval=0, timestamp=5,
            ulen=0, rlen=0, fvol=Counter()
        )
        # q1 observed forward volume shows that the newly inserted document had volume 3!
        q1 = LeakageRecord(
            query_id=1, token="tok1", interval=1, timestamp=20,
            ulen=1, rlen=1, fvol=Counter({3: 1})
        )

        eq = EqualityResult(
            groups={0: [0, 1]},
            group_of={0: 0, 1: 0},
            linked_pairs={0: 1, 1: None},
        )
        aux = Aux(known_docs=[doc1, doc2, doc3])

        preds = via.run([q0, q1], aux, eq, events)
        # L[kwA] = 1, V[kwA] = Counter({2: 1}) -> volume 3 is missing!
        # L[kwB] = 1, V[kwB] = Counter({3: 1}) -> volume 3 is present!
        # Thus kwA is dropped, and kwB is uniquely assigned to q0!
        assert preds[0] == "kwB"
        assert preds[1] is None
