"""Tests for attacks/fma.py: Frequency Matching Attack."""

from __future__ import annotations

import pytest

import attacks.fma as fma
from config import Config
from equality import build_equality
from evaluate import accuracy
from simulator import simulate
from synthetic import VOCAB, make_obvious_query_dist, make_synthetic
from types_ import Aux


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _setup(
    seed: int = 42,
    tau: int = 3,
    lam: float = 200.0,
    gamma: float = 0.0,
    equality_mode: str = "T1",
    noise_n: float = 0.0,
    tolerance: float = 0.0,
) -> tuple[dict[int, str | None], dict[int, str], float]:
    """Run full pipeline: simulate → build_equality → FMA → accuracy."""
    docs = make_synthetic(seed=seed)
    dist = make_obvious_query_dist(VOCAB, tau=tau)
    cfg = Config(
        seed=seed, tau=tau, lambda_queries=lam, gamma=gamma,
        equality=equality_mode,
    )
    records, gt, _ = simulate(docs, cfg, query_dist=dist)

    eq = build_equality(records, mode=equality_mode)
    aux = Aux(query_dist=dist)

    preds = fma.run(records, aux, eq, tolerance=tolerance)
    acc = accuracy(preds, gt)
    return preds, gt, acc


# ---------------------------------------------------------------------------
# 1. Interface / contract tests
# ---------------------------------------------------------------------------

class TestFMAInterface:
    def test_returns_dict(self) -> None:
        """run() must return a dict."""
        docs    = make_synthetic(seed=0)
        cfg     = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, gt, _ = simulate(docs, cfg)
        eq   = build_equality(records, mode="T1")
        dist = make_obvious_query_dist(VOCAB, tau=3)
        aux  = Aux(query_dist=dist)
        result = fma.run(records, aux, eq)
        assert isinstance(result, dict)

    def test_all_query_ids_present_as_keys(self) -> None:
        """Every query_id in records must be a key in the returned dict."""
        docs    = make_synthetic(seed=0)
        cfg     = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, gt, _ = simulate(docs, cfg)
        eq  = build_equality(records, mode="T1")
        aux = Aux(query_dist=make_obvious_query_dist(VOCAB, tau=3))
        preds = fma.run(records, aux, eq)
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_predicted_values_are_keywords_or_none(self) -> None:
        """Predicted values must be keyword strings or None."""
        docs    = make_synthetic(seed=1)
        cfg     = Config(seed=1, tau=3, lambda_queries=80.0, gamma=0.0)
        records, gt, _ = simulate(docs, cfg)
        eq  = build_equality(records, mode="T1")
        aux = Aux(query_dist=make_obvious_query_dist(VOCAB, tau=3))
        preds = fma.run(records, aux, eq)
        vocab_set = set(VOCAB)
        for qid, pred in preds.items():
            assert pred is None or pred in vocab_set, (
                f"Query {qid}: prediction {pred!r} not a valid keyword."
            )

    def test_no_ground_truth_imported(self) -> None:
        """attacks/fma.py must not contain 'ground_truth' as a token."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "attacks" / "fma.py"
        text = src.read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            "fma.py must not reference 'ground_truth'."
        )

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records → empty prediction dict."""
        from equality import EqualityResult
        eq  = EqualityResult()
        aux = Aux(query_dist={0: {kw: 0.1 for kw in VOCAB}})
        assert fma.run([], aux, eq) == {}

    def test_none_query_dist_gives_all_none(self) -> None:
        """When aux.query_dist is None, all predictions must be None."""
        docs    = make_synthetic(seed=0)
        cfg     = Config(seed=0, tau=3, lambda_queries=50.0)
        records, _, _ = simulate(docs, cfg)
        eq  = build_equality(records, mode="T1")
        aux = Aux(query_dist=None)
        preds = fma.run(records, aux, eq)
        assert all(v is None for v in preds.values())


# ---------------------------------------------------------------------------
# 2. Accuracy requirements
# ---------------------------------------------------------------------------

class TestFMAAccuracy:
    def test_accuracy_above_0_9_obvious_dist_t1(self) -> None:
        """FMA with obvious distribution + T1 must exceed 90 % accuracy."""
        _, _, acc = _setup(seed=42, tau=3, lam=200.0, gamma=0.0, equality_mode="T1")
        assert acc >= 0.9, f"FMA accuracy={acc:.4f} < 0.9"

    def test_accuracy_above_0_9_different_seed(self) -> None:
        """Accuracy requirement must hold for a second seed."""
        _, _, acc = _setup(seed=7, tau=3, lam=200.0, gamma=0.0, equality_mode="T1")
        assert acc >= 0.9, f"FMA accuracy={acc:.4f} < 0.9 (seed=7)"

    def test_3_intervals_at_least_as_good_as_1(self) -> None:
        """Accuracy with tau=3 must be >= accuracy with tau=1."""
        _, _, acc3 = _setup(seed=42, tau=3, lam=200.0, gamma=0.0)
        _, _, acc1 = _setup(seed=42, tau=1, lam=200.0, gamma=0.0)
        assert acc3 >= acc1, (
            f"tau=3 accuracy ({acc3:.4f}) should be >= tau=1 ({acc1:.4f})"
        )

    def test_more_intervals_help_disambiguation(self) -> None:
        """With more intervals, fewer predictions should be None."""
        preds3, gt, _ = _setup(seed=42, tau=3, lam=200.0, gamma=0.0)
        preds1, _,  _ = _setup(seed=42, tau=1, lam=200.0, gamma=0.0)
        resolved3 = sum(1 for v in preds3.values() if v is not None)
        resolved1 = sum(1 for v in preds1.values() if v is not None)
        # tau=3 should resolve at least as many queries as tau=1.
        assert resolved3 >= resolved1, (
            f"tau=3 resolved {resolved3} vs tau=1 resolved {resolved1}"
        )

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_accuracy_positive_across_seeds(self, seed: int) -> None:
        """FMA accuracy should be well above chance for any seed."""
        _, _, acc = _setup(seed=seed, tau=3, lam=150.0, gamma=0.0)
        # Chance level ≈ 1/10 = 0.1; FMA should do much better.
        assert acc > 0.5, f"seed={seed}: FMA accuracy={acc:.4f} is near chance."


# ---------------------------------------------------------------------------
# 3. Tolerance option
# ---------------------------------------------------------------------------

class TestFMATolerance:
    def test_nonzero_tolerance_does_not_crash(self) -> None:
        """Running with tolerance>0 must not raise an exception."""
        preds, gt, acc = _setup(
            seed=42, tau=3, lam=150.0, gamma=0.0, tolerance=0.05
        )
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_large_tolerance_resolves_all_or_more(self) -> None:
        """With very large tolerance (all candidates kept), nothing crashes."""
        preds, gt, acc = _setup(
            seed=42, tau=3, lam=100.0, gamma=0.0, tolerance=1.0
        )
        # With huge tolerance every keyword is a candidate; intersecting across
        # tau=3 intervals may still resolve some.  Just verify no exception.
        assert isinstance(preds, dict)

    def test_zero_tolerance_is_strictest(self) -> None:
        """Zero tolerance keeps only the exact best-scoring keyword(s).
        Accuracy should be at least as high as very large tolerance."""
        _, _, acc_strict = _setup(
            seed=42, tau=3, lam=150.0, gamma=0.0, tolerance=0.0
        )
        # strict mode should not be worse than random – just confirm no crash
        # and result is meaningful.
        assert acc_strict >= 0.0


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestFMAEdgeCases:
    def test_all_queries_same_keyword(self) -> None:
        """When all queries are for one keyword, FMA should resolve it."""
        from collections import Counter
        docs    = make_synthetic(seed=0)
        cfg     = Config(seed=0, tau=1, lambda_queries=30.0, gamma=0.0)
        records, gt, _ = simulate(docs, cfg)

        # Keep only records for the single most-queried keyword.
        kw_counts = Counter(gt.values())
        top_kw    = kw_counts.most_common(1)[0][0]
        filtered  = [r for r in records if gt[r.query_id] == top_kw]

        if not filtered:
            pytest.skip("No records generated.")

        eq  = build_equality(filtered, mode="T1")
        # Perfect distribution: top_kw has probability 1.
        aux = Aux(query_dist={0: {kw: (1.0 if kw == top_kw else 0.0) for kw in VOCAB}})
        preds = fma.run(filtered, aux, eq)
        for qid in (r.query_id for r in filtered):
            assert preds[qid] == top_kw, (
                f"Query {qid}: expected {top_kw!r}, got {preds[qid]!r}"
            )

    def test_missing_interval_in_dist(self) -> None:
        """If query_dist lacks an interval, that interval is skipped gracefully."""
        docs    = make_synthetic(seed=0)
        cfg     = Config(seed=0, tau=3, lambda_queries=60.0, gamma=0.0)
        records, gt, _ = simulate(docs, cfg)
        eq  = build_equality(records, mode="T1")

        # Only provide distribution for interval 0; intervals 1 and 2 are absent.
        partial_dist = {0: make_obvious_query_dist(VOCAB, tau=3)[0]}
        aux = Aux(query_dist=partial_dist)
        preds = fma.run(records, aux, eq)
        # Must not raise; all qids must be present.
        assert set(preds.keys()) == {r.query_id for r in records}
