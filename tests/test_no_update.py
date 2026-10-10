"""Tests for attacks/no_update.py: Attack without update knowledge (Section 6.3)."""

from __future__ import annotations

import pathlib
import pytest

import attacks.no_update as no_update
from config import Config
from equality import build_equality
from evaluate import accuracy
from simulator import make_aux_data_dist, make_aux_pre_snapshot, simulate
from synthetic import VOCAB, make_synthetic
from types_ import Aux, LeakageRecord


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _run_no_update(
    seed: int = 42,
    tau: int = 3,
    lam: float = 60.0,
    gamma: float = 0.0,
    mu0: int = 30,
    equality_mode: str = "T1",
) -> tuple[dict[int, str | None], dict[int, str], list[LeakageRecord]]:
    """Simulate and run no_update attack on synthetic data."""
    docs = make_synthetic(seed=seed)
    cfg = Config(
        seed=seed,
        tau=tau,
        lambda_queries=lam,
        gamma=gamma,
        equality=equality_mode,
    )
    records, gt, _ = simulate(docs, cfg)
    eq = build_equality(records, mode=equality_mode)
    snapshot = make_aux_pre_snapshot(docs, mu0=mu0)
    aux = Aux(pre_snapshot=snapshot)

    preds = no_update.run(records, aux, eq, mu0=mu0)
    return preds, gt, records


# ---------------------------------------------------------------------------
# 1. Interface & contract tests (including user prompt requirement)
# ---------------------------------------------------------------------------

class TestNoUpdateInterface:
    def test_returns_entry_for_every_query_on_synthetic_data(self) -> None:
        """Prompt requirement: Test that it returns an entry for every query on the synthetic data."""
        preds, _, records = _run_no_update(seed=42, lam=60.0, mu0=30)
        assert len(records) > 0, "Simulation produced no queries."
        assert len(preds) == len(records), (
            f"Expected {len(records)} entries, got {len(preds)}"
        )
        assert set(preds.keys()) == {r.query_id for r in records}, (
            "Not every query_id is present in the returned dictionary."
        )

    @pytest.mark.parametrize("seed", [0, 1, 7, 99])
    def test_all_query_ids_present_across_seeds(self, seed: int) -> None:
        """Every query_id must be present as a key across diverse seeds."""
        preds, _, records = _run_no_update(seed=seed, lam=50.0, mu0=20)
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_predicted_values_are_keywords_or_none(self) -> None:
        """Returned values must be either valid keyword strings or None."""
        preds, _, _ = _run_no_update(seed=42, lam=80.0, mu0=30)
        vocab_set = set(VOCAB)
        for qid, pred in preds.items():
            assert pred is None or pred in vocab_set, (
                f"Query {qid}: invalid prediction {pred!r}."
            )

    def test_no_ground_truth_imported(self) -> None:
        """attacks/no_update.py must not contain 'ground_truth' as a token."""
        src = pathlib.Path(__file__).parent.parent / "attacks" / "no_update.py"
        text = src.read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            "no_update.py must not reference 'ground_truth'."
        )

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records → empty prediction dict."""
        from equality import EqualityResult
        eq = EqualityResult()
        aux = Aux(pre_snapshot=[])
        assert no_update.run([], aux, eq, mu0=0) == {}

    def test_none_pre_snapshot_gives_all_none(self) -> None:
        """When aux.pre_snapshot is None, all query predictions must be None."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=40.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(pre_snapshot=None)
        preds = no_update.run(records, aux, eq, mu0=0)
        assert len(preds) == len(records)
        assert all(v is None for v in preds.values())


# ---------------------------------------------------------------------------
# 2. Accuracy of resolved queries
# ---------------------------------------------------------------------------

class TestNoUpdateAccuracy:
    def test_resolved_queries_are_accurate(self) -> None:
        """When the attack narrows down to a single candidate, it should be correct."""
        preds, gt, _ = _run_no_update(seed=42, lam=100.0, mu0=30)
        resolved = {qid: pred for qid, pred in preds.items() if pred is not None}
        if resolved:
            correct = sum(1 for qid, pred in resolved.items() if pred == gt[qid])
            acc = correct / len(resolved)
            assert acc >= 0.9, f"Resolved queries accuracy={acc:.4f} < 0.9"


# ---------------------------------------------------------------------------
# 3. Fallback behavior (if no query closest after mu0, use q)
# ---------------------------------------------------------------------------

class TestNoUpdateFallback:
    def test_large_mu0_falls_back_to_q(self) -> None:
        """When mu0 is beyond all query timestamps, fallback q~ = q executes cleanly."""
        preds, _, records = _run_no_update(seed=42, lam=50.0, mu0=999_999)
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_mu0_zero_runs_cleanly(self) -> None:
        """When mu0 = 0, all queries have timestamp >= mu0 and attack executes cleanly."""
        preds, _, records = _run_no_update(seed=42, lam=50.0, mu0=0)
        assert set(preds.keys()) == {r.query_id for r in records}


class TestNoUpdateVariants:
    def test_pvia_variant_returns_all_queries_and_validates_alpha(self) -> None:
        docs = make_synthetic(seed=12)
        records, _, _ = simulate(docs, Config(seed=12, tau=2, lambda_queries=20))
        equality = build_equality(records, "T1")
        aux = Aux(pre_snapshot=make_aux_pre_snapshot(docs, mu0=20))

        predictions = no_update.no_update_pvia(records, aux, equality, alpha=0.8, mu0=20)

        assert set(predictions) == {record.query_id for record in records}
        with pytest.raises(ValueError, match="alpha"):
            no_update.no_update_pvia(records, aux, equality, alpha=1.0)

    def test_lvia_variant_uses_interval_distribution(self) -> None:
        records = [
            LeakageRecord(0, "a", 0, 0, 2, 2),
            LeakageRecord(1, "b", 1, 1, 4, 4),
        ]
        equality = build_equality(records, "exact")
        aux = Aux(data_dist={0: {"first": 2}, 1: {"second": 4}})

        predictions = no_update.no_update_lvia(records, aux, equality, theta=0)

        assert predictions == {0: "first", 1: "second"}
