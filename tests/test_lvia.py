"""Tests for attacks/lvia.py: Length-based Volumetric Inference Attack."""

from __future__ import annotations

import pathlib
import numpy as np
import pytest

import attacks.lvia as lvia
from config import Config
from equality import build_equality
from evaluate import accuracy
from simulator import make_aux_data_dist, simulate
from synthetic import VOCAB, make_synthetic
from types_ import Aux, LeakageRecord


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _run_lvia(
    seed: int = 1,
    tau: int = 3,
    lam: float = 200.0,
    gamma: float = 0.0,
    mode: str = "K",
    eta: float = 0.5,
    equality_mode: str = "T1",
) -> tuple[dict[int, str | None], dict[int, str]]:
    """Simulate and run LVIA for a given seed and mode."""
    rng = np.random.default_rng(seed)
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

    data_dist_eta = 1.0 if mode.upper() == "K" else eta
    dist = make_aux_data_dist(docs, tau=tau, eta=data_dist_eta, rng=rng)
    aux = Aux(data_dist=dist)

    preds = lvia.run(records, aux, eq, mode=mode, eta=eta)
    return preds, gt


def _mean_acc_over_seeds(
    seeds: list[int],
    mode: str = "K",
    tau: int = 3,
    eta: float = 0.5,
) -> float:
    """Compute mean accuracy over the given seeds."""
    accs: list[float] = []
    for s in seeds:
        preds, gt = _run_lvia(seed=s, tau=tau, mode=mode, eta=eta)
        accs.append(accuracy(preds, gt))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# 1. Interface & validation tests
# ---------------------------------------------------------------------------

class TestLVIAInterface:
    def test_returns_dict(self) -> None:
        """run() must return a dict."""
        preds, _ = _run_lvia(seed=0, lam=50.0)
        assert isinstance(preds, dict)

    def test_all_query_ids_present_as_keys(self) -> None:
        """Every query_id in records must be a key in the returned dict."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        dist = make_aux_data_dist(docs, tau=3, eta=1.0, rng=np.random.default_rng(0))
        aux = Aux(data_dist=dist)
        preds = lvia.run(records, aux, eq, mode="K")
        assert set(preds.keys()) == {r.query_id for r in records}

    def test_predicted_values_are_keywords_or_none(self) -> None:
        """Predicted values must be keyword strings or None."""
        preds, _ = _run_lvia(seed=1, lam=80.0, mode="K")
        vocab_set = set(VOCAB)
        for qid, pred in preds.items():
            assert pred is None or pred in vocab_set, (
                f"Query {qid}: prediction {pred!r} not a valid keyword."
            )

    def test_no_ground_truth_imported(self) -> None:
        """attacks/lvia.py must not contain 'ground_truth' as a token."""
        src = pathlib.Path(__file__).parent.parent / "attacks" / "lvia.py"
        text = src.read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            "lvia.py must not reference 'ground_truth'."
        )

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records → empty prediction dict."""
        from equality import EqualityResult
        eq = EqualityResult()
        aux = Aux(data_dist={0: {"kw0": 5}})
        assert lvia.run([], aux, eq, mode="K") == {}

    def test_none_data_dist_gives_all_none(self) -> None:
        """When aux.data_dist is None, all predictions must be None."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(data_dist=None)
        preds = lvia.run(records, aux, eq, mode="K")
        assert all(v is None for v in preds.values())

    def test_case_insensitive_modes(self) -> None:
        """Both lower and uppercase mode names ('k', 's', 'K', 'S') must be accepted."""
        docs = make_synthetic(seed=0)
        cfg = Config(seed=0, tau=3, lambda_queries=50.0, gamma=0.0)
        records, _, _ = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        dist = make_aux_data_dist(docs, tau=3, eta=1.0, rng=np.random.default_rng(0))
        aux = Aux(data_dist=dist)
        preds_upper = lvia.run(records, aux, eq, mode="K")
        preds_lower = lvia.run(records, aux, eq, mode="k")
        assert preds_upper == preds_lower

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode string must raise ValueError."""
        with pytest.raises(ValueError, match="Expected 'K' or 'S'"):
            lvia.run([], Aux(data_dist={}), None, mode="Z")

    def test_invalid_eta_raises(self) -> None:
        """eta not in (0, 1] in mode S must raise ValueError."""
        with pytest.raises(ValueError):
            lvia.run([], Aux(data_dist={0: {"k": 1}}), None, mode="S", eta=0.0)
        with pytest.raises(ValueError):
            lvia.run([], Aux(data_dist={0: {"k": 1}}), None, mode="S", eta=1.5)


# ---------------------------------------------------------------------------
# 2. Accuracy trends (prompt requirements)
# ---------------------------------------------------------------------------

class TestLVIATrends:
    SEEDS = [1, 2, 3, 4, 5]

    def test_mode_k_at_least_as_accurate_as_mode_s(self) -> None:
        """Mode K must be at least as accurate as mode S averaged over 5 seeds."""
        acc_K = _mean_acc_over_seeds(self.SEEDS, mode="K", tau=3)
        acc_S = _mean_acc_over_seeds(self.SEEDS, mode="S", tau=3, eta=0.5)

        assert acc_K >= acc_S, (
            f"Expected acc(K)={acc_K:.4f} >= acc(S)={acc_S:.4f}"
        )

    def test_3_intervals_at_least_as_accurate_as_1(self) -> None:
        """Accuracy with tau=3 must be at least as high as tau=1 averaged over 5 seeds."""
        acc_tau3 = _mean_acc_over_seeds(self.SEEDS, mode="K", tau=3)
        acc_tau1 = _mean_acc_over_seeds(self.SEEDS, mode="K", tau=1)

        assert acc_tau3 >= acc_tau1, (
            f"Expected acc(tau=3)={acc_tau3:.4f} >= acc(tau=1)={acc_tau1:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Hand-crafted candidate matching & tie breaking
# ---------------------------------------------------------------------------

class TestLVIACandidateMatching:
    def test_exact_match_in_mode_k(self) -> None:
        """In mode K with theta=0, exact response length match resolves uniquely."""
        from equality import EqualityResult

        # Two records: q0 in interval 0 with rlen 10; q1 in interval 1 with rlen 25
        q0 = LeakageRecord(query_id=0, token="tok0", interval=0, timestamp=5, ulen=10, rlen=10)
        q1 = LeakageRecord(query_id=1, token="tok1", interval=1, timestamp=20, ulen=25, rlen=25)

        eq = EqualityResult(
            groups={0: [0, 1]},
            group_of={0: 0, 1: 0},
            linked_pairs={0: 1, 1: None},
        )

        # In interval 0: kwA has count 10 (cum 10), kwB has count 15 (cum 15)
        # In interval 1: kwA has count 15 (cum 25), kwB has count 10 (cum 25)
        # Note: in interval 1 both have cum 25, but in interval 0 only kwA has cum 10!
        # Pooling across intervals: kwA gets votes in interval 0 and 1 (2 votes).
        # kwB gets vote only in interval 1 (1 vote).
        aux = Aux(data_dist={
            0: {"kwA": 10, "kwB": 15},
            1: {"kwA": 15, "kwB": 10},
        })

        preds = lvia.run([q0, q1], aux, eq, mode="K")
        assert preds[0] == "kwA"
        assert preds[1] == "kwA"

    def test_relative_window_in_mode_s(self) -> None:
        """In mode S with eta=0.5, counts scaled by 2 match within theta window."""
        from equality import EqualityResult

        q0 = LeakageRecord(query_id=0, token="tok0", interval=0, timestamp=5, ulen=20, rlen=20)
        eq = EqualityResult(groups={0: [0]}, group_of={0: 0}, linked_pairs={0: None})

        # Sampled count is 11 (scaled by 1/0.5 = 22, within 15% of 20)
        aux = Aux(data_dist={0: {"kwA": 11, "kwB": 2}})
        preds = lvia.run([q0], aux, eq, mode="S", eta=0.5)
        assert preds[0] == "kwA"
