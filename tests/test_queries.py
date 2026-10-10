"""Tests for generate_query_distribution and draw_queries in simulator.py."""

import csv
import pathlib

import numpy as np
import pytest

from simulator import draw_queries, generate_query_distribution
from synthetic import VOCAB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SMALL = VOCAB[:5]   # use a 5-keyword slice for fast tests


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. generate_query_distribution – synthetic mode
# ---------------------------------------------------------------------------

class TestSyntheticQueryDist:
    def test_returns_tau_intervals(self) -> None:
        """Result must have exactly tau entries keyed 0..tau-1."""
        dist = generate_query_distribution(VOCAB, tau=3, rng=make_rng())
        assert set(dist.keys()) == {0, 1, 2}

    def test_each_interval_covers_full_vocab(self) -> None:
        """Every keyword in vocab must appear in each interval's distribution."""
        dist = generate_query_distribution(VOCAB, tau=3, rng=make_rng())
        for t, d in dist.items():
            assert set(d.keys()) == set(VOCAB), (
                f"Interval {t} is missing keywords: {set(VOCAB) - set(d.keys())}"
            )

    def test_probabilities_sum_to_one(self) -> None:
        """Each interval's probabilities must sum to 1.0 within float tolerance."""
        dist = generate_query_distribution(VOCAB, tau=5, rng=make_rng())
        for t, d in dist.items():
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Interval {t} sums to {total}, not 1.0"
            )

    def test_all_probabilities_positive(self) -> None:
        """Every probability must be strictly positive."""
        dist = generate_query_distribution(VOCAB, tau=3, rng=make_rng())
        for t, d in dist.items():
            for kw, p in d.items():
                assert p > 0, f"Interval {t}, keyword {kw}: p={p}"

    def test_distributions_differ_across_intervals(self) -> None:
        """Interval distributions must not all be identical."""
        dist = generate_query_distribution(VOCAB, tau=3, rng=make_rng())
        vec0 = list(dist[0].values())
        vec1 = list(dist[1].values())
        assert vec0 != pytest.approx(vec1), (
            "Intervals 0 and 1 have the same distribution."
        )

    def test_reproducible_with_same_seed(self) -> None:
        """Two calls with the same seed must produce identical distributions."""
        dist_a = generate_query_distribution(VOCAB, tau=3, rng=make_rng(7))
        dist_b = generate_query_distribution(VOCAB, tau=3, rng=make_rng(7))
        for t in range(3):
            for kw in VOCAB:
                assert dist_a[t][kw] == pytest.approx(dist_b[t][kw]), (
                    f"Mismatch at interval={t}, keyword={kw}"
                )

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different distributions."""
        dist_a = generate_query_distribution(VOCAB, tau=3, rng=make_rng(1))
        dist_b = generate_query_distribution(VOCAB, tau=3, rng=make_rng(2))
        vecs_a = [list(dist_a[t].values()) for t in range(3)]
        vecs_b = [list(dist_b[t].values()) for t in range(3)]
        assert vecs_a != vecs_b

    @pytest.mark.parametrize("tau", [1, 2, 4])
    def test_various_tau(self, tau: int) -> None:
        """generate_query_distribution must handle any tau >= 1."""
        dist = generate_query_distribution(VOCAB_SMALL, tau=tau, rng=make_rng())
        assert len(dist) == tau
        for d in dist.values():
            assert abs(sum(d.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. generate_query_distribution – trends mode
# ---------------------------------------------------------------------------

class TestTrendsQueryDist:
    def test_missing_file_raises_file_not_found(self, tmp_path: pathlib.Path) -> None:
        """trends mode must raise FileNotFoundError if the CSV is absent."""
        # The CSV doesn't exist in the default datasets/trends/ location.
        with pytest.raises(FileNotFoundError, match="trends.csv"):
            generate_query_distribution(VOCAB_SMALL, tau=2, rng=make_rng(),
                                        mode="trends")

    def test_trends_loads_and_normalises(self, tmp_path: pathlib.Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        """trends mode must load the CSV, normalise rows, and return tau dicts."""
        # Create a minimal trends.csv in a temp location and redirect pathlib.
        trends_dir = tmp_path / "datasets" / "trends"
        trends_dir.mkdir(parents=True)
        csv_file = trends_dir / "trends.csv"

        vocab = VOCAB_SMALL
        with csv_file.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=vocab)
            writer.writeheader()
            # Two intervals with raw counts
            writer.writerow({kw: str(i + 1) for i, kw in enumerate(vocab)})
            writer.writerow({kw: str(5 - i) for i, kw in enumerate(vocab)})

        # Monkey-patch pathlib.Path so "datasets/trends/trends.csv" resolves
        # to our temp file.
        monkeypatch.chdir(tmp_path)
        dist = generate_query_distribution(vocab, tau=2, rng=make_rng(),
                                           mode="trends")
        assert set(dist.keys()) == {0, 1}
        for d in dist.values():
            assert abs(sum(d.values()) - 1.0) < 1e-9
            for kw in vocab:
                assert d[kw] > 0

    def test_unknown_mode_raises_value_error(self) -> None:
        """An unrecognised mode string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            generate_query_distribution(VOCAB_SMALL, tau=1, rng=make_rng(),
                                        mode="invalid")


# ---------------------------------------------------------------------------
# 3. draw_queries
# ---------------------------------------------------------------------------

class TestDrawQueries:
    def _uniform_dist(self, vocab: list[str]) -> dict[str, float]:
        """Helper: uniform distribution over vocab."""
        p = 1.0 / len(vocab)
        return {kw: p for kw in vocab}

    def test_all_keywords_in_vocab(self) -> None:
        """Every keyword returned must belong to the input distribution."""
        dist = self._uniform_dist(VOCAB)
        queries = draw_queries(dist, lam=100.0, tau=1, rng=make_rng())
        assert all(kw in VOCAB for kw in queries)

    def test_returns_list_of_strings(self) -> None:
        """draw_queries must return a list of str."""
        dist = self._uniform_dist(VOCAB_SMALL)
        queries = draw_queries(dist, lam=20.0, tau=1, rng=make_rng())
        assert isinstance(queries, list)
        assert all(isinstance(q, str) for q in queries)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed must produce an identical query list."""
        dist = self._uniform_dist(VOCAB)
        q_a = draw_queries(dist, lam=50.0, tau=1, rng=make_rng(42))
        q_b = draw_queries(dist, lam=50.0, tau=1, rng=make_rng(42))
        assert q_a == q_b

    def test_different_seeds_differ(self) -> None:
        """Different seeds (virtually always) produce different query lists."""
        dist = self._uniform_dist(VOCAB)
        q_a = draw_queries(dist, lam=200.0, tau=1, rng=make_rng(1))
        q_b = draw_queries(dist, lam=200.0, tau=1, rng=make_rng(2))
        assert q_a != q_b

    def test_count_approximately_poisson(self) -> None:
        """Mean query count over many realisations ≈ lam / tau."""
        dist = self._uniform_dist(VOCAB)
        lam, tau = 30.0, 3
        expected_mean = lam / tau   # = 10
        counts = [
            len(draw_queries(dist, lam=lam, tau=tau,
                             rng=np.random.default_rng(s)))
            for s in range(500)
        ]
        observed_mean = sum(counts) / len(counts)
        # Poisson mean should be within 10 % of expected.
        assert abs(observed_mean - expected_mean) / expected_mean < 0.1, (
            f"Observed mean={observed_mean:.2f}, expected≈{expected_mean}"
        )

    def test_zero_lambda_mostly_empty(self) -> None:
        """With lam=0 every draw must return an empty list (Poisson(0) = 0)."""
        dist = self._uniform_dist(VOCAB)
        for seed in range(20):
            result = draw_queries(dist, lam=0.0, tau=1,
                                  rng=np.random.default_rng(seed))
            assert result == [], f"Expected [], got {result} (seed={seed})"

    def test_query_rates_split_by_tau(self) -> None:
        """Halving tau while doubling lam keeps lam/tau constant; counts match."""
        dist = self._uniform_dist(VOCAB)
        rng_a = make_rng(5)
        rng_b = make_rng(5)
        # Both draw from Poisson(10).
        count_a = len(draw_queries(dist, lam=10.0, tau=1, rng=rng_a))
        count_b = len(draw_queries(dist, lam=20.0, tau=2, rng=rng_b))
        # Same rng state means same Poisson draw.
        assert count_a == count_b

    def test_skewed_dist_biases_queries(self) -> None:
        """A heavily skewed distribution should concentrate queries on one keyword."""
        # Put 99% mass on kw0.
        dist = {kw: (0.99 if kw == "kw0" else 0.01 / 9) for kw in VOCAB}
        queries = draw_queries(dist, lam=1000.0, tau=1, rng=make_rng(0))
        kw0_frac = queries.count("kw0") / len(queries)
        assert kw0_frac > 0.90, (
            f"Expected >90% kw0, got {kw0_frac:.2%}"
        )

    def test_probabilities_reused_sum_to_one_internally(self) -> None:
        """draw_queries must not crash if the input probs don't sum to exactly 1."""
        # Introduce tiny floating-point error.
        dist = {kw: 0.1 + (1e-16 if kw == "kw0" else 0.0) for kw in VOCAB}
        # Should not raise.
        result = draw_queries(dist, lam=50.0, tau=1, rng=make_rng())
        assert all(kw in VOCAB for kw in result)
