"""Tests for auxiliary knowledge generators in simulator.py."""

from datetime import datetime, timezone

import numpy as np
import pytest

from simulator import (
    build_update_schedule,
    generate_query_distribution,
    make_aux_data_dist,
    make_aux_known_docs,
    make_aux_pre_snapshot,
    make_aux_query_dist,
)
from synthetic import VOCAB, make_synthetic
from types_ import Document, Event


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def corpus() -> list[Document]:
    return make_synthetic(seed=0)


@pytest.fixture(scope="module")
def all_events(corpus) -> list[Event]:
    """All events produced by a full update schedule with gamma=0.2."""
    rng = make_rng(0)
    schedule = build_update_schedule(corpus, tau=3, gamma=0.2, rng=rng)
    return [e for interval in schedule for e in interval]


@pytest.fixture(scope="module")
def true_dist() -> dict[int, dict[str, float]]:
    return generate_query_distribution(VOCAB, tau=3, rng=make_rng(0))


# ---------------------------------------------------------------------------
# 1. make_aux_query_dist
# ---------------------------------------------------------------------------

class TestMakeAuxQueryDist:
    def test_same_keys_as_input(self, true_dist) -> None:
        """Output must have the same interval keys and keyword keys as input."""
        noisy = make_aux_query_dist(true_dist, noise_n=0.0, rng=make_rng())
        assert set(noisy.keys()) == set(true_dist.keys())
        for t in true_dist:
            assert set(noisy[t].keys()) == set(true_dist[t].keys())

    def test_probabilities_sum_to_one_zero_noise(self, true_dist) -> None:
        """With noise_n=0, probabilities must still sum to 1.0."""
        noisy = make_aux_query_dist(true_dist, noise_n=0.0, rng=make_rng())
        for t, d in noisy.items():
            assert abs(sum(d.values()) - 1.0) < 1e-9, (
                f"Interval {t} sums to {sum(d.values())}"
            )

    def test_probabilities_sum_to_one_with_noise(self, true_dist) -> None:
        """With noise_n > 0, probabilities must still sum to 1.0 after renorm."""
        noisy = make_aux_query_dist(true_dist, noise_n=0.5, rng=make_rng())
        for t, d in noisy.items():
            assert abs(sum(d.values()) - 1.0) < 1e-9

    def test_all_probs_positive_after_noise(self, true_dist) -> None:
        """All probabilities must remain positive after adding noise."""
        noisy = make_aux_query_dist(true_dist, noise_n=1.0, rng=make_rng())
        for t, d in noisy.items():
            for kw, p in d.items():
                assert p > 0, f"Interval {t}, keyword {kw}: p={p}"

    def test_zero_noise_preserves_relative_order(self, true_dist) -> None:
        """With noise_n=0 the ranking of keywords must be unchanged."""
        noisy = make_aux_query_dist(true_dist, noise_n=0.0, rng=make_rng())
        for t in true_dist:
            orig_sorted = sorted(true_dist[t], key=true_dist[t].get, reverse=True)
            noisy_sorted = sorted(noisy[t], key=noisy[t].get, reverse=True)
            assert orig_sorted == noisy_sorted, (
                f"Interval {t}: ranking changed under zero noise."
            )

    def test_nonzero_noise_changes_distribution(self, true_dist) -> None:
        """Large noise_n must produce output different from the input."""
        noisy = make_aux_query_dist(true_dist, noise_n=10.0, rng=make_rng())
        for t in true_dist:
            orig_vals = list(true_dist[t].values())
            noisy_vals = [noisy[t][kw] for kw in true_dist[t]]
            assert orig_vals != pytest.approx(noisy_vals), (
                f"Interval {t}: noise had no effect."
            )

    def test_reproducible_same_seed(self, true_dist) -> None:
        """Same seed must produce identical noisy distributions."""
        noisy_a = make_aux_query_dist(true_dist, noise_n=0.3, rng=make_rng(7))
        noisy_b = make_aux_query_dist(true_dist, noise_n=0.3, rng=make_rng(7))
        for t in true_dist:
            for kw in true_dist[t]:
                assert noisy_a[t][kw] == pytest.approx(noisy_b[t][kw])

    def test_different_seeds_differ(self, true_dist) -> None:
        """Different seeds must (virtually always) produce different output."""
        noisy_a = make_aux_query_dist(true_dist, noise_n=0.5, rng=make_rng(1))
        noisy_b = make_aux_query_dist(true_dist, noise_n=0.5, rng=make_rng(2))
        vals_a = [noisy_a[0][kw] for kw in VOCAB]
        vals_b = [noisy_b[0][kw] for kw in VOCAB]
        assert vals_a != pytest.approx(vals_b)

    def test_negative_noise_raises(self, true_dist) -> None:
        """Negative noise_n must raise ValueError."""
        with pytest.raises(ValueError):
            make_aux_query_dist(true_dist, noise_n=-0.1, rng=make_rng())

    def test_single_dict_support(self) -> None:
        """Single keyword distribution (non-interval) should also be supported."""
        single_dist = {"k1": 0.7, "k2": 0.3}
        noisy = make_aux_query_dist(single_dist, noise_n=0.5, rng=make_rng(42))
        assert set(noisy.keys()) == {"k1", "k2"}
        assert abs(sum(noisy.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. make_aux_known_docs
# ---------------------------------------------------------------------------

class TestMakeAuxKnownDocs:
    def test_returns_list_of_int(self, all_events) -> None:
        """Return value must be a list of integers."""
        result = make_aux_known_docs(all_events, alpha=0.3, rng=make_rng())
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_all_ids_were_inserted(self, all_events) -> None:
        """Every returned doc_id must appear in an 'add' event."""
        inserted = {e.doc_id for e in all_events if e.op == "add"}
        result = make_aux_known_docs(all_events, alpha=0.3, rng=make_rng())
        assert set(result) <= inserted, (
            f"Unknown doc_ids returned: {set(result) - inserted}"
        )

    def test_no_duplicates(self, all_events) -> None:
        """Returned list must not contain duplicate doc_ids."""
        result = make_aux_known_docs(all_events, alpha=0.5, rng=make_rng())
        assert len(result) == len(set(result))

    def test_fraction_close_to_alpha(self, all_events) -> None:
        """Sample size must be round(alpha * n_inserted)."""
        n_inserted = len({e.doc_id for e in all_events if e.op == "add"})
        for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
            result = make_aux_known_docs(all_events, alpha=alpha, rng=make_rng())
            expected = round(alpha * n_inserted)
            assert abs(len(result) - expected) <= 1, (
                f"alpha={alpha}: expected ~{expected} docs, got {len(result)}"
            )

    def test_alpha_zero_raises(self, all_events) -> None:
        """alpha=0 must raise ValueError (open interval)."""
        with pytest.raises(ValueError, match="open interval"):
            make_aux_known_docs(all_events, alpha=0.0, rng=make_rng())

    def test_alpha_one_raises(self, all_events) -> None:
        """alpha=1 must raise ValueError (open interval)."""
        with pytest.raises(ValueError, match="open interval"):
            make_aux_known_docs(all_events, alpha=1.0, rng=make_rng())

    def test_alpha_negative_raises(self, all_events) -> None:
        """Negative alpha must raise ValueError."""
        with pytest.raises(ValueError):
            make_aux_known_docs(all_events, alpha=-0.1, rng=make_rng())

    def test_reproducible_same_seed(self, all_events) -> None:
        """Same seed must yield identical doc_id lists."""
        r_a = make_aux_known_docs(all_events, alpha=0.4, rng=make_rng(5))
        r_b = make_aux_known_docs(all_events, alpha=0.4, rng=make_rng(5))
        assert r_a == r_b

    def test_different_seeds_differ(self, all_events) -> None:
        """Different seeds must yield different samples."""
        r_a = make_aux_known_docs(all_events, alpha=0.5, rng=make_rng(1))
        r_b = make_aux_known_docs(all_events, alpha=0.5, rng=make_rng(2))
        assert r_a != r_b

    def test_empty_events_returns_empty(self) -> None:
        """Empty event log must return an empty list."""
        result = make_aux_known_docs([], alpha=0.5, rng=make_rng())
        assert result == []


# ---------------------------------------------------------------------------
# 3. make_aux_data_dist
# ---------------------------------------------------------------------------

class TestMakeAuxDataDist:
    def test_returns_tau_intervals(self, corpus) -> None:
        """Result must have exactly tau interval keys."""
        result = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng())
        assert set(result.keys()) == {0, 1, 2}

    def test_counts_are_nonneg_integers(self, corpus) -> None:
        """All count values must be non-negative integers."""
        result = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng())
        for t, kw_counts in result.items():
            for kw, cnt in kw_counts.items():
                assert isinstance(cnt, int) and cnt >= 0, (
                    f"Interval {t}, keyword {kw}: count={cnt}"
                )

    def test_all_keywords_in_vocab(self, corpus) -> None:
        """Any keyword appearing in the counts must belong to VOCAB."""
        result = make_aux_data_dist(corpus, tau=3, eta=1.0, rng=make_rng())
        vocab_set = set(VOCAB)
        for t, kw_counts in result.items():
            assert set(kw_counts.keys()) <= vocab_set, (
                f"Interval {t} has out-of-vocab keywords: "
                f"{set(kw_counts.keys()) - vocab_set}"
            )

    def test_eta_one_uses_all_docs(self, corpus) -> None:
        """eta=1 must use the full dataset; total keyword count matches corpus."""
        result = make_aux_data_dist(corpus, tau=1, eta=1.0, rng=make_rng())
        total_counted = sum(result[0].values())
        total_expected = sum(len(d.keywords) for d in corpus)
        assert total_counted == total_expected

    def test_smaller_eta_gives_fewer_counts(self, corpus) -> None:
        """eta=0.2 should produce a smaller total count than eta=0.8."""
        rng_lo = make_rng(0)
        rng_hi = make_rng(0)
        lo = make_aux_data_dist(corpus, tau=1, eta=0.2, rng=rng_lo)
        hi = make_aux_data_dist(corpus, tau=1, eta=0.8, rng=rng_hi)
        assert sum(lo[0].values()) < sum(hi[0].values())

    def test_eta_out_of_range_raises(self, corpus) -> None:
        """eta=0 or eta>1 must raise ValueError."""
        with pytest.raises(ValueError):
            make_aux_data_dist(corpus, tau=3, eta=0.0, rng=make_rng())
        with pytest.raises(ValueError):
            make_aux_data_dist(corpus, tau=3, eta=1.5, rng=make_rng())

    def test_reproducible_same_seed(self, corpus) -> None:
        """Same seed must produce identical distributions."""
        r_a = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng(9))
        r_b = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng(9))
        assert r_a == r_b

    def test_different_seeds_differ(self, corpus) -> None:
        """Different seeds should produce different samples."""
        r_a = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng(1))
        r_b = make_aux_data_dist(corpus, tau=3, eta=0.5, rng=make_rng(2))
        assert r_a != r_b

    @pytest.mark.parametrize("tau", [1, 2, 5])
    def test_various_tau(self, corpus, tau: int) -> None:
        """Function must work for any tau >= 1."""
        result = make_aux_data_dist(corpus, tau=tau, eta=0.5, rng=make_rng())
        assert len(result) == tau


# ---------------------------------------------------------------------------
# 4. make_aux_pre_snapshot
# ---------------------------------------------------------------------------

class TestMakeAuxPreSnapshot:
    def test_mu0_zero_returns_empty(self, corpus) -> None:
        """mu0=0 must return an empty list."""
        result = make_aux_pre_snapshot(corpus, mu0=0)
        assert result == []

    def test_mu0_negative_returns_empty(self, corpus) -> None:
        """mu0 < 0 must return an empty list."""
        result = make_aux_pre_snapshot(corpus, mu0=-5)
        assert result == []

    def test_mu0_larger_than_dataset_returns_all(self, corpus) -> None:
        """mu0 >= len(dataset) must return all documents."""
        result = make_aux_pre_snapshot(corpus, mu0=len(corpus) + 100)
        assert len(result) == len(corpus)

    def test_returns_exact_count(self, corpus) -> None:
        """Return value must contain exactly min(mu0, len(dataset)) documents."""
        for mu0 in (1, 10, 33, 50, 99):
            result = make_aux_pre_snapshot(corpus, mu0=mu0)
            assert len(result) == min(mu0, len(corpus)), (
                f"mu0={mu0}: expected {min(mu0, len(corpus))} docs, got {len(result)}"
            )

    def test_returns_earliest_by_timestamp(self, corpus) -> None:
        """Returned documents must be the ones with the smallest timestamps."""
        mu0 = 20
        result = make_aux_pre_snapshot(corpus, mu0=mu0)
        sorted_corpus = sorted(corpus, key=lambda d: d.timestamp)
        expected_ids = {d.doc_id for d in sorted_corpus[:mu0]}
        actual_ids = {d.doc_id for d in result}
        assert actual_ids == expected_ids

    def test_result_is_sorted_by_timestamp(self, corpus) -> None:
        """Returned list must be ordered by ascending timestamp."""
        result = make_aux_pre_snapshot(corpus, mu0=50)
        timestamps = [d.timestamp for d in result]
        assert timestamps == sorted(timestamps)

    def test_result_is_list_of_documents(self, corpus) -> None:
        """Return value must be a list of Document instances."""
        result = make_aux_pre_snapshot(corpus, mu0=10)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_deterministic_no_rng(self, corpus) -> None:
        """make_aux_pre_snapshot takes no rng; two calls must be identical."""
        r1 = make_aux_pre_snapshot(corpus, mu0=30)
        r2 = make_aux_pre_snapshot(corpus, mu0=30)
        assert r1 == r2
