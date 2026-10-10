"""Tests for build_update_schedule in simulator.py."""

import numpy as np
import pytest

from simulator import DynamicDB, build_update_schedule
from synthetic import make_synthetic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded numpy Generator."""
    return np.random.default_rng(seed)


def run_schedule(seed: int = 42, gamma: float = 0.0, tau: int = 3):
    """Helper: build a schedule from the canonical synthetic corpus."""
    docs = make_synthetic(seed=0)   # fixed corpus, vary rng seed separately
    rng = make_rng(seed)
    return build_update_schedule(docs, tau=tau, gamma=gamma, rng=rng)


def count_ops(schedule: list, op: str) -> int:
    """Count total events of a given op ('add' or 'del') across all intervals."""
    return sum(1 for interval in schedule for e in interval if e.op == op)


# ---------------------------------------------------------------------------
# 1. Return-type structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_tau_sublists(self) -> None:
        """build_update_schedule must return exactly tau sublists."""
        schedule = run_schedule(tau=3)
        assert len(schedule) == 3

    def test_sublists_contain_events(self) -> None:
        """Each sublist must be a list (possibly empty) of Event objects."""
        from types_ import Event
        schedule = run_schedule(tau=3)
        for interval_events in schedule:
            assert isinstance(interval_events, list)
            for e in interval_events:
                assert isinstance(e, Event)

    def test_total_adds_equals_dataset_size(self) -> None:
        """Total 'add' events must equal the number of documents in the dataset
        multiplied by the average number of keywords per document
        (one event per keyword per add call)."""
        docs = make_synthetic(seed=0)
        rng = make_rng(42)
        schedule = build_update_schedule(docs, tau=3, gamma=0.0, rng=rng)
        # Each add(doc, mu) appends len(doc.keywords) events.
        expected_adds = sum(len(d.keywords) for d in docs)
        actual_adds = count_ops(schedule, "add")
        assert actual_adds == expected_adds

    def test_mu_values_are_globally_monotonic(self) -> None:
        """Event mu values must be non-decreasing across the whole schedule."""
        schedule = run_schedule(gamma=0.2)
        all_mus = [e.mu for interval in schedule for e in interval]
        assert all_mus == sorted(all_mus), "mu values are not monotonically non-decreasing."


# ---------------------------------------------------------------------------
# 2. gamma = 0 → no deletions
# ---------------------------------------------------------------------------

class TestNoDeletions:
    def test_zero_gamma_no_del_events(self) -> None:
        """With gamma=0, there must be zero 'del' events in the entire schedule."""
        schedule = run_schedule(gamma=0.0)
        assert count_ops(schedule, "del") == 0

    def test_zero_gamma_all_adds(self) -> None:
        """With gamma=0, every event must be an 'add'."""
        schedule = run_schedule(gamma=0.0)
        for interval in schedule:
            for e in interval:
                assert e.op == "add", f"Unexpected del event: {e}"


# ---------------------------------------------------------------------------
# 3. gamma = 0.2 → approximately 20 % deletions
# ---------------------------------------------------------------------------

class TestApproxDeletionFraction:
    """The deletion fraction (del_calls / total_doc_count) should be ≈ gamma."""

    @pytest.mark.parametrize("gamma", [0.1, 0.2, 0.3, 0.5])
    def test_deletion_fraction_close_to_gamma(self, gamma: float) -> None:
        """Deletion fraction must be within ±8 percentage points of gamma."""
        docs = make_synthetic(seed=0)
        rng = make_rng(7)
        schedule = build_update_schedule(docs, tau=3, gamma=gamma, rng=rng)

        # Count unique doc_ids that appear in a 'del' event.
        deleted_doc_ids = {
            e.doc_id for interval in schedule for e in interval if e.op == "del"
        }
        fraction = len(deleted_doc_ids) / len(docs)
        assert abs(fraction - gamma) <= 0.08, (
            f"gamma={gamma}: deletion fraction={fraction:.3f}, "
            f"expected ≈{gamma} (±0.08)"
        )

    def test_gamma_02_deletes_roughly_20_percent(self) -> None:
        """Specific check: gamma=0.2 on 100-doc corpus → 15–25 deletions."""
        docs = make_synthetic(seed=0)
        assert len(docs) == 100
        rng = make_rng(42)
        schedule = build_update_schedule(docs, tau=3, gamma=0.2, rng=rng)
        n_del_calls = count_ops(schedule, "del")
        # Each delete(doc, mu) appends len(doc.keywords) del events.
        # Count unique deleted doc_ids instead.
        deleted_ids = {
            e.doc_id for interval in schedule for e in interval if e.op == "del"
        }
        assert 15 <= len(deleted_ids) <= 25, (
            f"Expected 15–25 deleted docs for gamma=0.2, got {len(deleted_ids)}"
        )

    def test_no_doc_deleted_before_being_inserted(self) -> None:
        """A 'del' event for a doc must always be preceded by an 'add' event."""
        docs = make_synthetic(seed=0)
        rng = make_rng(42)
        schedule = build_update_schedule(docs, tau=3, gamma=0.2, rng=rng)
        seen_adds: set[int] = set()
        for interval in schedule:
            for e in interval:
                if e.op == "add":
                    seen_adds.add(e.doc_id)
                else:  # del
                    assert e.doc_id in seen_adds, (
                        f"doc_id={e.doc_id} was deleted before being added."
                    )

    def test_no_doc_deleted_twice(self) -> None:
        """Each doc_id must be the subject of at most one delete *call*
        (a single delete call emits one 'del' event per keyword, so the same
        doc_id legitimately appears in multiple del events for different keywords)."""
        docs = make_synthetic(seed=0)
        rng = make_rng(42)
        schedule = build_update_schedule(docs, tau=3, gamma=0.2, rng=rng)

        # Collect the (interval_index, doc_id) pairs where a deletion occurred.
        # A doc may appear in del events for multiple keywords within the same
        # interval — that is correct.  It must NOT appear in del events in two
        # different intervals.
        deleted_in_interval: dict[int, int] = {}   # doc_id -> first interval index
        for t, interval in enumerate(schedule):
            for e in interval:
                if e.op == "del":
                    if e.doc_id in deleted_in_interval:
                        assert deleted_in_interval[e.doc_id] == t, (
                            f"doc_id={e.doc_id} was deleted in interval "
                            f"{deleted_in_interval[e.doc_id]} and again in interval {t}."
                        )
                    else:
                        deleted_in_interval[e.doc_id] = t


# ---------------------------------------------------------------------------
# 4. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_schedule(self) -> None:
        """Two calls with the same rng seed must produce identical schedules."""
        docs = make_synthetic(seed=0)
        sched_a = build_update_schedule(docs, tau=3, gamma=0.2, rng=make_rng(7))
        sched_b = build_update_schedule(docs, tau=3, gamma=0.2, rng=make_rng(7))

        for ia, ib in zip(sched_a, sched_b):
            assert len(ia) == len(ib)
            for ea, eb in zip(ia, ib):
                assert ea == eb, f"Events differ: {ea} vs {eb}"

    def test_different_seeds_differ(self) -> None:
        """Different rng seeds should (almost certainly) yield different deletions."""
        docs = make_synthetic(seed=0)
        sched_a = build_update_schedule(docs, tau=3, gamma=0.2, rng=make_rng(1))
        sched_b = build_update_schedule(docs, tau=3, gamma=0.2, rng=make_rng(2))

        del_ids_a = {e.doc_id for iv in sched_a for e in iv if e.op == "del"}
        del_ids_b = {e.doc_id for iv in sched_b for e in iv if e.op == "del"}
        assert del_ids_a != del_ids_b, (
            "Different seeds produced identical deletion sets (extremely unlikely)."
        )


# ---------------------------------------------------------------------------
# 5. Interval partitioning
# ---------------------------------------------------------------------------

class TestIntervalPartitioning:
    def test_docs_assigned_to_intervals_by_timestamp(self) -> None:
        """Documents in later intervals must have timestamps >= those in earlier intervals."""
        docs = make_synthetic(seed=0)
        rng = make_rng(0)
        schedule = build_update_schedule(docs, tau=3, gamma=0.0, rng=rng)

        # Collect the earliest add timestamp (by mu) per interval.
        interval_min_mu = []
        for interval in schedule:
            add_mus = [e.mu for e in interval if e.op == "add"]
            if add_mus:
                interval_min_mu.append(min(add_mus))

        # Min mu in each interval should be non-decreasing.
        assert interval_min_mu == sorted(interval_min_mu), (
            "Interval mu ranges overlap, suggesting incorrect partitioning."
        )

    def test_all_docs_inserted_exactly_once(self) -> None:
        """Every document in the dataset must appear in exactly one 'add' group."""
        docs = make_synthetic(seed=0)
        rng = make_rng(0)
        schedule = build_update_schedule(docs, tau=3, gamma=0.0, rng=rng)

        added_ids = [e.doc_id for iv in schedule for e in iv if e.op == "add"]
        # Each add generates len(keywords) events; doc_ids should repeat = len(keywords) times.
        # Unique doc_ids must equal the dataset.
        assert set(added_ids) == {d.doc_id for d in docs}

    @pytest.mark.parametrize("tau", [1, 2, 5])
    def test_schedule_length_equals_tau(self, tau: int) -> None:
        """build_update_schedule must return exactly tau sublists for any tau."""
        docs = make_synthetic(seed=0)
        rng = make_rng(42)
        schedule = build_update_schedule(docs, tau=tau, gamma=0.0, rng=rng)
        assert len(schedule) == tau
