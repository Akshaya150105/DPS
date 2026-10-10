"""Tests for simulate(dataset, config) in simulator.py."""

import numpy as np
import pytest

from config import Config
from simulator import DynamicDB, simulate
from synthetic import make_synthetic
from types_ import Event, LeakageRecord


@pytest.fixture(scope="module")
def synthetic_docs():
    """Deterministic synthetic dataset of 100 documents."""
    return make_synthetic(seed=42)


class TestSimulateCoreProperties:
    def test_return_tuple_structure(self, synthetic_docs) -> None:
        """simulate must return (records, ground_truth, events) with correct types."""
        cfg = Config(seed=42, tau=3, lambda_queries=20.0, gamma=0.1)
        records, ground_truth, events = simulate(synthetic_docs, cfg)

        assert isinstance(records, list)
        assert isinstance(ground_truth, dict)
        assert isinstance(events, list)

        assert all(isinstance(r, LeakageRecord) for r in records)
        assert all(isinstance(k, int) and isinstance(v, str) for k, v in ground_truth.items())
        assert all(isinstance(e, Event) for e in events)

    def test_ground_truth_is_separate_and_matches_query_ids(self, synthetic_docs) -> None:
        """Ground truth must map each query_id in records to its true keyword."""
        cfg = Config(seed=42, tau=3, lambda_queries=30.0, gamma=0.1)
        records, ground_truth, _ = simulate(synthetic_docs, cfg)

        assert len(records) > 0
        assert len(ground_truth) == len(records)
        record_qids = [r.query_id for r in records]
        assert record_qids == list(range(len(records)))
        assert set(ground_truth.keys()) == set(record_qids)

    def test_rlen_equals_live_matching_documents(self, synthetic_docs) -> None:
        """rlen in each LeakageRecord must equal live matching documents at query time.

        We replay the simulation events up to each query's logical timestamp in a
        DynamicDB and verify that db.current_docs_for(keyword) matches rlen.
        """
        cfg = Config(seed=42, tau=3, lambda_queries=40.0, gamma=0.2)
        records, ground_truth, events = simulate(synthetic_docs, cfg)

        assert len(records) > 0

        # Build map of doc_id -> Document for replay
        doc_map = {d.doc_id: d for d in synthetic_docs}

        # For each distinct interval / timestamp, verify live documents
        for rec in records:
            kw = ground_truth[rec.query_id]

            # Reconstruct live documents from events up to rec.timestamp
            matching_adds = {e.doc_id for e in events if e.keyword == kw and e.mu <= rec.timestamp and e.op == "add"}
            matching_dels = {e.doc_id for e in events if e.keyword == kw and e.mu <= rec.timestamp and e.op == "del"}
            live_doc_count = len(matching_adds - matching_dels)

            assert rec.rlen == live_doc_count, (
                f"Query {rec.query_id} for '{kw}': rlen={rec.rlen} != live={live_doc_count}"
            )

    def test_a_equals_ulen_plus_rlen_div_two_is_always_integer(self, synthetic_docs) -> None:
        """a = (ulen + rlen) / 2 must always be an integer for every record."""
        cfg = Config(seed=42, tau=3, lambda_queries=50.0, gamma=0.2)
        records, ground_truth, events = simulate(synthetic_docs, cfg)

        assert len(records) > 0
        for rec in records:
            sum_un = rec.ulen + rec.rlen
            assert sum_un % 2 == 0, (
                f"Query {rec.query_id}: ulen={rec.ulen} + rlen={rec.rlen} = {sum_un} is not even."
            )
            a = sum_un // 2
            d = (rec.ulen - rec.rlen) // 2
            assert a >= 0
            assert d >= 0
            assert a >= d

    def test_same_seed_gives_identical_output(self, synthetic_docs) -> None:
        """Same seed must produce identical records, ground_truth, and events."""
        cfg1 = Config(seed=123, tau=3, lambda_queries=30.0, gamma=0.15, equality="T1")
        cfg2 = Config(seed=123, tau=3, lambda_queries=30.0, gamma=0.15, equality="T1")

        rec1, gt1, ev1 = simulate(synthetic_docs, cfg1)
        rec2, gt2, ev2 = simulate(synthetic_docs, cfg2)

        assert len(rec1) == len(rec2)
        assert gt1 == gt2
        assert ev1 == ev2

        for r1, r2 in zip(rec1, rec2):
            assert r1.query_id == r2.query_id
            assert r1.token == r2.token
            assert r1.interval == r2.interval
            assert r1.timestamp == r2.timestamp
            assert r1.ulen == r2.ulen
            assert r1.rlen == r2.rlen
            assert r1.fvol == r2.fvol

    def test_different_seeds_differ(self, synthetic_docs) -> None:
        """Different seeds produce different simulation runs."""
        cfg1 = Config(seed=1, tau=3, lambda_queries=30.0, gamma=0.15)
        cfg2 = Config(seed=2, tau=3, lambda_queries=30.0, gamma=0.15)

        rec1, gt1, _ = simulate(synthetic_docs, cfg1)
        rec2, gt2, _ = simulate(synthetic_docs, cfg2)

        # Either count of queries differs or the keywords differ
        assert (len(rec1) != len(rec2)) or (gt1 != gt2)


class TestSimulateModesAndEdgeCases:
    def test_t1_tokens_reveal_prefix_equality(self, synthetic_docs) -> None:
        """In T1 mode, queries for the same keyword have identical prefixes."""
        cfg = Config(seed=42, tau=3, lambda_queries=60.0, gamma=0.1, equality="T1")
        records, ground_truth, _ = simulate(synthetic_docs, cfg)

        kw_prefixes: dict[str, set[str]] = {}
        for rec in records:
            kw = ground_truth[rec.query_id]
            prefix = rec.token.split("|")[0]
            kw_prefixes.setdefault(kw, set()).add(prefix)

        # Each keyword must map to exactly one prefix
        for kw, prefixes in kw_prefixes.items():
            assert len(prefixes) == 1, f"Keyword '{kw}' has multiple prefixes: {prefixes}"

    def test_t2_tokens_are_opaque(self, synthetic_docs) -> None:
        """In T2 mode, tokens do not contain '|'."""
        cfg = Config(seed=42, tau=3, lambda_queries=30.0, gamma=0.1, equality="T2")
        records, _, _ = simulate(synthetic_docs, cfg)

        for rec in records:
            assert "|" not in rec.token

    def test_zero_gamma_means_ulen_equals_rlen(self, synthetic_docs) -> None:
        """With gamma=0.0 (no deletions), ulen must equal rlen for all queries."""
        cfg = Config(seed=42, tau=3, lambda_queries=30.0, gamma=0.0)
        records, _, _ = simulate(synthetic_docs, cfg)

        assert len(records) > 0
        for rec in records:
            assert rec.ulen == rec.rlen, (
                f"Query {rec.query_id}: ulen={rec.ulen} != rlen={rec.rlen} under gamma=0"
            )

    def test_zero_lambda_produces_no_records(self, synthetic_docs) -> None:
        """With lambda_queries=0, no queries are issued; events are still generated."""
        cfg = Config(seed=42, tau=3, lambda_queries=0.0, gamma=0.1)
        records, ground_truth, events = simulate(synthetic_docs, cfg)

        assert records == []
        assert ground_truth == {}
        assert len(events) > 0

    def test_empty_dataset_returns_empty(self) -> None:
        """simulate on empty dataset returns empty records, gt, and events."""
        cfg = Config(seed=42, tau=3, lambda_queries=10.0)
        records, ground_truth, events = simulate([], cfg)

        assert records == []
        assert ground_truth == {}
        assert events == []
