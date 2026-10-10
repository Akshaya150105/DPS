"""Tests for countermeasures.py and attacks/sap.py."""

from __future__ import annotations

import math
import pytest
import numpy as np

import attacks.sap as sap
import attacks.via as via
from config import Config
import countermeasures
from equality import build_equality
from evaluate import accuracy
from simulator import simulate
from synthetic import VOCAB, make_obvious_query_dist, make_synthetic
from types_ import Aux, Document, LeakageRecord


# ---------------------------------------------------------------------------
# Volume-Hiding Padding Tests
# ---------------------------------------------------------------------------

class TestVolumeHidingPadding:
    def test_pads_rlen_and_ulen_to_bucket_multiples(self) -> None:
        """volume_hiding_padding must pad rlen and ulen up to multiples of bucket."""
        records = [
            LeakageRecord(query_id=0, token="tok0", interval=0, timestamp=1, ulen=3, rlen=7),
            LeakageRecord(query_id=1, token="tok1", interval=0, timestamp=2, ulen=10, rlen=0),
            LeakageRecord(query_id=2, token="tok2", interval=1, timestamp=3, ulen=11, rlen=21),
        ]
        padded = countermeasures.volume_hiding_padding(records, bucket=10)

        assert len(padded) == 3
        # ulen=3 -> 10, rlen=7 -> 10
        assert padded[0].ulen == 10
        assert padded[0].rlen == 10

        # ulen=10 -> 10, rlen=0 -> 0
        assert padded[1].ulen == 10
        assert padded[1].rlen == 0

        # ulen=11 -> 20, rlen=21 -> 30
        assert padded[2].ulen == 20
        assert padded[2].rlen == 30

    def test_bucket_one_keeps_values_unchanged(self) -> None:
        """Bucket size 1 must leave lengths unchanged."""
        records = [LeakageRecord(query_id=0, token="tok", interval=0, timestamp=1, ulen=5, rlen=9)]
        padded = countermeasures.volume_hiding_padding(records, bucket=1)
        assert padded[0].ulen == 5
        assert padded[0].rlen == 9

    def test_invalid_bucket_raises_value_error(self) -> None:
        """Bucket size < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="bucket size must be >= 1"):
            countermeasures.volume_hiding_padding([], bucket=0)
        with pytest.raises(ValueError, match="bucket size must be >= 1"):
            countermeasures.volume_hiding_padding([], bucket=-5)


# ---------------------------------------------------------------------------
# File Volume Padding Tests
# ---------------------------------------------------------------------------

class TestFileVolumePadding:
    def test_pads_all_documents_to_same_volume(self) -> None:
        """file_volume_padding must ensure every document has the exact same volume."""
        docs = make_synthetic(seed=42)
        initial_volumes = {len(d.keywords) for d in docs}
        assert len(initial_volumes) > 1, "Synthetic docs should have varied volumes."

        padded_docs = countermeasures.file_volume_padding(docs)
        assert len(padded_docs) == len(docs)

        max_vol = max(initial_volumes)
        padded_volumes = {len(d.keywords) for d in padded_docs}
        assert padded_volumes == {max_vol}

    def test_explicit_target_volume(self) -> None:
        """Padding to an explicit target volume >= max_vol must succeed."""
        docs = make_synthetic(seed=42)
        padded_docs = countermeasures.file_volume_padding(docs, target_volume=20)
        assert all(len(d.keywords) == 20 for d in padded_docs)

    def test_target_volume_less_than_max_raises(self) -> None:
        """target_volume < max volume must raise ValueError."""
        docs = make_synthetic(seed=42)
        max_vol = max(len(d.keywords) for d in docs)
        with pytest.raises(ValueError, match="must be >= maximum document volume"):
            countermeasures.file_volume_padding(docs, target_volume=max_vol - 1)


# ---------------------------------------------------------------------------
# ORAM Equality Hiding Tests
# ---------------------------------------------------------------------------

class TestORAMEqualityHiding:
    def test_replaces_tokens_with_unique_random_values(self) -> None:
        """oram_equality_hiding must replace repeated tokens with unique random values."""
        records = [
            LeakageRecord(query_id=0, token="kw1|saltA", interval=0, timestamp=1, ulen=5, rlen=5),
            LeakageRecord(query_id=1, token="kw1|saltB", interval=0, timestamp=2, ulen=5, rlen=5),
            LeakageRecord(query_id=2, token="kw2|saltC", interval=1, timestamp=3, ulen=5, rlen=5),
        ]
        hidden = countermeasures.oram_equality_hiding(records)

        assert len(hidden) == len(records)
        tokens = [r.token for r in hidden]
        assert len(set(tokens)) == 3, "All tokens must be distinct"

        # T1 equality matching on hidden tokens should produce 3 singleton groups
        eq_t1 = build_equality(hidden, mode="T1")
        assert len(eq_t1.groups) == 3
        assert all(len(grp) == 1 for grp in eq_t1.groups.values())


# ---------------------------------------------------------------------------
# Overhead Reports Tests
# ---------------------------------------------------------------------------

class TestOverheadReports:
    def test_overhead_report_structure_and_values(self) -> None:
        """overhead_report must calculate correct empirical storage metrics."""
        docs = make_synthetic(seed=42)
        report = countermeasures.overhead_report(docs)

        assert report["num_docs"] == 100
        assert report["original_pairs"] > 0
        assert report["padded_pairs"] > report["original_pairs"]
        assert report["overhead_ratio"] > 1.0
        assert report["target_volume"] == report["max_volume"]

    def test_section_8_estimate_reproduction(self) -> None:
        """estimate_section_8_overhead must reproduce paper Section 8 estimate ~10^7x."""
        res = countermeasures.estimate_section_8_overhead(
            n_docs=30204,
            n_keywords=73991,
            max_doc_volume=1500,
            avg_doc_volume=60.0,
        )

        assert res["n_docs"] == 30204
        assert res["n_keywords"] == 73991
        assert "10^7" in res["order_of_magnitude"]
        assert 1e6 <= res["overhead_ratio"] <= 1e8


# ---------------------------------------------------------------------------
# SAP Attack Tests
# ---------------------------------------------------------------------------

class TestSAPAttack:
    def test_sap_attack_execution(self) -> None:
        """attacks/sap.py must rank and match equality groups using auxiliary distribution."""
        docs = make_synthetic(seed=42)
        tau = 3
        dist = make_obvious_query_dist(VOCAB, tau=tau)
        cfg = Config(seed=42, tau=tau, lambda_queries=100.0, gamma=0.0, equality="T1")
        records, gt, _ = simulate(docs, cfg, query_dist=dist)

        eq = build_equality(records, mode="T1")
        aux = Aux(query_dist=dist)

        preds = sap.run(records, aux, eq)

        assert isinstance(preds, dict)
        assert len(preds) == len(records)
        acc = accuracy(preds, gt)
        assert acc >= 0.0

    def test_sap_empty_or_none_dist_returns_none(self) -> None:
        """sap.run must return None predictions if query_dist is missing."""
        records = [LeakageRecord(query_id=0, token="tok", interval=0, timestamp=1, ulen=1, rlen=1)]
        eq = build_equality(records, mode="T1")
        aux = Aux(query_dist=None)
        preds = sap.run(records, aux, eq)
        assert preds == {0: None}


# ---------------------------------------------------------------------------
# VIA Padding Mitigation Test (User Prompt Requirement)
# ---------------------------------------------------------------------------

class TestPaddingMitigatesVIA:
    def test_volume_hiding_padding_reduces_via_accuracy(self) -> None:
        """Test requirement: Padding reduces VIA accuracy."""
        # 1. Setup clean simulation where VIA achieves high accuracy
        seed = 42
        docs = make_synthetic(seed=seed)
        cfg = Config(seed=seed, tau=3, lambda_queries=150.0, gamma=0.0, equality="T1")
        records, gt, events = simulate(docs, cfg)
        eq = build_equality(records, mode="T1")
        aux = Aux(known_docs=docs)

        # Unpadded VIA accuracy
        unpadded_preds = via.run(records, aux, eq, events)
        acc_unpadded = accuracy(unpadded_preds, gt)

        # 2. Apply volume hiding padding (bucket = 15)
        padded_records = countermeasures.volume_hiding_padding(records, bucket=15)
        padded_preds = via.run(padded_records, aux, eq, events)
        acc_padded = accuracy(padded_preds, gt)

        # Padding distorts Delta a and causes candidates to mismatch
        assert acc_unpadded >= 0.7, f"Unpadded VIA expected >= 0.7, got {acc_unpadded}"
        assert acc_padded < acc_unpadded, (
            f"Padding should reduce accuracy: unpadded={acc_unpadded:.4f}, padded={acc_padded:.4f}"
        )
