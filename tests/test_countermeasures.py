"""Tests for padding and equality-hiding countermeasures."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pytest

from attacks import via
from countermeasures import (
    file_volume_padding,
    overhead_report,
    oram_equality_hiding,
    paper_section_8_estimate,
    volume_hiding_padding,
)
from equality import EqualityResult, build_equality
from types_ import Aux, Document, Event, LeakageRecord


def _document(doc_id: int, keywords: set[str]) -> Document:
    return Document(
        doc_id=doc_id,
        keywords=frozenset(keywords),
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestVolumeHidingPadding:
    def test_rounds_each_leakage_length_up_without_mutating_input(self) -> None:
        record = LeakageRecord(
            query_id=1,
            token="original-token",
            interval=0,
            timestamp=5,
            ulen=5,
            rlen=7,
            fvol=Counter({2: 3}),
        )

        padded = volume_hiding_padding([record], bucket=4)

        assert (padded[0].ulen, padded[0].rlen) == (8, 8)
        assert padded[0].token == record.token
        assert padded[0].fvol == record.fvol
        assert padded[0] is not record
        assert (record.ulen, record.rlen) == (5, 7)
        assert padded[0].fvol is not record.fvol

    @pytest.mark.parametrize("bucket", [0, -1, True, 1.5])
    def test_rejects_invalid_bucket(self, bucket: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            volume_hiding_padding([], bucket)  # type: ignore[arg-type]

    def test_padding_reduces_via_accuracy_on_explicit_leakage_fixture(self) -> None:
        records = [
            LeakageRecord(0, "q0", 0, 1, 0, 0, Counter()),
            LeakageRecord(1, "q1", 1, 10, 1, 1, Counter({1: 1})),
        ]
        equality = EqualityResult(
            groups={0: [0, 1]},
            group_of={0: 0, 1: 0},
            linked_pairs={0: 1, 1: None},
        )
        kw_a = _document(1, {"kwA"})
        kw_b = _document(2, {"kwB"})
        events = [
            Event(mu=0, op="add", keyword="kwB", doc_id=2),
            Event(mu=5, op="add", keyword="kwA", doc_id=1),
        ]
        aux = Aux(known_docs=[kw_a, kw_b])

        baseline = via.run(records, aux, equality, events)
        padded_records = volume_hiding_padding(records, bucket=4)
        padded = via.run(padded_records, aux, equality, events)

        assert baseline[0] == "kwA"
        assert padded[0] is None


class TestFileVolumePadding:
    def test_makes_all_document_volumes_equal_and_preserves_original_keywords(self) -> None:
        dataset = [
            _document(1, {"a", "b", "c"}),
            _document(2, {"d"}),
            _document(3, {"e", "f"}),
        ]

        padded = file_volume_padding(dataset)

        assert {len(doc.keywords) for doc in padded} == {3}
        for original, result in zip(dataset, padded):
            assert original.keywords <= result.keywords
            assert original.doc_id == result.doc_id
            assert original.timestamp == result.timestamp
        assert [len(doc.keywords) for doc in dataset] == [3, 1, 2]

    def test_avoids_collisions_with_existing_padding_namespace(self) -> None:
        dataset = [
            _document(1, {"__dps_volume_padding__1_0__"}),
            _document(2, {"real", "keyword"}),
        ]

        padded = file_volume_padding(dataset)

        assert len(padded[0].keywords) == len(padded[1].keywords) == 2
        assert "__dps_volume_padding__1_0__" in padded[0].keywords

    def test_empty_dataset_is_valid(self) -> None:
        assert file_volume_padding([]) == []

    def test_overhead_report_estimates_posting_entries(self) -> None:
        dataset = [_document(1, {"a", "b"}), _document(2, {"c"})]

        report = overhead_report(dataset)

        assert report == {
            "document_count": 2,
            "original_posting_entries": 3,
            "padded_posting_entries": 4,
            "padding_entries": 1,
            "overhead_ratio": pytest.approx(4 / 3),
            "overhead_percent": pytest.approx(100 / 3),
            "target_volume": 2,
        }

    def test_reports_paper_section_8_worst_case_estimate(self) -> None:
        estimate = paper_section_8_estimate()

        assert estimate["email_count"] == 30_204
        assert estimate["keyword_count"] == 73_991
        assert estimate["paper_reported_overhead_factor"] == pytest.approx(107.0)
        assert estimate["paper_reported_overhead"] == "about 107x storage"
        assert estimate["worst_case_posting_entries"] == 2_234_824_164

    def test_does_not_apply_paper_ratio_to_custom_corpus_sizes(self) -> None:
        estimate = paper_section_8_estimate(email_count=10, keyword_count=20)

        assert estimate["paper_reported_overhead_factor"] is None
        assert estimate["paper_reported_overhead"] is None
        assert estimate["worst_case_posting_entries"] == 200


class TestOramEqualityHiding:
    def test_replaces_every_token_with_independent_seeded_values(self) -> None:
        records = [
            LeakageRecord(0, "secret-keyword-a|0", 0, 1, 2, 2, Counter({1: 2})),
            LeakageRecord(1, "secret-keyword-a|1", 0, 2, 3, 3, Counter({1: 3})),
            LeakageRecord(2, "secret-keyword-b|0", 0, 3, 1, 1, Counter({2: 1})),
        ]

        first = oram_equality_hiding(records, seed=19)
        repeated = oram_equality_hiding(records, seed=19)

        assert [record.token for record in first] == [record.token for record in repeated]
        assert len({record.token for record in first}) == len(records)
        assert all(
            hidden.token != original.token
            and "secret-keyword" not in hidden.token
            for hidden, original in zip(first, records)
        )
        assert build_equality(first, "T1").groups == {record.query_id: [record.query_id] for record in records}
        assert build_equality(first, "T2", delta=0.5).groups[0] == [0, 1]
        assert records[0].token == "secret-keyword-a|0"

    def test_unseeded_token_values_are_fresh(self) -> None:
        records = [LeakageRecord(0, "secret", 0, 0, 0, 0), LeakageRecord(1, "secret", 0, 1, 0, 0)]

        assert oram_equality_hiding(records) != oram_equality_hiding(records)
