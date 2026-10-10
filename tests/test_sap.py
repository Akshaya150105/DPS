"""Tests for the simplified auxiliary-distribution frequency attack."""

from __future__ import annotations

import pytest

from attacks import sap
from equality import EqualityResult
from types_ import Aux, LeakageRecord


def _record(query_id: int, interval: int = 0) -> LeakageRecord:
    return LeakageRecord(query_id, f"token-{query_id}", interval, query_id, 0, 0)


def test_recovers_unique_frequency_matches_from_equality_groups() -> None:
    records = [_record(0), _record(1), _record(2)]
    equality = EqualityResult(
        groups={0: [0, 1], 2: [2]},
        group_of={0: 0, 1: 0, 2: 2},
        linked_pairs={0: 1, 1: None, 2: None},
    )
    aux = Aux(query_dist={0: {"common": 2 / 3, "rare": 1 / 3}})

    predictions = sap.run(records, aux, equality)

    assert predictions == {0: "common", 1: "common", 2: "rare"}


def test_uses_frequency_match_across_multiple_intervals() -> None:
    records = [
        _record(0, 0), _record(1, 0), _record(2, 0), _record(3, 0),
        _record(4, 1), _record(5, 1), _record(6, 1), _record(7, 1),
    ]
    equality = EqualityResult(
        groups={0: [0, 1, 2, 4, 5], 3: [3, 6, 7]},
        group_of={0: 0, 1: 0, 2: 0, 4: 0, 5: 0, 3: 3, 6: 3, 7: 3},
        linked_pairs={0: 1, 1: 2, 2: 4, 4: 5, 5: None, 3: 6, 6: 7, 7: None},
    )
    aux = Aux(query_dist={
        0: {"topic": 0.75, "other": 0.25},
        1: {"topic": 0.5, "other": 0.5},
    })

    predictions = sap.run(records, aux, equality)

    assert all(predictions[qid] == "topic" for qid in (0, 1, 2, 4, 5))
    assert all(predictions[qid] == "other" for qid in (3, 6, 7))


def test_ties_and_missing_distribution_abstain() -> None:
    records = [_record(0), _record(1)]
    equality = EqualityResult(
        groups={0: [0], 1: [1]},
        group_of={0: 0, 1: 1},
        linked_pairs={0: None, 1: None},
    )

    tied = sap.run(records, Aux(query_dist={0: {"a": 0.5, "b": 0.5}}), equality)
    missing = sap.run(records, Aux(query_dist=None), equality)

    assert tied == {0: None, 1: None}
    assert missing == {0: None, 1: None}


def test_rejects_invalid_distribution() -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        sap.run([_record(0)], Aux(query_dist={0: {"kw": -0.1}}), EqualityResult(group_of={0: 0}))


def test_empty_records_returns_empty_mapping() -> None:
    assert sap.run([], Aux(query_dist={}), EqualityResult()) == {}
