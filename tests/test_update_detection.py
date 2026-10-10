"""Tests for the deletion/update-pattern detection experiment."""

import pytest

from attacks.update_detection import detect_deletions, run_gamma_experiment
from equality import EqualityResult
from synthetic import make_synthetic
from types_ import LeakageRecord


def test_detect_deletions_uses_latest_cumulative_lengths_per_group() -> None:
    records = [
        LeakageRecord(0, "first", 0, 0, 2, 2),
        LeakageRecord(1, "latest", 1, 1, 7, 3),
        LeakageRecord(2, "insert-only", 1, 1, 4, 4),
    ]
    equality = EqualityResult(
        groups={0: [0, 1], 2: [2]},
        group_of={0: 0, 1: 0, 2: 2},
    )

    estimates = detect_deletions(records, equality)

    assert (estimates[0].insertions, estimates[0].deletions) == (5, 2)
    assert estimates[0].prediction == "client deletes"
    assert (estimates[2].insertions, estimates[2].deletions) == (4, 0)
    assert estimates[2].prediction == "insert only"


def test_detect_deletions_rejects_invalid_parity() -> None:
    records = [LeakageRecord(0, "bad", 0, 0, 3, 2)]
    equality = EqualityResult(groups={0: [0]}, group_of={0: 0})

    with pytest.raises(ValueError, match="parity"):
        detect_deletions(records, equality)


def test_gamma_experiment_returns_bounded_scores() -> None:
    rows = run_gamma_experiment(
        make_synthetic(seed=3),
        gammas=(0.0, 0.2),
        seed=3,
        tau=2,
        lambda_queries=10,
    )

    assert [row["gamma"] for row in rows] == [0.0, 0.2]
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)
