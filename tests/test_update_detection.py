"""Tests for attacks/update_detection.py: update-type detection (deletion vs insert-only)."""

from __future__ import annotations

import pytest

import attacks.update_detection as ud
from config import Config
from equality import build_equality
from simulator import simulate
from synthetic import make_synthetic
from types_ import LeakageRecord


class TestDetectDeletions:
    def test_calculation_of_a_and_d(self) -> None:
        """detect_deletions must correctly compute d = (u - n)/2 and a = (u + n)/2."""
        # Query 0: u=10, n=6 -> a=8, d=2 (>0 -> 'client deletes')
        # Query 1: u=5, n=5  -> a=5, d=0 (==0 -> 'insert only')
        records = [
            LeakageRecord(query_id=0, token="tokA", interval=0, timestamp=1, ulen=10, rlen=6),
            LeakageRecord(query_id=1, token="tokB", interval=0, timestamp=2, ulen=5, rlen=5),
        ]
        equality = {0: [0], 1: [1]}

        res = ud.detect_deletions(records, equality)

        # Group 0
        assert res.predictions[0] == "client deletes"
        assert res.group_estimates[0] == (8.0, 2.0)
        assert res.query_predictions[0] == "client deletes"
        assert res.query_estimates[0] == (8.0, 2.0)

        # Group 1
        assert res.predictions[1] == "insert only"
        assert res.group_estimates[1] == (5.0, 0.0)
        assert res.query_predictions[1] == "insert only"
        assert res.query_estimates[1] == (5.0, 0.0)

    def test_multi_query_group_picks_latest_state(self) -> None:
        """When an equality group has multiple queries, latest state determines group prediction."""
        records = [
            # Earlier query at timestamp 10: no deletions yet (u=4, n=4 -> d=0)
            LeakageRecord(query_id=0, token="tok|1", interval=0, timestamp=10, ulen=4, rlen=4),
            # Later query at timestamp 20: 2 deletions occurred (u=6, n=2 -> d=2)
            LeakageRecord(query_id=1, token="tok|2", interval=1, timestamp=20, ulen=6, rlen=2),
        ]
        equality = {0: [0, 1]}

        res = ud.detect_deletions(records, equality)

        # Group prediction should reflect latest query (d=2.0)
        assert res.predictions[0] == "client deletes"
        assert res.group_estimates[0] == (4.0, 2.0)

        # Query-specific predictions should reflect each query's point in time
        assert res.query_predictions[0] == "insert only"
        assert res.query_predictions[1] == "client deletes"

    def test_unpacking_tuple_support(self) -> None:
        """Result must support tuple unpacking (predictions, group_estimates)."""
        records = [
            LeakageRecord(query_id=0, token="tok", interval=0, timestamp=1, ulen=8, rlen=4)
        ]
        equality = {0: [0]}

        preds, ests = ud.detect_deletions(records, equality)
        assert preds[0] == "client deletes"
        assert ests[0] == (6.0, 2.0)


class TestUpdateDetectionExperiment:
    def test_detection_experiment_gamma_zero_perfect_insert_only(self) -> None:
        """When gamma=0 (no deletions), all groups must be classified as 'insert only' with 100% accuracy."""
        results = ud.run_detection_experiment(
            gammas=[0.0],
            n_seeds=2,
            tau=2,
            lam=50.0,
            print_results=False,
        )
        assert 0.0 in results
        assert results[0.0] == pytest.approx(1.0)

    def test_detection_experiment_across_gammas(self) -> None:
        """Experiment across [0, 0.05, 0.1, 0.2] must achieve high accuracy."""
        results = ud.run_detection_experiment(
            gammas=[0.0, 0.05, 0.1, 0.2],
            n_seeds=3,
            tau=3,
            lam=80.0,
            print_results=False,
        )
        assert len(results) == 4
        for gamma, acc in results.items():
            assert acc >= 0.9, f"Expected acc >= 0.9 for gamma={gamma}, got {acc}"


class TestDocstringDocumentation:
    def test_docstring_describes_paper_extension(self) -> None:
        """Module docstring must describe update-type detection as a paper capability extension."""
        doc = ud.__doc__
        assert doc is not None
        assert "extension" in doc.lower()
        assert "u = a + d" in doc or "d = (u - n) / 2" in doc
