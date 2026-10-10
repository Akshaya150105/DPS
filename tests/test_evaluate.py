"""Tests for evaluate.py: accuracy() and log_result()."""

from __future__ import annotations

import csv
import pathlib

import pytest

from evaluate import _COLUMNS, accuracy, log_result


# ---------------------------------------------------------------------------
# 1. accuracy()
# ---------------------------------------------------------------------------

class TestAccuracy:
    # -- Basic correctness --

    def test_all_correct(self) -> None:
        """Every prediction matches ground truth → 1.0."""
        gt   = {0: "kw0", 1: "kw1", 2: "kw2"}
        pred = {0: "kw0", 1: "kw1", 2: "kw2"}
        assert accuracy(pred, gt) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """No prediction matches → 0.0."""
        gt   = {0: "kw0", 1: "kw1"}
        pred = {0: "kw1", 1: "kw0"}
        assert accuracy(pred, gt) == pytest.approx(0.0)

    def test_half_correct(self) -> None:
        """Exactly half correct → 0.5."""
        gt   = {0: "kw0", 1: "kw1", 2: "kw2", 3: "kw3"}
        pred = {0: "kw0", 1: "kw1", 2: "kw9", 3: "kw9"}
        assert accuracy(pred, gt) == pytest.approx(0.5)

    # -- None handling --

    def test_none_prediction_counts_as_wrong(self) -> None:
        """Explicit None must be counted as incorrect."""
        gt   = {0: "kw0", 1: "kw1"}
        pred = {0: None, 1: "kw1"}
        # 1 correct out of 2
        assert accuracy(pred, gt) == pytest.approx(0.5)

    def test_all_none_predictions_gives_zero(self) -> None:
        gt   = {0: "kw0", 1: "kw1"}
        pred = {0: None, 1: None}
        assert accuracy(pred, gt) == pytest.approx(0.0)

    def test_missing_prediction_counts_as_wrong(self) -> None:
        """A query_id absent from predictions is treated as wrong."""
        gt   = {0: "kw0", 1: "kw1", 2: "kw2"}
        pred = {0: "kw0"}      # 1 and 2 missing
        assert accuracy(pred, gt) == pytest.approx(1 / 3)

    # -- Edge cases --

    def test_empty_ground_truth_returns_one(self) -> None:
        """Vacuous correctness: no queries → 1.0."""
        assert accuracy({}, {}) == pytest.approx(1.0)

    def test_empty_predictions_with_nonempty_gt(self) -> None:
        """All queries unanswered → 0.0."""
        gt = {0: "kw0", 1: "kw1"}
        assert accuracy({}, gt) == pytest.approx(0.0)

    def test_extra_predictions_ignored(self) -> None:
        """Extra query_ids in predictions (not in gt) do not affect the score."""
        gt   = {0: "kw0"}
        pred = {0: "kw0", 99: "kw1", 100: None}
        assert accuracy(pred, gt) == pytest.approx(1.0)

    def test_single_query_correct(self) -> None:
        assert accuracy({7: "kw3"}, {7: "kw3"}) == pytest.approx(1.0)

    def test_single_query_wrong(self) -> None:
        assert accuracy({7: "kw9"}, {7: "kw3"}) == pytest.approx(0.0)

    def test_return_type_is_float(self) -> None:
        result = accuracy({0: "kw0"}, {0: "kw0"})
        assert isinstance(result, float)

    def test_correct_and_total_counts(self) -> None:
        """Verify the arithmetic: 3 correct out of 5."""
        gt   = {i: f"kw{i}" for i in range(5)}
        pred = {0: "kw0", 1: "kw1", 2: "kw2", 3: "wrong", 4: None}
        assert accuracy(pred, gt) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# 2. log_result()
# ---------------------------------------------------------------------------

class TestLogResult:
    # -- Header creation --

    def test_creates_file_if_not_exists(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "results" / "out.csv"
        log_result({"dataset": "synthetic", "accuracy": 0.9}, p)
        assert p.exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "a" / "b" / "c" / "results.csv"
        log_result({}, p)
        assert p.exists()

    def test_header_written_once(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        log_result({"dataset": "d1"}, p)
        log_result({"dataset": "d2"}, p)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(_COLUMNS), "First line must be the header."
        # Only one header row.
        header_count = sum(1 for l in lines if l == ",".join(_COLUMNS))
        assert header_count == 1

    def test_no_double_header_on_existing_file(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        log_result({"seed": 1}, p)
        log_result({"seed": 2}, p)
        log_result({"seed": 3}, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    # -- Column schema --

    def test_fixed_column_order(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        log_result({}, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == _COLUMNS

    def test_all_columns_present_in_each_row(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        row = {
            "dataset": "synthetic", "keyword_count": 10, "attack": "IKK",
            "equality": "T1", "tau": 3, "lambda": 50.0, "gamma": 0.1,
            "alpha": 0.3, "eta": 0.5, "noise": 0.0,
            "correct": 45, "total": 50, "accuracy": 0.9,
            "runtime_s": 1.23, "seed": 42,
        }
        log_result(row, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            written = next(reader)
        for col in _COLUMNS:
            assert col in written

    def test_unknown_keys_silently_ignored(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        # 'extra_col' is not in _COLUMNS.
        log_result({"dataset": "test", "extra_col": "should_be_ignored"}, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        assert "extra_col" not in row

    def test_missing_keys_written_as_empty_string(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        log_result({"dataset": "minimal"}, p)   # all other cols absent
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        assert row["dataset"] == "minimal"
        for col in _COLUMNS:
            if col != "dataset":
                assert row[col] == "", f"Column {col!r} should be empty string."

    # -- Multiple rows --

    def test_multiple_rows_appended_correctly(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        rows = [
            {"dataset": "synthetic", "seed": 1, "accuracy": 0.8},
            {"dataset": "synthetic", "seed": 2, "accuracy": 0.85},
            {"dataset": "synthetic", "seed": 3, "accuracy": 0.9},
        ]
        for r in rows:
            log_result(r, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            written = list(reader)
        assert len(written) == 3
        assert written[0]["seed"] == "1"
        assert written[1]["seed"] == "2"
        assert written[2]["seed"] == "3"

    def test_values_round_trip_correctly(self, tmp_path: pathlib.Path) -> None:
        """Numeric and string values must survive the CSV round-trip."""
        p = tmp_path / "out.csv"
        log_result({
            "dataset": "real", "keyword_count": 500, "tau": 10,
            "gamma": 0.2, "accuracy": 0.75, "seed": 99,
        }, p)
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        assert row["dataset"] == "real"
        assert float(row["accuracy"]) == pytest.approx(0.75)
        assert int(row["seed"]) == 99
        assert int(row["keyword_count"]) == 500

    def test_empty_path_parent_is_cwd(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """log_result must work when parent dir already exists."""
        monkeypatch.chdir(tmp_path)
        log_result({"dataset": "test"}, tmp_path / "simple.csv")
        assert (tmp_path / "simple.csv").exists()

    # -- Integration: accuracy + log_result --

    def test_accuracy_value_logged_correctly(self, tmp_path: pathlib.Path) -> None:
        gt   = {0: "kw0", 1: "kw1", 2: "kw2", 3: "kw3", 4: "kw4"}
        pred = {0: "kw0", 1: "kw1", 2: "kw2", 3: "bad",  4: None}
        acc  = accuracy(pred, gt)
        correct = sum(1 for qid, kw in gt.items() if pred.get(qid) == kw)

        p = tmp_path / "results.csv"
        log_result({
            "dataset": "synthetic", "attack": "test",
            "correct": correct, "total": len(gt), "accuracy": acc,
            "seed": 0,
        }, p)

        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)

        assert float(row["accuracy"]) == pytest.approx(3 / 5)
        assert int(row["correct"]) == 3
        assert int(row["total"]) == 5
