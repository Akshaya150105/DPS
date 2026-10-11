"""Tests for Google Trends fetching and query distribution generation.

Tests Patch 6b:
1. Hand-made trends.csv with columns month, keyword, prob read by
   generate_query_distribution(mode='trends') and mapped to intervals.
2. Missing file fallback to synthetic mode with clearly logged warning.
3. scripts/fetch_trends.py: fetching, retrying, caching, and normalization.
"""

from __future__ import annotations

import csv
import logging
import pathlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.fetch_trends import (
    fetch_batch_with_retry,
    fetch_trends,
    format_timeframe,
    load_vocab_file,
)
from simulator import generate_query_distribution


# ---------------------------------------------------------------------------
# 1. generate_query_distribution with small hand-made CSV
# ---------------------------------------------------------------------------

class TestHandMadeTrendsCSV:
    """Test reading and interval mapping from a small hand-made trends.csv."""

    def test_trends_handmade_csv_equal_months_and_tau(self, tmp_path: pathlib.Path) -> None:
        """Hand-made CSV with 2 months mapped to tau=2 intervals."""
        csv_file = tmp_path / "trends.csv"
        vocab = ["apple", "banana", "cherry"]

        # Month 1: apple=0.5, banana=0.3, cherry=0.2
        # Month 2: apple=0.1, banana=0.2, cherry=0.7
        rows = [
            {"month": "2020-01", "keyword": "apple", "prob": 0.5},
            {"month": "2020-01", "keyword": "banana", "prob": 0.3},
            {"month": "2020-01", "keyword": "cherry", "prob": 0.2},
            {"month": "2020-02", "keyword": "apple", "prob": 0.1},
            {"month": "2020-02", "keyword": "banana", "prob": 0.2},
            {"month": "2020-02", "keyword": "cherry", "prob": 0.7},
        ]

        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["month", "keyword", "prob"])
            writer.writeheader()
            writer.writerows(rows)

        dist = generate_query_distribution(
            vocab=vocab,
            tau=2,
            mode="trends",
            csv_path=csv_file,
        )

        assert set(dist.keys()) == {0, 1}
        # Interval 0 corresponds to 2020-01
        assert dist[0]["apple"] == pytest.approx(0.5)
        assert dist[0]["banana"] == pytest.approx(0.3)
        assert dist[0]["cherry"] == pytest.approx(0.2)
        assert abs(sum(dist[0].values()) - 1.0) < 1e-9

        # Interval 1 corresponds to 2020-02
        assert dist[1]["apple"] == pytest.approx(0.1)
        assert dist[1]["banana"] == pytest.approx(0.2)
        assert dist[1]["cherry"] == pytest.approx(0.7)
        assert abs(sum(dist[1].values()) - 1.0) < 1e-9

    def test_trends_handmade_csv_more_months_than_tau(self, tmp_path: pathlib.Path) -> None:
        """Hand-made CSV with 4 months mapped to tau=2 intervals (averaging 2 months per interval)."""
        csv_file = tmp_path / "trends.csv"
        vocab = ["alpha", "beta"]

        rows = [
            {"month": "2021-01", "keyword": "alpha", "prob": 0.8},
            {"month": "2021-01", "keyword": "beta", "prob": 0.2},
            {"month": "2021-02", "keyword": "alpha", "prob": 0.6},
            {"month": "2021-02", "keyword": "beta", "prob": 0.4},
            {"month": "2021-03", "keyword": "alpha", "prob": 0.4},
            {"month": "2021-03", "keyword": "beta", "prob": 0.6},
            {"month": "2021-04", "keyword": "alpha", "prob": 0.2},
            {"month": "2021-04", "keyword": "beta", "prob": 0.8},
        ]

        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["month", "keyword", "prob"])
            writer.writeheader()
            writer.writerows(rows)

        dist = generate_query_distribution(
            vocab=vocab,
            tau=2,
            mode="trends",
            csv_path=csv_file,
        )

        assert set(dist.keys()) == {0, 1}
        # Interval 0: avg of months 1 & 2 -> alpha = (0.8 + 0.6) / 2 = 0.7, beta = 0.3
        assert dist[0]["alpha"] == pytest.approx(0.7)
        assert dist[0]["beta"] == pytest.approx(0.3)
        assert abs(sum(dist[0].values()) - 1.0) < 1e-9

        # Interval 1: avg of months 3 & 4 -> alpha = (0.4 + 0.2) / 2 = 0.3, beta = 0.7
        assert dist[1]["alpha"] == pytest.approx(0.3)
        assert dist[1]["beta"] == pytest.approx(0.7)
        assert abs(sum(dist[1].values()) - 1.0) < 1e-9

    def test_trends_handmade_csv_fewer_months_than_tau(self, tmp_path: pathlib.Path) -> None:
        """Hand-made CSV with 2 months mapped to tau=3 intervals."""
        csv_file = tmp_path / "trends.csv"
        vocab = ["w1", "w2"]

        rows = [
            {"month": "2022-01", "keyword": "w1", "prob": 0.9},
            {"month": "2022-01", "keyword": "w2", "prob": 0.1},
            {"month": "2022-02", "keyword": "w1", "prob": 0.3},
            {"month": "2022-02", "keyword": "w2", "prob": 0.7},
        ]

        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["month", "keyword", "prob"])
            writer.writeheader()
            writer.writerows(rows)

        dist = generate_query_distribution(
            vocab=vocab,
            tau=3,
            mode="trends",
            csv_path=csv_file,
        )

        assert len(dist) == 3
        for t in range(3):
            assert abs(sum(dist[t].values()) - 1.0) < 1e-9
            assert "w1" in dist[t] and "w2" in dist[t]

    def test_trends_handmade_csv_with_missing_vocab_keywords(self, tmp_path: pathlib.Path) -> None:
        """Vocab contains keywords not present in trends.csv; they should be handled with prob 0.0."""
        csv_file = tmp_path / "trends.csv"
        vocab = ["in_csv", "not_in_csv"]

        rows = [
            {"month": "2023-01", "keyword": "in_csv", "prob": 1.0},
        ]

        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["month", "keyword", "prob"])
            writer.writeheader()
            writer.writerows(rows)

        dist = generate_query_distribution(
            vocab=vocab,
            tau=1,
            mode="trends",
            csv_path=csv_file,
        )

        assert dist[0]["in_csv"] == pytest.approx(1.0)
        assert dist[0]["not_in_csv"] == pytest.approx(0.0)
        assert abs(sum(dist[0].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Missing file fallback with logged warning
# ---------------------------------------------------------------------------

class TestMissingTrendsFallback:
    """Test fallback to synthetic mode and warning logging when CSV is missing."""

    def test_missing_file_logs_warning_and_falls_back_to_synthetic(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When trends.csv is absent, generate_query_distribution logs a warning and returns synthetic dist."""
        missing_csv = tmp_path / "non_existent_trends.csv"
        vocab = ["cat", "dog", "bird"]
        tau = 3

        with caplog.at_level(logging.WARNING):
            dist = generate_query_distribution(
                vocab=vocab,
                tau=tau,
                rng=np.random.default_rng(123),
                mode="trends",
                csv_path=missing_csv,
            )

        # Check that a clear warning was logged
        warning_logged = any(
            "not found" in record.message.lower() and "falling back" in record.message.lower()
            for record in caplog.records
        )
        assert warning_logged, f"Expected warning was not logged: {[r.message for r in caplog.records]}"

        # Check that synthetic distribution was returned properly
        assert len(dist) == tau
        for t in range(tau):
            assert set(dist[t].keys()) == set(vocab)
            assert abs(sum(dist[t].values()) - 1.0) < 1e-9
            for kw in vocab:
                assert dist[t][kw] > 0

    def test_missing_file_with_fallback_false_raises_filenotfound(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When fallback_on_missing=False is passed explicitly, FileNotFoundError is raised."""
        missing_csv = tmp_path / "absent.csv"
        with pytest.raises(FileNotFoundError):
            generate_query_distribution(
                vocab=["a", "b"],
                tau=2,
                mode="trends",
                csv_path=missing_csv,
                fallback_on_missing=False,
            )


# ---------------------------------------------------------------------------
# 3. scripts/fetch_trends.py unit tests
# ---------------------------------------------------------------------------

class TestFetchTrendsScript:
    """Test helper functions, caching, retries, and normalization in fetch_trends."""

    def test_load_vocab_file_formats(self, tmp_path: pathlib.Path) -> None:
        """load_vocab_file correctly parses .txt, .json, and .csv vocab files."""
        # Text file
        txt_file = tmp_path / "vocab.txt"
        txt_file.write_text("word1\n# comment\nword2\n\nword3\n", encoding="utf-8")
        assert load_vocab_file(txt_file) == ["word1", "word2", "word3"]

        # JSON file (list)
        json_file = tmp_path / "vocab.json"
        json_file.write_text('["kwA", "kwB"]', encoding="utf-8")
        assert load_vocab_file(json_file) == ["kwA", "kwB"]

        # CSV file
        csv_file = tmp_path / "vocab.csv"
        csv_file.write_text("keyword,freq\nterm1,10\nterm2,20\n", encoding="utf-8")
        assert load_vocab_file(csv_file) == ["term1", "term2"]

    def test_format_timeframe(self) -> None:
        """format_timeframe formats start and end into 'YYYY-MM-DD YYYY-MM-DD'."""
        tf = format_timeframe("2004-01-01", "2004-12-31")
        assert tf == "2004-01-01 2004-12-31"

        tf_month = format_timeframe("2004-01", "2004-02")
        assert tf_month == "2004-01-01 2004-02-29"

    def test_fetch_batch_retry_and_caching(self, tmp_path: pathlib.Path) -> None:
        """fetch_batch_with_retry retries on error and saves result to cache."""
        cache_dir = tmp_path / "cache"
        batch = ["tech", "market"]
        timeframe = "2005-01-01 2005-02-28"

        mock_df = pd.DataFrame(
            {"tech": [40, 60], "market": [20, 30], "isPartial": [False, False]},
            index=pd.to_datetime(["2005-01-01", "2005-02-01"]),
        )

        mock_pytrend = MagicMock()
        # Fail once with rate limit simulation, succeed on second attempt
        mock_pytrend.interest_over_time.side_effect = [
            RuntimeError("Rate limit 429"),
            mock_df,
        ]

        df_result = fetch_batch_with_retry(
            pytrend=mock_pytrend,
            batch=batch,
            timeframe=timeframe,
            cache_dir=cache_dir,
            max_retries=3,
            initial_delay=0.01,
            jitter=0.0,
        )

        assert mock_pytrend.interest_over_time.call_count == 2
        assert "tech" in df_result.columns and "market" in df_result.columns
        assert "isPartial" not in df_result.columns

        # Verify caching: second call should load from cache without hitting mock_pytrend
        mock_pytrend.reset_mock()
        df_cached = fetch_batch_with_retry(
            pytrend=mock_pytrend,
            batch=batch,
            timeframe=timeframe,
            cache_dir=cache_dir,
            max_retries=3,
        )
        mock_pytrend.interest_over_time.assert_not_called()
        assert len(df_cached) == 2

    def test_fetch_trends_normalizes_to_probability_distribution(
        self, tmp_path: pathlib.Path
    ) -> None:
        """fetch_trends normalizes raw interest into monthly distributions summing to 1.0."""
        out_csv = tmp_path / "datasets" / "trends" / "trends.csv"
        cache_dir = tmp_path / "cache"
        keywords = ["energy", "finance"]

        mock_df = pd.DataFrame(
            {"energy": [30.0, 50.0], "finance": [10.0, 50.0]},
            index=pd.to_datetime(["2006-01-01", "2006-02-01"]),
        )

        mock_pytrend = MagicMock()
        mock_pytrend.interest_over_time.return_value = mock_df

        df_out = fetch_trends(
            keywords=keywords,
            start="2006-01-01",
            end="2006-02-28",
            out_path=out_csv,
            cache_dir=cache_dir,
            request_delay=0.0,
            pytrend=mock_pytrend,
        )

        assert out_csv.exists()
        assert list(df_out.columns) == ["month", "keyword", "prob"]

        # Month 1: energy = 30 / (30 + 10) = 0.75, finance = 0.25
        m1 = df_out[df_out["month"] == "2006-01"].set_index("keyword")["prob"]
        assert m1["energy"] == pytest.approx(0.75)
        assert m1["finance"] == pytest.approx(0.25)
        assert abs(m1.sum() - 1.0) < 1e-9

        # Month 2: energy = 50 / (50 + 50) = 0.5, finance = 0.5
        m2 = df_out[df_out["month"] == "2006-02"].set_index("keyword")["prob"]
        assert m2["energy"] == pytest.approx(0.5)
        assert m2["finance"] == pytest.approx(0.5)
        assert abs(m2.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Anchor-based cross-batch normalization
# ---------------------------------------------------------------------------

class _ScaledTrends:
    """Fake pytrends: true interest is fixed; each request rescales so its max is 100."""

    TRUE = {"anchor": 20.0, "a": 10.0, "b": 40.0, "c": 5.0, "d": 80.0, "e": 30.0, "f": 60.0}

    def build_payload(self, batch, timeframe):
        self.batch = batch

    def interest_over_time(self):
        idx = pd.date_range("2021-01-01", periods=2, freq="MS")
        top = max(self.TRUE[k] for k in self.batch)
        return pd.DataFrame({k: [round(self.TRUE[k] / top * 100)] * 2 for k in self.batch}, index=idx)


def test_fetch_trends_anchor_puts_batches_on_one_scale(tmp_path: pathlib.Path) -> None:
    kws = ["a", "b", "c", "d", "e", "f"]
    df = fetch_trends(kws, "2021-01", "2021-02", out_path=tmp_path / "t.csv", cache_dir=None,
                      request_delay=0, pytrend=_ScaledTrends(), anchor="anchor")
    month = df[df["month"] == "2021-01"].set_index("keyword")["prob"]
    true = pd.Series({k: _ScaledTrends.TRUE[k] for k in kws})
    expected = true / true.sum()
    assert "anchor" not in month.index
    for k in kws:
        assert month[k] == pytest.approx(expected[k], rel=0.05)


def test_fetch_trends_without_anchor_keeps_batches_separate(tmp_path: pathlib.Path) -> None:
    kws = ["a", "b", "c", "d", "e", "f"]
    df = fetch_trends(kws, "2021-01", "2021-02", out_path=tmp_path / "t.csv", cache_dir=None,
                      request_delay=0, pytrend=_ScaledTrends())
    month = df[df["month"] == "2021-01"].set_index("keyword")["prob"]
    # "d" (true 80) and "f" (true 60) each top their own batch, so without an
    # anchor both read as 100 and come out equal.
    assert month["f"] == pytest.approx(month["d"])
