"""Tests for experiment charts and date-based paper figures."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import pathlib

import pytest

import plots
from types_ import Document


def _write_csv(path: pathlib.Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _dated_docs(month_count: int = 6, keyword_count: int = 60) -> list[Document]:
    docs: list[Document] = []
    for month_offset in range(month_count):
        year = 2024 + (month_offset // 12)
        month = month_offset % 12 + 1
        for keyword_index in range(keyword_count):
            docs.append(Document(
                doc_id=len(docs),
                keywords=frozenset({f"kw{keyword_index}"}),
                timestamp=datetime(year, month, 1, tzinfo=timezone.utc),
            ))
    return docs


class TestPlotExperimentResults:
    def test_plots_parameter_sweeps_with_seed_aggregation(self, tmp_path: pathlib.Path) -> None:
        result_csv = tmp_path / "results.csv"
        rows = []
        for tau in (1, 3):
            for gamma in (0.0, 0.2):
                for seed, accuracy in ((1, 0.4), (2, 0.6)):
                    rows.append({
                        "dataset": "enron",
                        "attack": "fma",
                        "equality": "T1",
                        "tau": tau,
                        "gamma": gamma,
                        "seed": seed,
                        "accuracy": accuracy + 0.05 * tau,
                    })
        _write_csv(result_csv, list(rows[0]), rows)

        outputs = plots.plot_experiment_results(result_csv, tmp_path / "figures")

        assert {path.name for path in outputs} == {
            "enron_fma_t1_vs_tau.png",
            "enron_fma_t1_vs_gamma.png",
        }
        assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)

    def test_uses_runner_default_and_alternate_csv_path(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = tmp_path / "results" / "results.csv"
        _write_csv(result, ["accuracy", "seed", "tau"], [{"accuracy": 0.5, "seed": 1, "tau": 2}])

        assert plots._find_results_csv(None) == pathlib.Path("results/results.csv")

        result.unlink()
        alternate = tmp_path / "results" / "csv" / "results.csv"
        _write_csv(alternate, ["accuracy", "seed", "tau"], [{"accuracy": 0.5, "seed": 1, "tau": 2}])
        assert plots._find_results_csv(None) == pathlib.Path("results/csv/results.csv")

    def test_missing_explicit_results_csv_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="Results CSV not found"):
            plots.plot_experiment_results(tmp_path / "missing.csv", tmp_path / "figures")


class TestMonthlyPlots:
    def test_creates_monthly_query_and_response_length_figures(self, tmp_path: pathlib.Path) -> None:
        docs = _dated_docs()
        query = plots.plot_monthly_query_distribution(docs, tmp_path / "query.png", seed=7)
        growth = plots.plot_monthly_response_growth(docs, tmp_path / "growth.png", seed=7)

        assert query.is_file() and query.stat().st_size > 0
        assert growth.is_file() and growth.stat().st_size > 0

    @pytest.mark.parametrize(
        ("function", "month_count", "kwargs", "message"),
        [
            (plots.plot_monthly_query_distribution, 1, {}, "requires 2 distinct months"),
            (plots.plot_monthly_response_growth, 5, {}, "requires 6 distinct months"),
            (
                plots.plot_monthly_response_growth,
                6,
                {"keyword_count": 61},
                "needs 61 distinct keywords",
            ),
        ],
    )
    def test_rejects_insufficient_dated_data(
        self,
        tmp_path: pathlib.Path,
        function,
        month_count: int,
        kwargs: dict[str, int],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            function(_dated_docs(month_count=month_count), tmp_path / "plot.png", **kwargs)

    def test_rejects_requested_month_without_documents(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="no real processed documents"):
            plots.plot_monthly_query_distribution(
                _dated_docs(),
                tmp_path / "query.png",
                months=("2023-12", "2024-01"),
            )


class TestPaperComparison:
    def test_writes_markdown_and_csv_comparison(self, tmp_path: pathlib.Path) -> None:
        ours = tmp_path / "ours.csv"
        paper = tmp_path / "paper_results.csv"
        headers = ["dataset", "attack", "equality", "tau", "accuracy", "seed"]
        _write_csv(ours, headers, [
            {"dataset": "Enron", "attack": "FMA", "equality": "T1", "tau": 3, "accuracy": 0.7, "seed": 1},
            {"dataset": "Enron", "attack": "FMA", "equality": "T1", "tau": 3, "accuracy": 0.9, "seed": 2},
        ])
        _write_csv(paper, headers, [
            {"dataset": "enron", "attack": "fma", "equality": "t1", "tau": 3, "accuracy": 0.6, "seed": 1},
            {"dataset": "enron", "attack": "fma", "equality": "t1", "tau": 3, "accuracy": 0.8, "seed": 2},
        ])
        md_path = tmp_path / "comparison.md"
        csv_path = tmp_path / "comparison.csv"

        rows = plots.build_paper_comparison_table(
            paper,
            ours,
            markdown_path=md_path,
            csv_path=csv_path,
        )

        assert len(rows) == 1
        assert rows[0]["paper_accuracy"] == pytest.approx(0.7)
        assert rows[0]["ours_mean_accuracy"] == pytest.approx(0.8)
        assert rows[0]["ours_std_accuracy"] == pytest.approx(0.1414213562)
        assert rows[0]["accuracy_delta"] == pytest.approx(0.1)
        assert "| Paper accuracy | Ours mean accuracy |" in md_path.read_text(encoding="utf-8")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 1
        assert float(csv_rows[0]["ours_mean_accuracy"]) == pytest.approx(0.8)

    def test_requires_matching_configurations(self, tmp_path: pathlib.Path) -> None:
        headers = ["dataset", "accuracy"]
        ours = tmp_path / "ours.csv"
        paper = tmp_path / "paper.csv"
        _write_csv(ours, headers, [{"dataset": "ours", "accuracy": 0.5}])
        _write_csv(paper, headers, [{"dataset": "paper", "accuracy": 0.6}])
        with pytest.raises(ValueError, match="No matching experiment configurations"):
            plots.build_paper_comparison_table(paper, ours)
