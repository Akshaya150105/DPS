"""Tests for plots.py: figure generation, error bars, extra plots, and paper comparison table."""

from __future__ import annotations

import pathlib
import pytest
import pandas as pd

import plots


@pytest.fixture
def mock_results_df() -> pd.DataFrame:
    """Create a rich mock DataFrame covering seeds and multiple experiment sweeps."""
    rows: list[dict] = []
    # FMA vs tau: tau in [1, 6, 12], T1 & T2, seeds 0, 1
    for t in [1, 6, 12]:
        for eq in ["T1", "T2"]:
            for s in [0, 1]:
                rows.append({
                    "dataset": "enron", "attack": "fma", "equality": eq,
                    "tau": t, "lambda": 120000.0, "gamma": 0.0, "alpha": 0.3,
                    "eta": 0.5, "noise": 0.0, "accuracy": 0.65 + 0.02 * t, "seed": s,
                })
    # FMA vs lambda: lambda in [120000, 240000], seeds 0, 1
    for lam in [120000.0, 240000.0]:
        for eq in ["T1", "T2"]:
            for s in [0, 1]:
                rows.append({
                    "dataset": "enron", "attack": "fma", "equality": eq,
                    "tau": 24, "lambda": lam, "gamma": 0.0, "alpha": 0.3,
                    "eta": 0.5, "noise": 0.0, "accuracy": 0.70 + 1e-6 * lam, "seed": s,
                })
    # FMA vs gamma: gamma in [0.0, 0.1, 0.2], seeds 0, 1
    for g in [0.0, 0.1, 0.2]:
        for eq in ["T1", "T2"]:
            for s in [0, 1]:
                rows.append({
                    "dataset": "enron", "attack": "fma", "equality": eq,
                    "tau": 24, "lambda": 120000.0, "gamma": g, "alpha": 0.3,
                    "eta": 0.5, "noise": 0.0, "accuracy": 0.75 - 0.2 * g, "seed": s,
                })
    # FMA vs tau & gamma (T2): tau in [6, 12], gamma in [0.0, 0.1]
    for t in [6, 12]:
        for g in [0.0, 0.1]:
            for s in [0, 1]:
                rows.append({
                    "dataset": "enron", "attack": "fma", "equality": "T2",
                    "tau": t, "lambda": 12000.0, "gamma": g, "alpha": 0.3,
                    "eta": 0.5, "noise": 0.0, "accuracy": 0.5 + 0.01 * t - 0.1 * g, "seed": s,
                })
    # FMA noise: noise in [0.0, 5.0, 10.0]
    for n in [0.0, 5.0, 10.0]:
        for s in [0, 1]:
            rows.append({
                "dataset": "enron", "attack": "fma", "equality": "T2",
                "tau": 24, "lambda": 120000.0, "gamma": 0.0, "alpha": 0.3,
                "eta": 0.5, "noise": n, "accuracy": 0.7 - 0.02 * n, "seed": s,
            })
    # PVIA alpha by gamma: alpha in [0.6, 0.8], gamma in [0.1, 0.2]
    for a in [0.6, 0.8]:
        for g in [0.1, 0.2]:
            for s in [0, 1]:
                rows.append({
                    "dataset": "enron", "attack": "pvia", "equality": "T1",
                    "tau": 10, "lambda": 500.0, "gamma": g, "alpha": a,
                    "eta": 0.5, "noise": 0.0, "accuracy": a - 0.5 * g, "seed": s,
                })
    # LVIA-K vs LVIA-S
    for atk in ["lvia-k", "lvia-s"]:
        for s in [0, 1]:
            rows.append({
                "dataset": "enron", "attack": atk, "equality": "T1",
                "tau": 1, "lambda": 500.0, "gamma": 0.0, "alpha": 0.3,
                "eta": 0.5, "noise": 0.0, "accuracy": 0.8 if atk == "lvia-k" else 0.1, "seed": s,
            })
    # LVIA-S vs tau: tau in [1, 2, 6]
    for t in [1, 2, 6]:
        for s in [0, 1]:
            rows.append({
                "dataset": "enron", "attack": "lvia-s", "equality": "T1",
                "tau": t, "lambda": 500.0, "gamma": 0.0, "alpha": 0.3,
                "eta": 0.5, "noise": 0.0, "accuracy": 0.1 + 0.05 * t, "seed": s,
            })
    # LVIA-S vs eta: eta in [0.2, 0.5]
    for e in [0.2, 0.5]:
        for s in [0, 1]:
            rows.append({
                "dataset": "enron", "attack": "lvia-s", "equality": "T1",
                "tau": 18, "lambda": 500.0, "gamma": 0.0, "alpha": 0.3,
                "eta": e, "noise": 0.0, "accuracy": e * 0.8, "seed": s,
            })
    return pd.DataFrame(rows)


class TestPlotGeneration:
    def test_aggregate_over_seeds(self, mock_results_df: pd.DataFrame) -> None:
        """aggregate_over_seeds must compute mean, std, and sample count."""
        sub = mock_results_df[mock_results_df["attack"] == "fma"]
        agg = plots.aggregate_over_seeds(sub, ["tau", "equality"])
        assert "mean" in agg.columns
        assert "std" in agg.columns
        assert "count" in agg.columns
        assert (agg["count"] >= 1).all()
        assert (agg["mean"] >= 0.0).all() and (agg["mean"] <= 1.0).all()

    def test_plot_fma_vs_tau(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_fma_vs_tau(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()
        assert pathlib.Path(path).stat().st_size > 0

    def test_plot_fma_vs_lambda(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_fma_vs_lambda(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_fma_vs_gamma(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_fma_vs_gamma(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_fma_vs_tau_and_gamma(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_fma_vs_tau_and_gamma(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_fma_noise(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_fma_noise(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_pvia_alpha_by_gamma(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_pvia_alpha_by_gamma(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_lvia_k_vs_s(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_lvia_k_vs_s(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_lvia_s_vs_tau(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_lvia_s_vs_tau(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_plot_lvia_s_vs_eta(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        path = plots.plot_lvia_s_vs_eta(mock_results_df, output_dir=tmp_path)
        assert path is not None
        assert pathlib.Path(path).exists()


class TestExtraPlots:
    def test_plot_query_distribution_two_months(self, tmp_path: pathlib.Path) -> None:
        """Extra Plot 1: Query distribution of 10 keywords across two months."""
        out = tmp_path / "extra1.png"
        res = plots.plot_query_distribution_two_months(dataset_name="synthetic", seed=42, output_path=out)
        assert pathlib.Path(res).exists()
        assert pathlib.Path(res).stat().st_size > 0

    def test_plot_response_length_growth(self, tmp_path: pathlib.Path) -> None:
        """Extra Plot 2: Response-length growth of 50 keywords over 6 months."""
        out = tmp_path / "extra2.png"
        res = plots.plot_response_length_growth(dataset_name="synthetic", tau=6, n_keywords=10, seed=42, output_path=out)
        assert pathlib.Path(res).exists()
        assert pathlib.Path(res).stat().st_size > 0


class TestComparisonTable:
    def test_build_comparison_table(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        """build_comparison_table must create markdown and CSV outputs comparing paper vs ours."""
        ours_csv = tmp_path / "ours.csv"
        mock_results_df.to_csv(ours_csv, index=False)

        paper_csv = tmp_path / "paper_results.csv"
        plots.create_paper_results_template(paper_csv)
        assert paper_csv.exists()

        out_md = tmp_path / "comparison.md"
        out_csv = tmp_path / "comparison.csv"

        md_str, comp_df = plots.build_comparison_table(
            paper_csv_path=paper_csv,
            ours_csv_path=ours_csv,
            output_md_path=out_md,
            output_csv_path=out_csv,
        )

        assert out_md.exists()
        assert out_csv.exists()
        assert isinstance(md_str, str)
        assert isinstance(comp_df, pd.DataFrame)
        assert not comp_df.empty
        assert "paper_accuracy" in comp_df.columns
        assert "difference" in comp_df.columns
        assert "our_result" in comp_df.columns

    def test_run_all_plots(self, mock_results_df: pd.DataFrame, tmp_path: pathlib.Path) -> None:
        """run_all_plots must generate experiment figures, extra figures, and comparison table."""
        ours_csv = tmp_path / "results.csv"
        mock_results_df.to_csv(ours_csv, index=False)

        fig_dir = tmp_path / "figures"
        paper_csv = tmp_path / "paper_results.csv"

        generated = plots.run_all_plots(
            results_csv=ours_csv,
            output_dir=fig_dir,
            paper_csv=paper_csv,
        )

        assert len(generated) >= 11
        for f in generated:
            assert pathlib.Path(f).exists()
            assert pathlib.Path(f).stat().st_size > 0
