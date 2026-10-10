"""Tests for run_experiments.py: single run, grid runner, deduplication, and YAML presets."""

from __future__ import annotations

import csv
import pathlib
import pytest
import yaml

from config import Config
import evaluate
import run_experiments


# ---------------------------------------------------------------------------
# TestRunOne
# ---------------------------------------------------------------------------

class TestRunOne:
    @pytest.mark.parametrize(
        "attack",
        [
            "fma", "via", "pvia", "lvia-k", "lvia-s", "no_update",
            "no-update-pvia", "no-update-lvia",
        ],
    )
    def test_run_one_synthetic_all_attacks(self, attack: str, tmp_path: pathlib.Path) -> None:
        """run_one must execute all supported attacks cleanly on the synthetic dataset."""
        csv_file = tmp_path / "results.csv"
        cfg = {
            "dataset": "synthetic",
            "attack": attack,
            "equality": "T1",
            "tau": 3,
            "lambda": 40.0,
            "gamma": 0.0,
            "alpha": 0.8,
            "eta": 0.5,
            "noise": 0.0,
            "seed": 42,
        }
        res = run_experiments.run_one(cfg, csv_path=csv_file)

        assert isinstance(res, dict)
        assert res["attack"] == run_experiments._normalize_attack_name(attack)
        assert res["dataset"] == "synthetic"
        assert 0.0 <= res["accuracy"] <= 1.0
        assert res["correct"] <= res["total"]
        assert res["total"] > 0
        assert res["runtime_s"] >= 0.0

        # Check CSV was created and contains row
        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["attack"] == run_experiments._normalize_attack_name(attack)
        assert rows[0]["dataset"] == "synthetic"
        assert float(rows[0]["accuracy"]) == pytest.approx(res["accuracy"], abs=1e-5)

    def test_run_one_with_config_object(self, tmp_path: pathlib.Path) -> None:
        """run_one must accept a Config dataclass instance."""
        csv_file = tmp_path / "results_cfg.csv"
        cfg = Config(
            seed=7,
            dataset="synthetic",
            tau=2,
            lambda_queries=30.0,
            gamma=0.0,
            equality="T1",
            attack="via",
        )
        res = run_experiments.run_one(cfg, csv_path=csv_file)
        assert res["attack"] == "via"
        assert res["seed"] == 7
        assert csv_file.exists()

    def test_run_one_t2_equality(self, tmp_path: pathlib.Path) -> None:
        """run_one must handle T2 equality inference."""
        csv_file = tmp_path / "results_t2.csv"
        cfg = {
            "dataset": "synthetic",
            "attack": "fma",
            "equality": "T2",
            "tau": 2,
            "lambda": 40.0,
            "gamma": 0.0,
            "noise": 0.0,
            "seed": 42,
        }
        res = run_experiments.run_one(cfg, csv_path=csv_file)
        assert res["equality"] == "T2"
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_run_one_noisy_fma(self, tmp_path: pathlib.Path) -> None:
        """run_one must support noise for FMA query distribution."""
        csv_file = tmp_path / "results_noise.csv"
        cfg = {
            "dataset": "synthetic",
            "attack": "fma",
            "equality": "T1",
            "tau": 2,
            "lambda": 40.0,
            "gamma": 0.0,
            "noise": 5.0,
            "seed": 42,
        }
        res = run_experiments.run_one(cfg, csv_path=csv_file)
        assert res["noise"] == 5.0
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_run_one_unknown_attack_raises(self) -> None:
        """run_one must raise ValueError when given an unknown attack name."""
        with pytest.raises(ValueError, match="Unknown attack"):
            run_experiments.run_one({"attack": "nonexistent_attack"})


# ---------------------------------------------------------------------------
# TestRunGrid
# ---------------------------------------------------------------------------

class TestRunGrid:
    def test_run_grid_tiny_synthetic(self, tmp_path: pathlib.Path) -> None:
        """run_grid must execute all combinations in a tiny grid on synthetic data."""
        csv_file = tmp_path / "grid_results.csv"
        grid = {
            "dataset": ["synthetic"],
            "attack": ["lvia-k", "no_update"],
            "equality": ["T1"],
            "tau": [2],
            "lambda": [30.0],
            "gamma": [0.0],
            "alpha": [0.8],
            "eta": [0.5],
            "noise": [0.0],
        }
        # 2 attacks * 2 seeds = 4 runs
        results = run_experiments.run_grid(
            grid=grid,
            n_seeds=2,
            csv_path=csv_file,
            dry_run=False,
            verbose=False,
        )

        assert len(results) == 4
        assert csv_file.exists()

        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

        attacks = {r["attack"] for r in rows}
        assert attacks == {"lvia-k", "no_update"}
        seeds = {int(r["seed"]) for r in rows}
        assert seeds == {0, 1}

    def test_run_grid_skipping_already_logged_rows(self, tmp_path: pathlib.Path) -> None:
        """run_grid must skip runs that already exist in the CSV."""
        csv_file = tmp_path / "skip_results.csv"
        grid = {
            "dataset": ["synthetic"],
            "attack": ["via"],
            "equality": ["T1"],
            "tau": [2],
            "lambda": [25.0],
            "gamma": [0.0],
            "alpha": [0.8],
            "eta": [0.5],
            "noise": [0.0],
        }
        # First run: 2 seeds executed
        first_res = run_experiments.run_grid(grid, n_seeds=2, csv_path=csv_file, verbose=False)
        assert len(first_res) == 2

        # Second run with same grid: 0 runs executed, 2 skipped
        second_res = run_experiments.run_grid(grid, n_seeds=2, csv_path=csv_file, verbose=False)
        assert len(second_res) == 0

        # Run with 3 seeds: only seed 2 should execute
        third_res = run_experiments.run_grid(grid, n_seeds=3, csv_path=csv_file, verbose=False)
        assert len(third_res) == 1
        assert third_res[0]["seed"] == 2

        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_run_grid_dry_run_flag(self, tmp_path: pathlib.Path) -> None:
        """--dry-run must only list the runs without executing or writing to CSV."""
        csv_file = tmp_path / "dry_results.csv"
        grid = {
            "dataset": ["synthetic"],
            "attack": ["fma"],
            "equality": ["T1", "T2"],
            "tau": [2],
            "lambda": [20.0],
            "gamma": [0.0],
            "alpha": [0.8],
            "eta": [0.5],
            "noise": [0.0],
        }
        planned = run_experiments.run_grid(
            grid=grid,
            n_seeds=3,
            csv_path=csv_file,
            dry_run=True,
            verbose=False,
        )

        # 2 equalities * 3 seeds = 6 planned runs
        assert len(planned) == 6
        assert not csv_file.exists(), "dry_run must not create or modify CSV file."


# ---------------------------------------------------------------------------
# TestPaperFigureYAMLs
# ---------------------------------------------------------------------------

class TestPaperFigureYAMLs:
    EXPECTED_GRIDS = [
        ("grids/fma_vs_tau.yaml", "FMA vs tau (T1, T2)", {"tau", "equality"}),
        ("grids/fma_vs_lambda.yaml", "FMA vs lambda", {"lambda"}),
        ("grids/fma_vs_gamma.yaml", "FMA vs gamma", {"gamma"}),
        ("grids/fma_vs_tau_gamma.yaml", "FMA vs tau and gamma", {"tau", "gamma"}),
        ("grids/fma_noise.yaml", "FMA with noise 0, 5, 10", {"noise"}),
        ("grids/pvia_alpha_by_gamma.yaml", "PVIA alpha by gamma", {"alpha", "gamma"}),
        ("grids/lvia_k_vs_s.yaml", "LVIA-K vs LVIA-S", {"attack"}),
        ("grids/lvia_s_vs_tau.yaml", "LVIA-S vs tau", {"tau"}),
        ("grids/lvia_s_vs_eta.yaml", "LVIA-S vs eta", {"eta"}),
        ("grids/no_update_variants.yaml", "No-update variants", {"attack", "alpha"}),
    ]

    @pytest.mark.parametrize("filepath, preset_name, varied_keys", EXPECTED_GRIDS)
    def test_yaml_file_exists_and_valid(
        self,
        filepath: str,
        preset_name: str,
        varied_keys: set[str],
    ) -> None:
        """Every requested YAML grid file must exist, be valid YAML, and vary expected parameters."""
        path = pathlib.Path(filepath)
        assert path.exists(), f"Grid file {filepath} is missing"

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        for key in ("dataset", "attack", "equality", "tau", "lambda", "gamma", "alpha", "eta", "noise"):
            assert key in data, f"Missing key {key} in {filepath}"

        # Check preset dictionary
        assert preset_name in run_experiments.PAPER_GRIDS

        # Check that expected keys have multiple values (or specific target values)
        for k in varied_keys:
            val = data[k]
            assert isinstance(val, list), f"{k} in {filepath} should be a list"
            assert len(val) >= 2 or k == "attack", f"{k} should have multiple values in {filepath}"

    def test_yaml_dry_run_all_grids(self) -> None:
        """Dry-run on all YAML grids must succeed without error."""
        for filepath, _, _ in self.EXPECTED_GRIDS:
            runs = run_experiments.run_grid(
                grid=filepath,
                n_seeds=1,
                csv_path="results/dummy.csv",
                dry_run=True,
                verbose=False,
            )
            assert len(runs) > 0


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_dry_run(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        """CLI with --dry-run must print planned runs cleanly."""
        monkeypatch.setattr(
            "sys.argv",
            ["run_experiments.py", "--grid", "grids/fma_vs_tau.yaml", "--dry-run", "--n-seeds", "1"],
        )
        run_experiments.main()
        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "Total runs:" in captured.out

    def test_cli_single_run_config_file(self, monkeypatch, tmp_path: pathlib.Path) -> None:
        config_file = tmp_path / "single.yaml"
        csv_file = tmp_path / "configured.csv"
        config_file.write_text(
            "dataset: synthetic\nattack: fma\ntau: 2\nlambda_queries: 20\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_experiments.py",
                "--config",
                str(config_file),
                "--csv-path",
                str(csv_file),
            ],
        )

        run_experiments.main()

        assert csv_file.exists()
