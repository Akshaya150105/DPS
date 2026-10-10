"""Tests for config.py: load_config, base.yaml, and synthetic.yaml."""

from __future__ import annotations

import pathlib
import pytest

from config import Config, load_config


class TestLoadConfig:
    def test_load_base_config(self) -> None:
        """load_config('configs/base.yaml') must load all base default values."""
        path = pathlib.Path("configs/base.yaml")
        assert path.exists(), "configs/base.yaml must exist."

        cfg = load_config(path)
        assert isinstance(cfg, Config)
        assert cfg.dataset == "enron"
        assert cfg.seed == 42
        assert cfg.keyword_count == 500
        assert cfg.tau == 10
        assert cfg.lambda_queries == 50.0
        assert cfg.gamma == 0.1
        assert cfg.alpha == 0.3
        assert cfg.eta == 0.5
        assert cfg.noise_n == 0.0
        assert cfg.equality == "T1"
        assert cfg.delta == 0.6

    def test_load_synthetic_config_overrides_dataset(self) -> None:
        """load_config('configs/synthetic.yaml') must override dataset to 'synthetic' and keep defaults."""
        path = pathlib.Path("configs/synthetic.yaml")
        assert path.exists(), "configs/synthetic.yaml must exist."

        cfg = load_config(path)
        assert isinstance(cfg, Config)
        assert cfg.dataset == "synthetic"
        # All other fields must fall back to default
        assert cfg.seed == 42
        assert cfg.keyword_count == 500
        assert cfg.tau == 10
        assert cfg.lambda_queries == 50.0
        assert cfg.gamma == 0.1
        assert cfg.alpha == 0.3
        assert cfg.eta == 0.5
        assert cfg.noise_n == 0.0
        assert cfg.equality == "T1"
        assert cfg.delta == 0.6

    def test_missing_file_raises_filenotfound(self) -> None:
        """load_config must raise FileNotFoundError for nonexistent paths."""
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_empty_yaml_returns_default_config(self, tmp_path: pathlib.Path) -> None:
        """load_config on an empty YAML file returns default Config."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        cfg = load_config(empty_file)
        assert cfg == Config()

    def test_aliases_and_extra_fields(self, tmp_path: pathlib.Path) -> None:
        """load_config should support 'lambda' and 'noise' aliases and ignore extra keys."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("lambda: 120.0\nnoise: 5.0\nunknown_field: 'ignored'\n", encoding="utf-8")
        cfg = load_config(test_file)
        assert cfg.lambda_queries == 120.0
        assert cfg.noise_n == 5.0
