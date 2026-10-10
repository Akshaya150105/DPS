"""Tests for YAML-based configuration loading."""

from config import Config, load_config


def test_load_config_applies_values_and_dataclass_defaults(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset: enron\nseed: 9\n", encoding="utf-8")

    config = load_config(path)

    assert config.dataset == "enron"
    assert config.seed == 9
    assert config.tau == Config().tau
    assert config.attack == Config().attack


def test_example_configs_load() -> None:
    base = load_config("configs/base.yaml")
    synthetic = load_config("configs/synthetic.yaml")

    assert base == Config()
    assert synthetic.dataset == "synthetic"
    assert synthetic.seed == Config().seed


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("not_a_config_field: true\n", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)
