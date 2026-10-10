"""Global experiment configuration."""

from dataclasses import dataclass, fields
import pathlib
from typing import Any


@dataclass
class Config:
    """Holds all hyperparameters for a single experiment run.

    Attributes:
        seed:           Master RNG seed passed to numpy Generator.
        dataset:        Name / path identifier for the dataset to use.
        keyword_count:  Number of distinct keywords in the simulated corpus.
        tau:            Number of update epochs (time intervals).
        lambda_queries: Expected number of queries issued per epoch (Poisson rate).
        gamma:          Fraction of documents touched per update epoch.
        alpha:          Fraction of updates that are deletions (0 = add-only).
        eta:            Adversary's query-distribution knowledge weight (IKK-style).
        noise_n:        Std-dev of Gaussian noise added to response counts (0 = no noise).
        equality:       Equality-inference mode: 'T1' (token prefix) or 'T2' (response sim).
        delta:          Response-similarity threshold for T2 equality inference.
    """

    seed: int = 42
    dataset: str = "synthetic"
    keyword_count: int = 500
    tau: int = 10
    lambda_queries: float = 50.0
    gamma: float = 0.1
    alpha: float = 0.3
    eta: float = 0.5
    noise_n: float = 0.0
    equality: str = "T1"
    delta: float = 0.6
    attack: str = "fma"


def load_config(path: str | pathlib.Path) -> Config:
    """Load YAML configuration, retaining dataclass defaults for missing fields."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configuration files.") from exc

    config_path = pathlib.Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        values: Any = yaml.safe_load(stream)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: '{config_path}'.")
    valid_fields = {item.name for item in fields(Config)}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown configuration field(s) in '{config_path}': {', '.join(unknown)}.")
    return Config(**values)
