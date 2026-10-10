import dataclasses
from dataclasses import dataclass, field
import pathlib
from typing import Any
import yaml


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
    """Read a YAML file and return a Config, with any missing field falling back to default.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config dataclass populated with values from YAML and defaults for missing fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML content is not a mapping.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}
    elif not isinstance(data, dict):
        raise ValueError(f"YAML configuration must be a mapping, got {type(data)}")

    # Support aliases
    if "lambda" in data and "lambda_queries" not in data:
        data["lambda_queries"] = data.pop("lambda")
    if "noise" in data and "noise_n" not in data:
        data["noise_n"] = data.pop("noise")

    # Filter to valid dataclass fields so unknown keys in YAML don't cause TypeError
    valid_fields = {f.name for f in dataclasses.fields(Config)}
    filtered_kwargs = {k: v for k, v in data.items() if k in valid_fields}

    return Config(**filtered_kwargs)
