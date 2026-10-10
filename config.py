"""Global experiment configuration."""

from dataclasses import dataclass, field


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
