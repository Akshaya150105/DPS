"""Experiment runner: orchestrates configurations, runs attacks, logs results, and supports grid sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pathlib
import pickle
import time
from typing import Any, Iterable

import numpy as np

import attacks.fma as fma
import attacks.lvia as lvia
import attacks.no_update as no_update
import attacks.pvia as pvia
import attacks.via as via
from config import Config
from equality import build_equality
import evaluate
from simulator import (
    generate_query_distribution,
    make_aux_data_dist,
    make_aux_known_docs,
    make_aux_pre_snapshot,
    make_aux_query_dist,
    simulate,
)
from synthetic import VOCAB, make_obvious_query_dist, make_synthetic
from types_ import Aux, Document


# ---------------------------------------------------------------------------
# Paper Grid Presets
# ---------------------------------------------------------------------------

PAPER_GRIDS: dict[str, str] = {
    "FMA vs tau (T1, T2)": "grids/fma_vs_tau.yaml",
    "fma_vs_tau": "grids/fma_vs_tau.yaml",
    "FMA vs lambda": "grids/fma_vs_lambda.yaml",
    "fma_vs_lambda": "grids/fma_vs_lambda.yaml",
    "FMA vs gamma": "grids/fma_vs_gamma.yaml",
    "fma_vs_gamma": "grids/fma_vs_gamma.yaml",
    "FMA vs tau and gamma": "grids/fma_vs_tau_gamma.yaml",
    "fma_vs_tau_gamma": "grids/fma_vs_tau_gamma.yaml",
    "FMA with noise 0, 5, 10": "grids/fma_noise.yaml",
    "fma_noise": "grids/fma_noise.yaml",
    "PVIA alpha by gamma": "grids/pvia_alpha_by_gamma.yaml",
    "pvia_alpha_gamma": "grids/pvia_alpha_by_gamma.yaml",
    "LVIA-K vs LVIA-S": "grids/lvia_k_vs_s.yaml",
    "lvia_k_vs_s": "grids/lvia_k_vs_s.yaml",
    "LVIA-S vs tau": "grids/lvia_s_vs_tau.yaml",
    "lvia_s_vs_tau": "grids/lvia_s_vs_tau.yaml",
    "LVIA-S vs eta": "grids/lvia_s_vs_eta.yaml",
    "lvia_s_vs_eta": "grids/lvia_s_vs_eta.yaml",
    "No-update variants": "grids/no_update_variants.yaml",
    "no_update_variants": "grids/no_update_variants.yaml",
}


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    seed: int = 42,
    keyword_count: int | None = None,
) -> tuple[list[Document], list[str]]:
    """Load or generate a document dataset.

    Args:
        dataset_name:  'synthetic', 'enron', 'lucene', or a path to a pickle file.
        seed:          RNG seed used for synthetic generation.
        keyword_count: Optional max vocabulary size to keep.

    Returns:
        (docs, vocab): List of Document objects and sorted vocabulary list.
    """
    name_clean = dataset_name.strip().lower()

    if name_clean in ("synthetic", "obvious", "trends"):
        docs = make_synthetic(seed=seed)
        vocab = sorted({kw for d in docs for kw in d.keywords})
        if keyword_count is not None and keyword_count < len(vocab):
            vocab = vocab[:keyword_count]
            allowed = set(vocab)
            docs = [
                Document(doc_id=d.doc_id, keywords=frozenset(k for k in d.keywords if k in allowed), timestamp=d.timestamp)
                for d in docs
            ]
            docs = [d for d in docs if len(d.keywords) > 0]
        return docs, vocab

    # Check potential file paths for preprocessed datasets
    candidates = [
        pathlib.Path(dataset_name),
        pathlib.Path("datasets/processed") / f"{dataset_name}.pkl",
        pathlib.Path("datasets/processed") / dataset_name,
    ]
    target_path = next((p for p in candidates if p.exists() and p.is_file()), None)

    if target_path is not None:
        with open(target_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and len(obj) >= 2:
            docs, vocab = obj[0], obj[1]
        elif isinstance(obj, list):
            docs = obj
            vocab = sorted({kw for d in docs for kw in d.keywords})
        else:
            raise ValueError(f"Unrecognized data format in {target_path}")

        if keyword_count is not None and keyword_count < len(vocab):
            vocab = vocab[:keyword_count]
            allowed = set(vocab)
            docs = [
                Document(doc_id=d.doc_id, keywords=frozenset(k for k in d.keywords if k in allowed), timestamp=d.timestamp)
                for d in docs
            ]
            docs = [d for d in docs if len(d.keywords) > 0]
        return docs, sorted(vocab)

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Expected 'synthetic' or preprocessed file in datasets/processed/"
    )


# ---------------------------------------------------------------------------
# Single Experiment Runner
# ---------------------------------------------------------------------------

def _normalize_attack_name(attack: str) -> str:
    """Normalize attack identifier to standard name."""
    a = attack.strip().lower().replace("_", "-")
    if a in (
        "fma", "via", "pvia", "lvia-k", "lvia-s",
        "no-update-pvia", "no-update-lvia",
    ):
        return a
    if a in ("no-update", "no_update", "noupdate", "no-update-full"):
        return "no_update"
    return a


def run_one(
    config: Config | dict[str, Any],
    csv_path: str | pathlib.Path | None = "results/results.csv",
) -> dict[str, Any]:
    """Execute a single attack experiment and log results.

    Steps:
      1. Loads the dataset.
      2. Simulates update epochs and search queries.
      3. Builds equality inference groups (T1 or T2).
      4. Builds adversary auxiliary knowledge for the chosen attack.
      5. Runs the attack (fma, via, pvia, lvia-k, lvia-s, no_update).
      6. Computes accuracy against ground truth.
      7. Logs result via :func:`evaluate.log_result`.

    Args:
        config:   :class:`~config.Config` object or dictionary containing hyperparameters.
        csv_path: Destination CSV file path (or None to skip logging).

    Returns:
        Dictionary containing the full result row logged to CSV.
    """
    start_time = time.perf_counter()

    # Extract configuration attributes
    if isinstance(config, dict):
        dataset_name = str(config.get("dataset", "synthetic"))
        raw_attack = str(config.get("attack", "fma"))
        equality_mode = str(config.get("equality", "T1")).upper()
        tau = int(config.get("tau", 10))
        lam = float(config.get("lambda", config.get("lambda_queries", 50.0)))
        gamma = float(config.get("gamma", 0.1))
        alpha = float(config.get("alpha", 0.8))
        eta = float(config.get("eta", 0.5))
        noise = float(config.get("noise", config.get("noise_n", 0.0)))
        seed = int(config.get("seed", 42))
        delta = float(config.get("delta", 0.6))
        keyword_count_cfg = config.get("keyword_count")
        tolerance = config.get("tolerance")
        mu0 = config.get("mu0")
    else:
        dataset_name = getattr(config, "dataset", "synthetic")
        raw_attack = getattr(config, "attack", "fma")
        equality_mode = str(getattr(config, "equality", "T1")).upper()
        tau = int(getattr(config, "tau", 10))
        lam = float(getattr(config, "lambda_queries", 50.0))
        gamma = float(getattr(config, "gamma", 0.1))
        alpha = float(getattr(config, "alpha", 0.8))
        eta = float(getattr(config, "eta", 0.5))
        noise = float(getattr(config, "noise_n", 0.0))
        seed = int(getattr(config, "seed", 42))
        delta = float(getattr(config, "delta", 0.6))
        keyword_count_cfg = getattr(config, "keyword_count", None)
        tolerance = getattr(config, "tolerance", None)
        mu0 = getattr(config, "mu0", None)

    attack_name = _normalize_attack_name(raw_attack)

    # 1. Load dataset
    docs, vocab = load_dataset(dataset_name, seed=seed, keyword_count=keyword_count_cfg)
    keyword_count = len(vocab)
    rng = np.random.default_rng(seed)

    # 2. Prepare simulation config
    sim_cfg = Config(
        seed=seed,
        dataset=dataset_name,
        keyword_count=keyword_count,
        tau=tau,
        lambda_queries=lam,
        gamma=gamma,
        alpha=alpha,
        eta=eta,
        noise_n=noise,
        equality=equality_mode,
        delta=delta,
        attack=attack_name,
    )

    # 3. Simulate updates & queries, build auxiliary data, and run attack
    if attack_name == "fma":
        if dataset_name == "obvious":
            true_dist = make_obvious_query_dist(vocab, tau)
        else:
            dist_mode = "trends" if dataset_name == "trends" else "synthetic"
            true_dist = generate_query_distribution(vocab, tau, rng, mode=dist_mode)

        aux_query_dist = make_aux_query_dist(true_dist, noise_n=noise, rng=rng)
        aux = Aux(query_dist=aux_query_dist)
        records, gt, events = simulate(docs, sim_cfg, query_dist=true_dist)
        eq = build_equality(records, mode=equality_mode, delta=delta)

        tol = float(tolerance) if tolerance is not None else 0.0
        preds = fma.run(records, aux, eq, tolerance=tol)

    elif attack_name == "via":
        records, gt, events = simulate(docs, sim_cfg)
        eq = build_equality(records, mode=equality_mode, delta=delta)
        aux = Aux(known_docs=docs)
        preds = via.run(records, aux, eq, events)

    elif attack_name == "pvia":
        records, gt, events = simulate(docs, sim_cfg)
        eq = build_equality(records, mode=equality_mode, delta=delta)
        # alpha must be strictly in (0, 1)
        valid_alpha = alpha if 0.0 < alpha < 1.0 else 0.8
        known_ids = make_aux_known_docs(events, alpha=valid_alpha, rng=rng)
        aux = Aux(known_docs=known_ids)
        preds = pvia.run(records, aux, eq, events, alpha=valid_alpha)

    elif attack_name in ("lvia-k", "lvia_k"):
        records, gt, events = simulate(docs, sim_cfg)
        eq = build_equality(records, mode=equality_mode, delta=delta)
        dist = make_aux_data_dist(docs, tau=tau, eta=1.0, rng=rng)
        aux = Aux(data_dist=dist)
        preds = lvia.run(records, aux, eq, mode="K")

    elif attack_name in ("lvia-s", "lvia_s"):
        records, gt, events = simulate(docs, sim_cfg)
        eq = build_equality(records, mode=equality_mode, delta=delta)
        valid_eta = eta if 0.0 < eta <= 1.0 else 0.5
        dist = make_aux_data_dist(docs, tau=tau, eta=valid_eta, rng=rng)
        aux = Aux(data_dist=dist)
        preds = lvia.run(records, aux, eq, mode="S", eta=valid_eta)

    elif attack_name in ("no_update", "no-update-pvia", "no-update-lvia"):
        records, gt, events = simulate(docs, sim_cfg)
        eq = build_equality(records, mode=equality_mode, delta=delta)
        if mu0 is None:
            mu0 = len(docs) // 2
        if attack_name == "no-update-lvia":
            valid_eta = eta if 0.0 < eta <= 1.0 else 0.5
            dist = make_aux_data_dist(docs, tau=tau, eta=valid_eta, rng=rng)
            aux = Aux(data_dist=dist)
            tolerance_value = float(tolerance) if tolerance is not None else 0.15
            preds = no_update.no_update_lvia(
                records, aux, eq, theta=tolerance_value, mu0=int(mu0)
            )
        else:
            snapshot = make_aux_pre_snapshot(docs, mu0=int(mu0))
            aux = Aux(pre_snapshot=snapshot)
            if attack_name == "no-update-pvia":
                preds = no_update.no_update_pvia(
                    records, aux, eq, alpha=alpha, mu0=int(mu0)
                )
            else:
                preds = no_update.run(records, aux, eq, mu0=int(mu0))

    else:
        raise ValueError(f"Unknown attack: {attack_name!r}")

    # 4. Compute accuracy
    acc = evaluate.accuracy(preds, gt)
    correct = sum(1 for qid, kw in gt.items() if preds.get(qid) == kw)
    total = len(gt)
    runtime_s = round(time.perf_counter() - start_time, 4)

    # 5. Format result row
    row_dict = {
        "dataset": dataset_name,
        "keyword_count": keyword_count,
        "attack": attack_name,
        "equality": equality_mode,
        "tau": tau,
        "lambda": lam,
        "gamma": gamma,
        "alpha": alpha,
        "eta": eta,
        "noise": noise,
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 6),
        "runtime_s": runtime_s,
        "seed": seed,
    }

    # 6. Log result
    if csv_path is not None:
        evaluate.log_result(row_dict, csv_path)

    return row_dict


# ---------------------------------------------------------------------------
# CSV Deduplication Helper
# ---------------------------------------------------------------------------

def _run_signature(
    dataset: str,
    attack: str,
    equality: str,
    tau: int | float,
    lam: int | float,
    gamma: float,
    alpha: float,
    eta: float,
    noise: float,
    seed: int,
) -> tuple:
    """Generate canonical key tuple matching an experiment run in CSV."""
    return (
        str(dataset).strip().lower(),
        _normalize_attack_name(str(attack)),
        str(equality).strip().upper(),
        int(tau),
        float(lam),
        round(float(gamma), 6),
        round(float(alpha), 6),
        round(float(eta), 6),
        round(float(noise), 6),
        int(seed),
    )


def read_completed_runs(csv_path: str | pathlib.Path) -> set[tuple]:
    """Parse existing CSV and return set of completed run signatures."""
    p = pathlib.Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()

    completed: set[tuple] = set()
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sig = _run_signature(
                    dataset=row.get("dataset", ""),
                    attack=row.get("attack", ""),
                    equality=row.get("equality", ""),
                    tau=int(float(row.get("tau", 0))),
                    lam=float(row.get("lambda", 0.0)),
                    gamma=float(row.get("gamma", 0.0)),
                    alpha=float(row.get("alpha", 0.0)),
                    eta=float(row.get("eta", 0.0)),
                    noise=float(row.get("noise", 0.0)),
                    seed=int(float(row.get("seed", 0))),
                )
                completed.add(sig)
            except (ValueError, TypeError):
                continue
    return completed


# ---------------------------------------------------------------------------
# Grid Runner
# ---------------------------------------------------------------------------

def _as_list(val: Any, default: list[Any]) -> list[Any]:
    """Wrap scalar values in a list or return default if None."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def run_grid(
    grid: dict[str, Any] | str | pathlib.Path,
    n_seeds: int | list[int] = 1,
    csv_path: str | pathlib.Path = "results/results.csv",
    dry_run: bool = False,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run all combinations of hyperparameters in a grid across seeds.

    Skips any (hyperparameter combination, seed) that already exists in *csv_path*.
    Displays a progress bar during execution.

    Args:
        grid:     Dictionary or path to YAML file specifying grid lists for:
                  tau, lambda, gamma, alpha, eta, noise, equality, dataset, attack.
        n_seeds:  Number of integer seeds [0 .. n_seeds-1] or explicit list of seeds.
        csv_path: Path to CSV output file.
        dry_run:  If True, prints planned runs without executing them.
        verbose:  If True, prints informative messages to stdout.

    Returns:
        List of result dictionaries for runs (or planned runs in dry_run mode).
    """
    # 1. Load grid if given as path or preset name
    if isinstance(grid, (str, pathlib.Path)):
        grid_str = str(grid)
        if grid_str in PAPER_GRIDS:
            grid_str = PAPER_GRIDS[grid_str]
        path = pathlib.Path(grid_str)
        if not path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid}")
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                grid_dict = yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required to load YAML grid files. Install with 'pip install pyyaml'.")
    else:
        grid_dict = grid

    # 2. Extract lists for Cartesian product
    datasets = _as_list(grid_dict.get("dataset"), ["synthetic"])
    attacks = _as_list(grid_dict.get("attack") or grid_dict.get("attacks"), ["fma"])
    equalities = _as_list(grid_dict.get("equality"), ["T1"])
    taus = _as_list(grid_dict.get("tau"), [10])
    lambdas = _as_list(grid_dict.get("lambda") or grid_dict.get("lambda_queries"), [50.0])
    gammas = _as_list(grid_dict.get("gamma"), [0.1])
    alphas = _as_list(grid_dict.get("alpha"), [0.8])
    etas = _as_list(grid_dict.get("eta"), [0.5])
    noises = _as_list(grid_dict.get("noise") or grid_dict.get("noise_n"), [0.0])

    # Seeds
    if isinstance(n_seeds, int):
        seeds = list(range(n_seeds))
    elif isinstance(n_seeds, (list, tuple)):
        seeds = list(n_seeds)
    elif "seeds" in grid_dict:
        seeds = _as_list(grid_dict["seeds"], [0])
    else:
        seeds = [0]

    # 3. Read completed runs from CSV
    completed = read_completed_runs(csv_path)

    # 4. Generate all planned run configs
    all_runs: list[dict[str, Any]] = []
    unskipped_runs: list[dict[str, Any]] = []

    for ds, att, eq, tau_val, lam_val, gam_val, alp_val, eta_val, nse_val, seed_val in itertools.product(
        datasets, attacks, equalities, taus, lambdas, gammas, alphas, etas, noises, seeds
    ):
        cfg_dict = {
            "dataset": ds,
            "attack": att,
            "equality": eq,
            "tau": int(tau_val),
            "lambda": float(lam_val),
            "gamma": float(gam_val),
            "alpha": float(alp_val),
            "eta": float(eta_val),
            "noise": float(nse_val),
            "seed": int(seed_val),
        }
        sig = _run_signature(
            dataset=ds,
            attack=att,
            equality=eq,
            tau=tau_val,
            lam=lam_val,
            gamma=gam_val,
            alpha=alp_val,
            eta=eta_val,
            noise=nse_val,
            seed=seed_val,
        )
        is_completed = sig in completed
        cfg_dict["_completed"] = is_completed
        all_runs.append(cfg_dict)
        if not is_completed:
            unskipped_runs.append(cfg_dict)

    if verbose:
        print(f"Total runs: {len(all_runs)} | Already completed: {len(all_runs) - len(unskipped_runs)} | To run: {len(unskipped_runs)}")

    # 5. Dry-run mode
    if dry_run:
        print("\n--- [DRY-RUN] Planned Experiment Runs ---")
        for i, run in enumerate(all_runs, 1):
            status = "SKIPPED (already in CSV)" if run["_completed"] else "TO RUN"
            print(
                f"[{i}/{len(all_runs)}] [{status}] dataset={run['dataset']} attack={run['attack']} "
                f"equality={run['equality']} tau={run['tau']} lambda={run['lambda']} "
                f"gamma={run['gamma']} alpha={run['alpha']} eta={run['eta']} "
                f"noise={run['noise']} seed={run['seed']}"
            )
        # Return cleaned configs
        clean_runs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_runs]
        return clean_runs

    # 6. Execute unskipped runs with progress bar
    if not unskipped_runs:
        if verbose:
            print("All runs in grid already completed.")
        return []

    results: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(unskipped_runs, desc="Experiments", unit="run")
    except ImportError:
        # Simple fallback text progress bar
        def _simple_progress(seq: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            total = len(seq)
            for idx, item in enumerate(seq, 1):
                if verbose:
                    print(f"Progress: [{idx}/{total}] Running {item['attack']} on {item['dataset']} (seed={item['seed']})...")
                yield item
        iterator = _simple_progress(unskipped_runs)

    for run_cfg in iterator:
        cfg = {k: v for k, v in run_cfg.items() if not k.startswith("_")}
        res = run_one(cfg, csv_path=csv_path)
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DSSE leakage-abuse attack experiments and parameter grids."
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Path to YAML grid configuration file, or paper grid preset name.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML configuration file for a single experiment.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run (default: 1).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/results.csv",
        help="Path to results CSV (default: results/results.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned runs without executing them.",
    )
    # Direct single-run arguments
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name.")
    parser.add_argument("--attack", type=str, default="fma", help="Attack: fma, via, pvia, lvia-k, lvia-s, no_update, no-update-pvia, no-update-lvia.")
    parser.add_argument("--equality", type=str, default="T1", help="Equality mode: T1 or T2.")
    parser.add_argument("--tau", type=int, default=3, help="Number of epochs/intervals.")
    parser.add_argument("--lambda", dest="lambda_queries", type=float, default=50.0, help="Poisson query rate.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Document deletion rate.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Known document fraction (PVIA).")
    parser.add_argument("--eta", type=float, default=0.5, help="Sampling fraction (LVIA-S).")
    parser.add_argument("--noise", type=float, default=0.0, help="Query distribution noise (FMA).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.grid and args.config:
        raise ValueError("--grid and --config cannot be used together.")
    if args.grid:
        run_grid(
            grid=args.grid,
            n_seeds=args.n_seeds,
            csv_path=args.csv_path,
            dry_run=args.dry_run,
        )
    elif args.config:
        from config import load_config

        cfg = load_config(args.config)
        if args.dry_run:
            print("[DRY-RUN] Single experiment:", cfg)
        else:
            result = run_one(cfg, csv_path=args.csv_path)
            print(f"Result: {result}")
    else:
        cfg = {
            "dataset": args.dataset,
            "attack": args.attack,
            "equality": args.equality,
            "tau": args.tau,
            "lambda": args.lambda_queries,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "eta": args.eta,
            "noise": args.noise,
            "seed": args.seed,
        }
        if args.dry_run:
            print("[DRY-RUN] Single experiment:", cfg)
        else:
            res = run_one(cfg, csv_path=args.csv_path)
            print(f"Result: {res}")


if __name__ == "__main__":
    main()
