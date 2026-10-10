"""Update-type detection attack: infer client deletions vs insert-only updates.

Reference & Extension Note:
In 'Leakage-Abuse Attacks Against Forward and Backward Private Searchable
Symmetric Encryption' (CCS 2023), the authors note that the relation
  u = a + d   (update length)
  n = a - d   (response length)
allows an adversary observing dynamic leakage to compute the number of
deletions as d = (u - n) / 2 and insertions as a = (u + n) / 2.
While the paper mentions this capability theoretically, it does not include
an empirical evaluation of update-type classification.

This module provides an empirical extension implementing and evaluating
this capability:
  1. For each equality group of queries, we compute (u, n) and estimate
     d = (u - n) / 2 and a = (u + n) / 2.
  2. If d > 0, we predict 'client deletes'; otherwise, 'insert only'.
  3. We evaluate classification accuracy against the ground-truth simulation
     log across deletion rates gamma in [0, 0.05, 0.1, 0.2].
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import pathlib
import sys
from typing import Any, Iterable, NamedTuple

# Ensure repository root is on sys.path when invoked directly
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from config import Config
from equality import EqualityResult, build_equality
from simulator import simulate
from synthetic import make_synthetic
from types_ import Event, LeakageRecord


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

class GroupEstimate(NamedTuple):
    """Estimated insertion count, deletion count, and update prediction."""

    a: float
    d: float
    prediction: str


class DetectionResult(tuple):
    """Result returned by :func:`detect_deletions`.

    Behaves as a 2-tuple ``(predictions, group_estimates)`` for clean unpacking:
      ``preds, ests = detect_deletions(records, equality)``

    Also exposes named attributes:
      - ``predictions``:       dict[group_id -> 'client deletes' | 'insert only']
      - ``group_estimates``:   dict[group_id -> (a, d)]
      - ``query_predictions``: dict[query_id -> 'client deletes' | 'insert only']
      - ``query_estimates``:   dict[query_id -> (a, d)]
    """

    predictions: dict[int, str]
    group_estimates: dict[int, tuple[float, float]]
    query_predictions: dict[int, str]
    query_estimates: dict[int, tuple[float, float]]

    def __new__(
        cls,
        predictions: dict[int, str],
        group_estimates: dict[int, tuple[float, float]],
        query_predictions: dict[int, str] | None = None,
        query_estimates: dict[int, tuple[float, float]] | None = None,
    ) -> DetectionResult:
        obj = super().__new__(cls, (predictions, group_estimates))
        obj.predictions = predictions
        obj.group_estimates = group_estimates
        obj.query_predictions = query_predictions or {}
        obj.query_estimates = query_estimates or {}
        return obj


# ---------------------------------------------------------------------------
# Core Update-Type Detection Function
# ---------------------------------------------------------------------------

def detect_deletions(
    records: list[LeakageRecord],
    equality: EqualityResult | dict[int, list[int]],
) -> DetectionResult:
    """Infer whether equality groups contain document deletions or are insert-only.

    For each query q in an equality group, we observe:
      u = q.ulen  (total operations touched = a + d)
      n = q.rlen  (current matching result set size = a - d)

    Solving for d:
      d = (u - n) / 2
      a = (u + n) / 2

    If d > 0, the adversary predicts 'client deletes'; otherwise 'insert only'.
    For each equality group, the estimated (a, d) counts reflect the latest
    observed query in that group.

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed in simulation.
        equality: :class:`~equality.EqualityResult` or dict mapping group_id -> list of query_ids.

    Returns:
        :class:`DetectionResult` tuple:
          - predictions: dict[group_id -> 'client deletes' | 'insert only']
          - group_estimates: dict[group_id -> (a, d)]
          - query_predictions: dict[query_id -> 'client deletes' | 'insert only']
          - query_estimates: dict[query_id -> (a, d)]
    """
    rec_map: dict[int, LeakageRecord] = {r.query_id: r for r in records}

    # Extract groups mapping (group_id -> list of query_ids)
    if hasattr(equality, "groups"):
        groups: dict[int, list[int]] = equality.groups
    elif isinstance(equality, dict):
        if equality and isinstance(next(iter(equality.values())), (list, tuple, set)):
            groups = {k: list(v) for k, v in equality.items()}
        else:
            # Assume inverted mapping query_id -> group_id
            inverted = defaultdict(list)
            for qid, gid in equality.items():
                inverted[gid].append(qid)
            groups = dict(inverted)
    else:
        raise TypeError(f"Unsupported equality type: {type(equality)}")

    query_predictions: dict[int, str] = {}
    query_estimates: dict[int, tuple[float, float]] = {}
    group_predictions: dict[int, str] = {}
    group_estimates: dict[int, tuple[float, float]] = {}

    for gid, qids in groups.items():
        max_d = 0.0
        max_a = 0.0
        latest_rec: LeakageRecord | None = None

        for qid in qids:
            rec = rec_map.get(qid)
            if rec is None:
                continue

            u = rec.ulen
            n = rec.rlen
            d = (u - n) / 2.0
            a = (u + n) / 2.0

            q_pred = "client deletes" if d > 0.0 else "insert only"
            query_predictions[qid] = q_pred
            query_estimates[qid] = (a, d)

            if latest_rec is None or rec.timestamp >= latest_rec.timestamp:
                latest_rec = rec
                max_d = d
                max_a = a

        # Group-level classification
        g_pred = "client deletes" if max_d > 0.0 else "insert only"
        group_predictions[gid] = g_pred
        group_estimates[gid] = (max_a, max_d)

    return DetectionResult(
        predictions=group_predictions,
        group_estimates=group_estimates,
        query_predictions=query_predictions,
        query_estimates=query_estimates,
    )


# ---------------------------------------------------------------------------
# Empirical Evaluation Experiment
# ---------------------------------------------------------------------------

def run_detection_experiment(
    gammas: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    n_seeds: int = 5,
    dataset_name: str = "synthetic",
    tau: int = 3,
    lam: float = 100.0,
    equality_mode: str = "T1",
    print_results: bool = True,
) -> dict[float, float]:
    """Run update-type detection across varying deletion rates gamma.

    Evaluates the accuracy of predicting 'client deletes' vs 'insert only'
    compared to the ground-truth simulation event log.

    Args:
        gammas:         List of deletion rates to evaluate (default: 0.0, 0.05, 0.1, 0.2).
        n_seeds:        Number of random seeds to average over (default: 5).
        dataset_name:   Dataset identifier (default: 'synthetic').
        tau:            Number of intervals (default: 3).
        lam:            Expected queries per interval (default: 100.0).
        equality_mode:  'T1' or 'T2' (default: 'T1').
        print_results:  If True, prints a formatted summary table.

    Returns:
        Dict mapping gamma -> mean classification accuracy across seeds.
    """
    gamma_list = list(gammas)
    mean_accuracies: dict[float, float] = {}
    std_accuracies: dict[float, float] = {}

    if print_results:
        print("\n" + "=" * 76)
        print("Update-Type Detection Experiment (Paper Extension)")
        print(f"Dataset: {dataset_name} | Tau: {tau} | Lambda: {lam} | Seeds: {n_seeds}")
        print("Evaluating classification: 'client deletes' (d > 0) vs 'insert only' (d == 0)")
        print("=" * 76)
        print(f"{'Gamma':<10} | {'Mean Acc':<12} | {'Std Dev':<10} | {'Avg Delete Groups':<18} | {'Avg Insert Groups':<18}")
        print("-" * 76)

    for gamma in gamma_list:
        seed_accs: list[float] = []
        del_group_counts: list[int] = []
        ins_group_counts: list[int] = []

        for seed in range(n_seeds):
            docs = make_synthetic(seed=seed)
            cfg = Config(
                seed=seed,
                tau=tau,
                lambda_queries=lam,
                gamma=gamma,
                equality=equality_mode,
            )
            records, gt, events = simulate(docs, cfg)
            eq = build_equality(records, mode=equality_mode)

            # Ground truth: set of keywords that had at least one deletion
            true_deleted_keywords = {e.keyword for e in events if e.op == "del"}

            # Run detection
            detection = detect_deletions(records, eq)

            correct = 0
            total = len(eq.groups)
            n_del_true = 0
            n_ins_true = 0

            for gid, qids in eq.groups.items():
                # Identify true keyword for the group (mode of queries in group)
                kws = [gt[qid] for qid in qids if qid in gt]
                if not kws:
                    continue
                true_kw = max(set(kws), key=kws.count)

                true_label = "client deletes" if true_kw in true_deleted_keywords else "insert only"
                if true_label == "client deletes":
                    n_del_true += 1
                else:
                    n_ins_true += 1

                pred_label = detection.predictions.get(gid)
                if pred_label == true_label:
                    correct += 1

            acc = correct / total if total > 0 else 1.0
            seed_accs.append(acc)
            del_group_counts.append(n_del_true)
            ins_group_counts.append(n_ins_true)

        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        mean_accuracies[gamma] = mean_acc
        std_accuracies[gamma] = std_acc

        avg_del = float(np.mean(del_group_counts))
        avg_ins = float(np.mean(ins_group_counts))

        if print_results:
            print(f"{gamma:<10.2f} | {mean_acc:<12.4f} | {std_acc:<10.4f} | {avg_del:<18.1f} | {avg_ins:<18.1f}")

    if print_results:
        print("=" * 76 + "\n")

    return mean_accuracies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run update-type detection experiment (deletion detection)."
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2],
        help="List of deletion rates gamma to evaluate (default: 0.0 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset name (default: synthetic).",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=3,
        help="Number of intervals (default: 3).",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_queries",
        type=float,
        default=100.0,
        help="Poisson query rate lambda (default: 100.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_detection_experiment(
        gammas=args.gammas,
        n_seeds=args.n_seeds,
        dataset_name=args.dataset,
        tau=args.tau,
        lam=args.lambda_queries,
        print_results=True,
    )


if __name__ == "__main__":
    main()
