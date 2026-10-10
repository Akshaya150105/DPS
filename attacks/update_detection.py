"""Infer client deletion behavior from update and response lengths.

This is the paper's update-type detection observation presented as a small
experiment extension; the paper motivates the capability but does not provide
this exact accuracy experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from equality import EqualityResult
from types_ import LeakageRecord


@dataclass(frozen=True)
class UpdateEstimate:
    """Most recent per-keyword insertion/deletion estimates and class label."""

    insertions: int
    deletions: int
    prediction: str


def detect_deletions(
    records: list[LeakageRecord],
    equality: EqualityResult,
) -> dict[int, UpdateEstimate]:
    """Estimate ``a`` and ``d`` for each equality group.

    The latest query by (timestamp, query_id) is used as the cumulative
    estimate for a keyword. It must satisfy the leakage parity identity
    ``a=(u+n)/2`` and ``d=(u-n)/2``. Groups are classified as ``"client
    deletes"`` if the estimated deletion count is positive, and
    ``"insert only"`` otherwise.
    """
    grouped_records: dict[int, list[LeakageRecord]] = {}
    for record in records:
        group_id = equality.group_of.get(record.query_id)
        if group_id is None:
            continue
        grouped_records.setdefault(group_id, []).append(record)

    result: dict[int, UpdateEstimate] = {}
    for group_id, group in grouped_records.items():
        latest = max(group, key=lambda item: (item.timestamp, item.query_id))
        if (latest.ulen + latest.rlen) % 2 or (latest.ulen - latest.rlen) % 2:
            raise ValueError(
                f"Group {group_id} has inconsistent leakage parity: "
                f"ulen={latest.ulen}, rlen={latest.rlen}."
            )
        insertions = (latest.ulen + latest.rlen) // 2
        deletions = (latest.ulen - latest.rlen) // 2
        if insertions < 0 or deletions < 0:
            raise ValueError(
                f"Group {group_id} has invalid lengths: "
                f"ulen={latest.ulen}, rlen={latest.rlen}."
            )
        result[group_id] = UpdateEstimate(
            insertions=insertions,
            deletions=deletions,
            prediction="client deletes" if deletions > 0 else "insert only",
        )
    return result


def run_gamma_experiment(
    docs: list,
    gammas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    *,
    seed: int = 42,
    tau: int = 3,
    lambda_queries: float = 200.0,
    equality_mode: str = "T1",
) -> list[dict[str, float | int]]:
    """Compare inferred delete/no-delete labels against simulator truth by gamma."""
    from config import Config
    from equality import build_equality
    from simulator import simulate

    results: list[dict[str, float | int]] = []
    for gamma in gammas:
        config = Config(
            seed=seed,
            tau=tau,
            lambda_queries=lambda_queries,
            gamma=gamma,
            equality=equality_mode,
        )
        records, ground_truth, events = simulate(docs, config)
        equality = build_equality(records, mode=equality_mode)
        estimates = detect_deletions(records, equality)
        true_deletions: dict[int, bool] = {}
        group_keywords: dict[int, set[str]] = {}
        for record in records:
            group_id = equality.group_of.get(record.query_id)
            if group_id is not None:
                group_keywords.setdefault(group_id, set()).add(ground_truth[record.query_id])
        deleted_keywords = {event.keyword for event in events if event.op == "del"}
        for group_id, keywords in group_keywords.items():
            true_deletions[group_id] = bool(keywords & deleted_keywords)

        correct = total = 0
        for group_id, estimate in estimates.items():
            predicted_deletes = estimate.prediction == "client deletes"
            actual_deletes = true_deletions.get(group_id, False)
            correct += int(predicted_deletes == actual_deletes)
            total += 1
        results.append({
            "gamma": gamma,
            "groups": total,
            "correct": correct,
            "accuracy": correct / total if total else 1.0,
        })
    return results


if __name__ == "__main__":
    from synthetic import make_synthetic

    for row in run_gamma_experiment(make_synthetic(seed=42)):
        print(
            f"gamma={row['gamma']:.2f} "
            f"accuracy={row['accuracy']:.3f} "
            f"({row['correct']}/{row['groups']} groups)"
        )
