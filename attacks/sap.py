"""Simplified frequency attack using equality groups and query distributions."""

from __future__ import annotations

import math
from collections import Counter

from equality import EqualityResult
from types_ import Aux, LeakageRecord


def run(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
) -> dict[int, str | None]:
    """Recover uniquely best-fitting equality groups from frequency leakage.

    For each equality group, this attack compares its observed count in each
    interval with ``interval_query_count * keyword_probability`` and selects
    the keyword with the smallest total absolute error. Ties are left
    unresolved. Unlike FMA, it does not build candidate sets or refine groups
    across intervals; it uses only equality groups and ``aux.query_dist``.
    """
    predictions: dict[int, str | None] = {record.query_id: None for record in records}
    if not records or not aux.query_dist:
        return predictions

    known_qids = set(predictions)
    by_interval: dict[int, list[LeakageRecord]] = {}
    group_counts: dict[int, Counter[int]] = {}
    for record in records:
        by_interval.setdefault(record.interval, []).append(record)
        group_id = equality.group_of.get(record.query_id)
        if group_id is not None:
            group_counts.setdefault(group_id, Counter())[record.interval] += 1

    total_queries = {interval: len(interval_records) for interval, interval_records in by_interval.items()}
    intervals_by_group: dict[int, set[int]] = {
        group_id: {interval for interval, count in counts.items() if count > 0}
        for group_id, counts in group_counts.items()
    }
    interval_distribution: dict[int, dict[str, float]] = {}
    for interval in total_queries:
        raw = aux.query_dist.get(interval)
        if not raw:
            continue
        probabilities = {str(keyword): float(probability) for keyword, probability in raw.items()}
        if any(not math.isfinite(probability) or probability < 0.0 for probability in probabilities.values()):
            raise ValueError(f"query_dist[{interval}] must contain finite, non-negative probabilities.")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise ValueError(f"query_dist[{interval}] must have positive total probability.")
        interval_distribution[interval] = {
            keyword: probability / total for keyword, probability in probabilities.items()
        }

    resolved: dict[int, str] = {}
    for group_id, counts in group_counts.items():
        observed_intervals = sorted(
            interval
            for interval in intervals_by_group[group_id]
            if interval in interval_distribution
        )
        if not observed_intervals:
            continue

        vocab = set.intersection(*(
            set(interval_distribution[interval])
            for interval in observed_intervals
        ))
        if not vocab:
            continue
        scores = {
            keyword: sum(
                abs(
                    counts[interval]
                    - total_queries[interval] * interval_distribution[interval][keyword]
                )
                for interval in observed_intervals
            )
            for keyword in vocab
        }
        best_score = min(scores.values())
        winners = [keyword for keyword, score in scores.items() if math.isclose(score, best_score, abs_tol=1e-12)]
        if len(winners) == 1:
            resolved[group_id] = winners[0]

    for query_id, group_id in equality.group_of.items():
        if query_id in known_qids:
            predictions[query_id] = resolved.get(group_id)
    return predictions
