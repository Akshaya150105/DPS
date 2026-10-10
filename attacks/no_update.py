"""Attacks without update knowledge (Section 6.3)."""

from __future__ import annotations

from collections import Counter, defaultdict

from equality import EqualityResult
from types_ import Aux, LeakageRecord


def _target_query(
    record: LeakageRecord,
    records_by_id: dict[int, LeakageRecord],
    equality: EqualityResult,
    mu0: int,
) -> LeakageRecord:
    """Return the earliest equality-linked query at or after the cutoff."""
    group_id = equality.group_of.get(record.query_id)
    group_ids = equality.groups.get(group_id, [record.query_id])
    linked = [records_by_id[qid] for qid in group_ids if qid in records_by_id]
    after_cutoff = [item for item in linked if item.timestamp >= mu0]
    return min(after_cutoff, key=lambda item: (item.timestamp, item.query_id)) if after_cutoff else record


def run(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
    mu0: int = 0,
) -> dict[int, str | None]:
    """Run the full-snapshot no-update attack from paper Section 6.3.

    The attacker uses only ``aux.pre_snapshot`` to calculate keyword document
    frequencies and document-volume sets. Ambiguous candidates remain None.
    """
    if mu0 is None:
        mu0 = getattr(aux, "mu0", 0)
    predictions: dict[int, str | None] = {record.query_id: None for record in records}
    if not records or aux.pre_snapshot is None:
        return predictions

    counts: Counter[str] = Counter()
    volumes_by_keyword: dict[str, set[int]] = defaultdict(set)
    for document in aux.pre_snapshot:
        volume = len(document.keywords)
        for keyword in document.keywords:
            counts[keyword] += 1
            volumes_by_keyword[keyword].add(volume)
    if not counts:
        return predictions

    records_by_id = {record.query_id: record for record in records}
    for record in records:
        target = _target_query(record, records_by_id, equality, mu0)
        insert_length = (target.ulen + target.rlen) / 2
        candidates = [
            keyword
            for keyword, count in counts.items()
            if insert_length >= count
            and all(
                count_for_volume <= 0 or volume in volumes_by_keyword[keyword]
                for volume, count_for_volume in target.fvol.items()
            )
        ]
        if len(candidates) == 1:
            predictions[record.query_id] = candidates[0]
    return predictions


def no_update_pvia(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
    alpha: float,
    mu0: int = 0,
) -> dict[int, str | None]:
    """Partial-snapshot no-update variant using volume-overlap ranking.

    ``alpha`` scales the known snapshot frequency bound. A unique maximum
    multiset overlap is assigned; ties remain unresolved.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1).")
    predictions: dict[int, str | None] = {record.query_id: None for record in records}
    if not records or aux.pre_snapshot is None:
        return predictions

    counts: Counter[str] = Counter()
    volumes_by_keyword: dict[str, Counter[int]] = defaultdict(Counter)
    for document in aux.pre_snapshot:
        volume = len(document.keywords)
        for keyword in document.keywords:
            counts[keyword] += 1
            volumes_by_keyword[keyword][volume] += 1
    if not counts:
        return predictions

    records_by_id = {record.query_id: record for record in records}
    for record in records:
        target = _target_query(record, records_by_id, equality, mu0)
        insert_length = (target.ulen + target.rlen) / 2
        candidates = [
            keyword for keyword, count in counts.items()
            if alpha * count <= insert_length
        ]
        if not candidates:
            continue

        def score(keyword: str) -> int:
            return sum(
                min(max(observed, 0), volumes_by_keyword[keyword][volume])
                for volume, observed in target.fvol.items()
            )

        scores = {keyword: score(keyword) for keyword in candidates}
        best_score = max(scores.values())
        winners = [keyword for keyword, value in scores.items() if value == best_score]
        if len(winners) == 1:
            predictions[record.query_id] = winners[0]
    return predictions


def no_update_lvia(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
    *,
    theta: float = 0.15,
    mu0: int = 0,
) -> dict[int, str | None]:
    """No-update LVIA using the per-interval auxiliary keyword distribution."""
    if theta < 0.0:
        raise ValueError("theta must be non-negative.")
    predictions: dict[int, str | None] = {record.query_id: None for record in records}
    if not records or not aux.data_dist:
        return predictions

    records_by_id = {record.query_id: record for record in records}
    for record in records:
        target = _target_query(record, records_by_id, equality, mu0)
        distribution = aux.data_dist.get(target.interval, {})
        if not distribution:
            continue
        insert_length = (target.ulen + target.rlen) / 2
        window = theta * max(1.0, insert_length) if theta < 1.0 else theta
        scores = {
            keyword: abs(count - insert_length)
            for keyword, count in distribution.items()
            if abs(count - insert_length) <= window
        }
        if scores:
            best_distance = min(scores.values())
            winners = [keyword for keyword, distance in scores.items() if distance == best_distance]
            if len(winners) == 1:
                predictions[record.query_id] = winners[0]
    return predictions
