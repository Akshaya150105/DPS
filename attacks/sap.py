"""Search Access Pattern (SAP) / Frequency Matching Attack.

Reference:
Simon Oya and Florian Kerschbaum, 'Hiding the Access Pattern is Not Enough:
Exploiting Search Pattern Leakage in Searchable Symmetric Encryption' (USENIX Security 2021).

Simplified frequency attack against dynamic searchable symmetric encryption:
Uses only query equality groups and the adversary's auxiliary query distribution.
Matches equality groups to keywords by ranking observed group frequencies against
expected keyword probabilities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from equality import EqualityResult
from types_ import Aux, LeakageRecord


def run(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
) -> dict[int, str | None]:
    """Run the simplified SAP frequency matching attack.

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed across
                  the simulation.
        aux:      Adversary auxiliary knowledge. Must contain ``query_dist``
                  (``{interval -> {keyword: probability}}``).
        equality: :class:`~equality.EqualityResult` grouping queries believed to
                  have the same keyword.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` if unresolved).
    """
    predictions: dict[int, str | None] = {r.query_id: None for r in records}

    if not records or aux.query_dist is None or not equality.groups:
        return predictions

    # 1. Aggregate keyword probabilities across intervals
    kw_probs: dict[str, float] = defaultdict(float)
    dist = aux.query_dist
    num_intervals = max(1, len(dist))

    for interval_idx, kw_map in dist.items():
        for kw, p in kw_map.items():
            kw_probs[kw] += p / num_intervals

    if not kw_probs:
        return predictions

    # 2. Sort candidate keywords in descending order of expected frequency
    sorted_keywords = sorted(kw_probs.keys(), key=lambda w: kw_probs[w], reverse=True)

    # 3. Sort equality groups in descending order of observed size
    # Group size is the number of query IDs in the equality group
    sorted_groups = sorted(
        equality.groups.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    # 4. Map the rank of equality groups to the rank of keywords
    for rank, (group_id, query_ids) in enumerate(sorted_groups):
        if rank < len(sorted_keywords):
            matched_kw = sorted_keywords[rank]
            for qid in query_ids:
                if qid in predictions:
                    predictions[qid] = matched_kw

    return predictions
