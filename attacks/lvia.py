"""Length-based Volumetric Inference Attack (LVIA).

Reference: Section 6.2 of 'Leakage-Abuse Attacks Against Forward and
Backward Private Searchable Symmetric Encryption' (CCS 2023).

LVIA exploits data-distribution knowledge (keyword occurrence frequencies)
without requiring known files:
  - Mode 'K' (known-data): The adversary possesses exact per-interval keyword
    counts in the database (eta = 1.0, theta = 0).
  - Mode 'S' (sampled-data): The adversary observes a sampled fraction eta
    of the database; keyword counts are scaled by 1/eta, and candidates are
    selected within a small relative response-length window +/- theta.

For each query and interval:
  1. Compute expected response length for each keyword:
       expected_rlen(w, t) = sum_{i=0}^t scaled_count(w, i)
  2. Candidates in interval t are keywords satisfying:
       |expected_rlen(w, t) - observed_rlen| <= theta
  3. Across intervals, queries belonging to the same equality group pool
     their candidate sets. The group is mapped to the keyword appearing in
     the most per-interval candidate sets (ties broken by minimal cumulative
     distance).
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
    mode: str = "K",
    *,
    eta: float | None = None,
    theta: float | None = None,
) -> dict[int, str | None]:
    """Run the Length-based Volumetric Inference Attack (LVIA).

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed across
                  the simulation.
        aux:      Adversary auxiliary knowledge. Must contain ``data_dist``
                  (mapping ``{interval -> {keyword -> count}}``).
        equality: :class:`~equality.EqualityResult` grouping queries by keyword.
        mode:     ``'K'`` for exact known-data counts; ``'S'`` for counts
                  estimated from a sampled fraction eta.
        eta:      Sampling fraction in (0, 1] when mode is 'S'. If None, reads
                  from ``aux.eta`` or defaults to 0.5.
        theta:    Tolerance window for candidate response length matching.
                  - For mode 'K': defaults to 0.0 (exact match).
                  - For mode 'S': defaults to a relative window of 15% of the
                    observed response length (min 2.0). If theta < 1.0,
                    treated as relative fraction; if theta >= 1.0, treated
                    as absolute count slack. The CCS 2023 paper leaves theta
                    as an open heuristic; a 15% relative window matches
                    empirical sampling variance well.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` when
        unresolved).

    Raises:
        ValueError: If mode is not 'K' or 'S', or if eta is not in (0, 1].
    """
    mode_upper = mode.strip().upper()
    if mode_upper not in ("K", "S"):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'K' or 'S'.")

    if mode_upper == "S":
        if eta is None:
            eta = getattr(aux, "eta", 0.5)
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"eta must be in the interval (0, 1], got {eta!r}.")
    else:
        eta = 1.0

    # Initialise all queries to unresolved (None).
    predictions: dict[int, str | None] = {rec.query_id: None for rec in records}

    if not records or aux.data_dist is None:
        return predictions

    # -----------------------------------------------------------------------
    # Step 1: Compute cumulative expected response length per interval
    # -----------------------------------------------------------------------
    # data_dist: interval -> {keyword: count}
    intervals = sorted(aux.data_dist.keys())
    vocab = sorted({kw for t in intervals for kw in aux.data_dist[t]})
    if not vocab:
        return predictions

    # cum_expected: interval -> keyword -> cumulative live matching count
    # Since rlen at interval t reflects documents accumulated up to t:
    cum_expected: dict[int, dict[str, float]] = {}
    running_counts: dict[str, float] = defaultdict(float)
    for t in intervals:
        cum_expected[t] = {}
        for kw in vocab:
            raw_count = aux.data_dist[t].get(kw, 0)
            scaled = (raw_count / eta) if mode_upper == "S" else float(raw_count)
            running_counts[kw] += scaled
            cum_expected[t][kw] = running_counts[kw]

    # -----------------------------------------------------------------------
    # Step 2: Group records by equality group and interval
    # -----------------------------------------------------------------------
    # group_interval_records: gid -> interval -> list of LeakageRecord
    group_interval_records: dict[int, dict[int, list[LeakageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        gid = equality.group_of.get(r.query_id)
        if gid is not None:
            group_interval_records[gid][r.interval].append(r)

    # -----------------------------------------------------------------------
    # Step 3: Per-interval candidate sets & pooling across intervals
    # -----------------------------------------------------------------------
    group_resolved: dict[int, str] = {}

    for gid, int_map in group_interval_records.items():
        votes: dict[str, int] = defaultdict(int)
        distances: dict[str, float] = defaultdict(float)

        for t, recs in int_map.items():
            if t not in cum_expected:
                continue

            observed_rlen = recs[0].rlen

            # Determine tolerance window theta
            if mode_upper == "K":
                th = 0.0 if theta is None else float(theta)
            else:
                if theta is None:
                    # Choice documented: 15% relative window, min 2.0
                    th = max(2.0, 0.15 * observed_rlen)
                elif theta < 1.0:
                    th = max(1.0, float(theta) * observed_rlen)
                else:
                    th = float(theta)

            cands_in_t: set[str] = set()
            for kw in vocab:
                exp = cum_expected[t][kw]
                dist = abs(exp - observed_rlen)
                if dist <= th:
                    cands_in_t.add(kw)
                distances[kw] += dist

            for kw in cands_in_t:
                votes[kw] += 1

        if not votes:
            continue

        max_votes = max(votes.values())
        if max_votes == 0:
            continue

        top_kws = [kw for kw, v in votes.items() if v == max_votes]
        if len(top_kws) == 1:
            group_resolved[gid] = top_kws[0]
        else:
            # Break ties by minimal cumulative distance to expected response length
            best_kw = min(top_kws, key=lambda kw: distances[kw])
            group_resolved[gid] = best_kw

    # -----------------------------------------------------------------------
    # Step 4: Map resolved group keyword to individual queries
    # -----------------------------------------------------------------------
    for r in records:
        gid = equality.group_of.get(r.query_id)
        if gid is not None and gid in group_resolved:
            predictions[r.query_id] = group_resolved[gid]

    return predictions
