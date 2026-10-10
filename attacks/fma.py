"""Frequency Matching Attack (FMA) against forward-private SSE.

Reference: Section 4.1 of 'Leakage-Abuse Attacks Against Forward and
Backward Private Searchable Symmetric Encryption' (CCS 2023).

The attack exploits the *query distribution* leakage: when the adversary
knows (or estimates) how often each keyword is queried per interval, they
can match equality groups to keywords by comparing the observed group size
with the expected count under the distribution.

Algorithm outline (per-interval phase then cross-interval refinement):
  1. Split records into per-interval buckets.
  2. In each interval t, for each equality group G in that interval:
       group_size is its observed frequency.
  3. A group's candidates are keywords w minimizing
       abs(group_size - total_queries_in_interval * p_w)
     using aux.query_dist, keeping ties (with an option to keep keywords
     within a tolerance for noisy distributions).
  4. A group with exactly one candidate is recovered.
  5. For the others, link the same equality group across intervals and
     intersect candidate sets.
  6. A group with one remaining candidate is recovered.
  Query IDs in recovered groups map to the keyword; unresolved map to None.

This module never imports or reads the true keyword labels.
"""

from __future__ import annotations

from collections import defaultdict

from equality import EqualityResult
from types_ import Aux, LeakageRecord


def run(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
    *,
    tolerance: float = 0.0,
) -> dict[int, str | None]:
    """Run the Frequency Matching Attack and return per-query predictions.

    Args:
        records:   All :class:`~types_.LeakageRecord` objects from the simulation.
        aux:       Adversary auxiliary knowledge. Must contain ``query_dist``
                   (``{interval -> {keyword: probability}}``). If absent or
                   empty, all predictions are ``None``.
        equality:  :class:`~equality.EqualityResult` grouping same-keyword
                   queries together.
        tolerance: Extra slack added to the minimum score when building
                   candidate sets; set > 0 for noisy distributions.
                   When tolerance in (0, 1], treated as frequency slack
                   (tolerance * total_queries_in_interval); otherwise count slack.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` when the
        attack cannot recover the keyword for that query).
    """
    # Initialise every query to unresolved (None).
    predictions: dict[int, str | None] = {rec.query_id: None for rec in records}

    if not records or aux.query_dist is None:
        return predictions

    query_dist: dict[int, dict[str, float]] = aux.query_dist

    # -----------------------------------------------------------------------
    # Step 1 – split records by interval
    # -----------------------------------------------------------------------
    by_interval: dict[int, list[LeakageRecord]] = defaultdict(list)
    for rec in records:
        by_interval[rec.interval].append(rec)

    # -----------------------------------------------------------------------
    # Step 2-3 – per-interval frequency matching
    # -----------------------------------------------------------------------
    # group_id -> list of candidate keyword sets (one set per interval where
    # the group appeared).
    group_candidate_sets: dict[int, list[set[str]]] = defaultdict(list)

    for t, interval_records in sorted(by_interval.items()):
        dist = query_dist.get(t)
        if not dist:
            continue

        sampled_records = [r for r in interval_records if not getattr(r, "is_probe", False)]
        eval_records = sampled_records if sampled_records else interval_records

        total_queries = len(eval_records)
        if total_queries == 0:
            continue

        # Collect group members for this interval.
        gid_to_records: dict[int, list[LeakageRecord]] = defaultdict(list)
        for rec in eval_records:
            gid = equality.group_of.get(rec.query_id)
            if gid is not None:
                gid_to_records[gid].append(rec)

        for gid, grp_records in gid_to_records.items():
            # Step 2: In each interval, each equality group's size is observed frequency.
            group_size = len(grp_records)

            # Step 3: Candidate keywords w minimizing
            # abs(group_size - total_queries_in_interval * p_w).
            scores: dict[str, float] = {
                w: abs(group_size - total_queries * p) for w, p in dist.items()
            }
            if not scores:
                continue

            min_score = min(scores.values())
            slack = (
                tolerance * total_queries
                if (0.0 < tolerance <= 1.0)
                else tolerance
            )
            threshold = min_score + slack + 1e-9

            candidates: set[str] = {w for w, s in scores.items() if s <= threshold}
            group_candidate_sets[gid].append(candidates)

    # -----------------------------------------------------------------------
    # Step 4 – a group with exactly one candidate is recovered
    # -----------------------------------------------------------------------
    group_resolved: dict[int, str] = {}
    unresolved_groups: set[int] = set()

    for gid, cand_list in group_candidate_sets.items():
        if not cand_list:
            continue
        # Check singletons across intervals where the group appeared
        singletons = [next(iter(c)) for c in cand_list if len(c) == 1]
        if singletons and all(s == singletons[0] for s in singletons):
            group_resolved[gid] = singletons[0]
        else:
            unresolved_groups.add(gid)

    # -----------------------------------------------------------------------
    # Step 5-6 – link equality group across intervals, intersect candidates
    # -----------------------------------------------------------------------
    for gid in unresolved_groups:
        cand_list = group_candidate_sets[gid]
        if not cand_list:
            continue
        combined = cand_list[0].copy()
        for s in cand_list[1:]:
            combined &= s
            if not combined:
                # If intersection became empty (noise / distribution drift),
                # fall back to the smallest candidate set.
                combined = min(cand_list, key=len)
                break

        # Step 6: a group with one remaining candidate is recovered
        if len(combined) == 1:
            group_resolved[gid] = next(iter(combined))

    # Assign predicted keyword to all query_ids in resolved equality groups.
    for qid, gid in equality.group_of.items():
        kw = group_resolved.get(gid)
        if kw is not None:
            predictions[qid] = kw

    return predictions
