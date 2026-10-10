"""Equality-inference algorithms (T1 prefix matching and T2 response similarity)."""

from __future__ import annotations

from dataclasses import dataclass, field

import leakage
from types_ import LeakageRecord


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union)
# ---------------------------------------------------------------------------

class UnionFind:
    """Path-compressed, union-by-rank disjoint set data structure.

    Attributes:
        parent: Maps each element to its parent (self if root).
        rank:   Approximate tree depth used to keep trees flat.
    """

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank:   dict[int, int] = {}

    def _ensure(self, x: int) -> None:
        """Register *x* as a singleton set if it has not been seen before."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        """Return the canonical root representative for *x* (path compression)."""
        self._ensure(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compress
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing *x* and *y* (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Attach smaller tree under larger tree.
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# EqualityResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class EqualityResult:
    """Output of :func:`build_equality`.

    Attributes:
        groups:       Dict mapping group_id -> list of query_ids in that group.
                      Each group_id is the smallest query_id in the group.
        group_of:     Dict mapping query_id -> group_id.
        linked_pairs: Dict mapping query_id -> next query_id (by timestamp) in
                      the same group, or ``None`` if it is the last in the group.
    """

    groups:       dict[int, list[int]] = field(default_factory=dict)
    group_of:     dict[int, int]       = field(default_factory=dict)
    linked_pairs: dict[int, int | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_equality(
    records: list[LeakageRecord],
    mode: str,
    delta: float = 0.6,
) -> EqualityResult:
    """Group search queries that are believed to be for the same keyword.

    Two inference strategies are supported:

    **T1** – *Token prefix matching*:
        Two queries are equal iff their token strings share the same prefix
        before the ``'|'`` separator.  This reflects the deterministic
        HMAC prefix emitted by T1 forward-private schemes.

    **T2** – *Response-set similarity*:
        Two queries are linked when
        ``rsp(fvol_a, fvol_b) >= delta``.
        To keep complexity manageable only queries in the **same or adjacent**
        intervals are compared.  All such linked pairs are fed into a
        Union-Find to produce transitive groups.

    Args:
        records: List of :class:`~types_.LeakageRecord` objects (any order).
        mode:    ``'T1'`` or ``'T2'`` (case-insensitive).
        delta:   RSP threshold for T2 linking (default 0.6, ignored in T1).

    Returns:
        An :class:`EqualityResult` with ``groups``, ``group_of``, and
        ``linked_pairs`` populated.

    Raises:
        ValueError: If *mode* is not ``'T1'`` or ``'T2'``.
    """
    mode_upper = mode.strip().upper()
    if mode_upper not in ("T1", "T2"):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'T1' or 'T2'.")

    uf = UnionFind()
    # Seed every query_id as its own set.
    for rec in records:
        uf._ensure(rec.query_id)

    if mode_upper == "T1":
        _link_t1(records, uf)
    else:
        _link_t2(records, uf, delta)

    return _build_result(records, uf)


# ---------------------------------------------------------------------------
# T1 strategy
# ---------------------------------------------------------------------------

def _link_t1(records: list[LeakageRecord], uf: UnionFind) -> None:
    """Group records with the same token prefix (before '|')."""
    # Map prefix -> first query_id seen with that prefix.
    prefix_to_root: dict[str, int] = {}

    for rec in sorted(records, key=lambda r: r.query_id):
        # If there is no '|' the whole token is the "prefix" (T2 fallback).
        prefix = rec.token.split("|")[0]
        if prefix in prefix_to_root:
            uf.union(prefix_to_root[prefix], rec.query_id)
        else:
            prefix_to_root[prefix] = rec.query_id


# ---------------------------------------------------------------------------
# T2 strategy
# ---------------------------------------------------------------------------

def _link_t2(records: list[LeakageRecord], uf: UnionFind, delta: float) -> None:
    """Link records whose RSP similarity >= delta, restricting to same/adjacent intervals."""
    # Bucket records by interval for O(n * bucket_size) comparisons.
    by_interval: dict[int, list[LeakageRecord]] = {}
    for rec in records:
        by_interval.setdefault(rec.interval, []).append(rec)

    all_intervals = sorted(by_interval)

    for i, t in enumerate(all_intervals):
        # Compare queries in interval t with those in t and t+1 only.
        adjacent = by_interval[t]
        if i + 1 < len(all_intervals):
            adjacent = adjacent + by_interval[all_intervals[i + 1]]

        current = by_interval[t]

        for j, rec_a in enumerate(current):
            for rec_b in adjacent:
                if rec_b.query_id <= rec_a.query_id:
                    # Avoid double-counting; only process (a, b) with b > a.
                    continue
                sim = leakage.rsp(rec_a.fvol, rec_b.fvol)
                if sim >= delta:
                    uf.union(rec_a.query_id, rec_b.query_id)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _build_result(records: list[LeakageRecord], uf: UnionFind) -> EqualityResult:
    """Convert a Union-Find into an EqualityResult."""
    # group_of: query_id -> root representative
    raw_group_of: dict[int, int] = {rec.query_id: uf.find(rec.query_id) for rec in records}

    # Normalise group IDs to the smallest query_id in each group.
    root_to_min: dict[int, int] = {}
    for qid, root in raw_group_of.items():
        if root not in root_to_min or qid < root_to_min[root]:
            root_to_min[root] = qid

    group_of:  dict[int, int]       = {qid: root_to_min[root] for qid, root in raw_group_of.items()}
    groups:    dict[int, list[int]] = {}
    for qid, gid in group_of.items():
        groups.setdefault(gid, []).append(qid)
    for lst in groups.values():
        lst.sort()

    # linked_pairs: each query -> the next query (by timestamp) in the same group.
    # "Next" is determined by the record's logical timestamp then query_id.
    id_to_rec: dict[int, LeakageRecord] = {rec.query_id: rec for rec in records}
    linked_pairs: dict[int, int | None] = {}
    for gid, members in groups.items():
        # Sort group members by (timestamp, query_id) to get chain order.
        ordered = sorted(members, key=lambda qid: (id_to_rec[qid].timestamp, qid))
        for k, qid in enumerate(ordered):
            linked_pairs[qid] = ordered[k + 1] if k + 1 < len(ordered) else None

    return EqualityResult(groups=groups, group_of=group_of, linked_pairs=linked_pairs)


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def equality_quality(
    equality: EqualityResult,
    ground_truth: dict[int, str],
) -> tuple[float, float]:
    """Compute pairwise precision and recall for equality inference.

    For each pair of query_ids (i, j) with i < j:
      - A pair is *predicted positive* iff ``group_of[i] == group_of[j]``.
      - A pair is *truly positive*    iff ``ground_truth[i] == ground_truth[j]``.

    Precision = TP / (TP + FP)   (fraction of predicted pairs that are correct)
    Recall    = TP / (TP + FN)   (fraction of true pairs that were found)

    Args:
        equality:     An :class:`EqualityResult` from :func:`build_equality`.
        ground_truth: Dict mapping query_id -> true keyword string.
                      (Used for scoring only; never available to the attacker.)

    Returns:
        ``(precision, recall)`` both in [0, 1].  Returns ``(1.0, 0.0)`` when
        there are no positive predictions, and ``(0.0, 1.0)`` when there are no
        true positive pairs.
    """
    qids = sorted(equality.group_of.keys())
    n = len(qids)

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = qids[i], qids[j]
            predicted = equality.group_of[qi] == equality.group_of[qj]
            actual    = ground_truth.get(qi) == ground_truth.get(qj)

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall
