"""Equality-inference algorithms (T1 prefix matching and T2 response similarity)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    forward_check=None,
) -> EqualityResult:
    """Group search queries that are believed to be for the same keyword.

    Five inference strategies are supported:

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

    **exact** – *Exact token equality* (e.g. BSigma):
        Two queries are equal iff their tokens compare equal (``==``).
        Works with any token type that supports equality.

    **partial** – *Shared first component* (e.g. Mitra, Janus++):
        Tokens are objects with a ``keyword_component`` attribute (or strings
        split on ``'|'``); two queries are grouped when their first component
        is identical.  Mirrors the keyword-derived prefix shared by Mitra
        tokens.

    **computational** – *Forward-derivability* (e.g. Fides):
        Accepts a *callable* ``forward_check(new_token, old_token) -> bool``
        that returns ``True`` when *old_token* is computable from *new_token*.
        Any pair for which ``forward_check`` returns ``True`` is merged into
        one equality group via Union-Find, giving transitive closure.

    Args:
        records:       List of :class:`~types_.LeakageRecord` objects (any order).
        mode:          ``'T1'``, ``'T2'``, ``'exact'``, ``'partial'``, or
                       ``'computational'`` (case-insensitive).
        delta:         RSP threshold for T2 linking (default 0.6, ignored in
                       all other modes).
        forward_check: Callable ``(new_token, old_token) -> bool`` required
                       when *mode* is ``'computational'``; ignored otherwise.

    Returns:
        An :class:`EqualityResult` with ``groups``, ``group_of``, and
        ``linked_pairs`` populated.

    Raises:
        ValueError: If *mode* is unrecognised, or if ``'computational'`` is
                    requested without a *forward_check* callable.
    """
    mode_upper = mode.strip().upper()
    _VALID_MODES = ("T1", "T2", "EXACT", "PARTIAL", "COMPUTATIONAL")
    if mode_upper not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {_VALID_MODES}."
        )
    if mode_upper == "COMPUTATIONAL" and forward_check is None:
        raise ValueError(
            "'computational' mode requires a forward_check callable."
        )

    uf = UnionFind()
    # Seed every query_id as its own set.
    for rec in records:
        uf._ensure(rec.query_id)

    if mode_upper == "T1":
        _link_t1(records, uf)
    elif mode_upper == "T2":
        _link_t2(records, uf, delta)
    elif mode_upper == "EXACT":
        _link_exact(records, uf)
    elif mode_upper == "PARTIAL":
        _link_partial(records, uf)
    else:  # COMPUTATIONAL
        _link_computational(records, uf, forward_check)

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
    """Link records whose RSP similarity >= delta, restricting to same/adjacent intervals.

    Probe queries are never linked to other queries in their own interval.

    Exact shortcuts keep this tractable at paper scale (10^5 queries):
      * Records in one interval with identical fvol ("a class") compare the
        same way against every other record, so similarity is computed once
        per pair of classes. A hit then links the members the rules allow:
        non-probe members across same-interval classes, all members across
        adjacent-interval classes.
      * rsp(A, B) <= min(|A|, |B|) / max(|A|, |B|), so class pairs whose
        sizes differ by more than that ratio allows are skipped.
      * Similarities for one class against a block of classes are computed
        together with numpy on dense volume vectors.
    """
    by_interval: dict[int, list[LeakageRecord]] = {}
    for rec in records:
        by_interval.setdefault(rec.interval, []).append(rec)

    # Dense volume axis shared by all intervals.
    volumes = sorted({v for rec in records for v in rec.fvol})
    vol_index = {v: k for k, v in enumerate(volumes)}

    class _Classes:
        """Distinct fvol classes of one interval, sorted by response size."""

        def __init__(self, recs: list[LeakageRecord]) -> None:
            by_sig: dict[frozenset, tuple[list[int], list[int]]] = {}
            for rec in sorted(recs, key=lambda r: r.query_id):
                sig = frozenset(rec.fvol.items())
                non_probes, probes = by_sig.setdefault(sig, ([], []))
                (probes if getattr(rec, "is_probe", False) else non_probes).append(rec.query_id)
            items = sorted(by_sig.items(), key=lambda kv: sum(c for _, c in kv[0]))
            self.sizes = np.array([sum(c for _, c in sig) for sig, _ in items], dtype=np.int64)
            self.non_probe = [np_ids[0] if np_ids else None for _, (np_ids, _) in items]
            # Non-probes in a class are mutually linked (when allowed), so the
            # first non-probe stands for all of them; probes stand alone.
            self.anchors = [([np_ids[0]] if np_ids else []) + p_ids for _, (np_ids, p_ids) in items]
            self.matrix = np.zeros((len(items), len(volumes)), dtype=np.int64)
            for row, (sig, (np_ids, _)) in enumerate(items):
                for v, c in sig:
                    self.matrix[row, vol_index[v]] = c
                # Same-interval links between identical non-probe records.
                if len(np_ids) > 1 and _rsp_sizes(self.sizes[row], self.sizes[row], self.sizes[row]) >= delta:
                    for q in np_ids[1:]:
                        uf.union(np_ids[0], q)

    def _rsp_sizes(inter, size_a, size_b):
        union = size_a + size_b - inter
        return np.where(union == 0, 1.0, inter / np.maximum(union, 1))

    classes = {t: _Classes(recs) for t, recs in by_interval.items()}
    prune = delta > 0

    def compare(a: "_Classes", b: "_Classes", same_interval: bool) -> None:
        for i in range(len(a.sizes)):
            sa = int(a.sizes[i])
            if prune:
                # Only classes with delta*sa <= sb <= sa/delta can reach the threshold.
                lo = int(np.searchsorted(b.sizes, delta * sa - 1e-9, side="left"))
                hi = int(np.searchsorted(b.sizes, sa / delta + 1e-9, side="right"))
            else:
                lo, hi = 0, len(b.sizes)
            if same_interval:
                if a.non_probe[i] is None:
                    continue
                lo = max(lo, i + 1)
            if lo >= hi:
                continue
            inter = np.minimum(b.matrix[lo:hi], a.matrix[i]).sum(axis=1)
            sims = _rsp_sizes(inter, sa, b.sizes[lo:hi])
            for off in np.nonzero(sims >= delta)[0]:
                j = lo + int(off)
                if same_interval:
                    # Probes never link within their own interval.
                    if b.non_probe[j] is not None:
                        uf.union(a.non_probe[i], b.non_probe[j])
                else:
                    first = a.anchors[i][0]
                    for q in a.anchors[i][1:] + b.anchors[j]:
                        uf.union(first, q)

    all_intervals = sorted(classes)
    for k, t in enumerate(all_intervals):
        compare(classes[t], classes[t], same_interval=True)
        if k + 1 < len(all_intervals):
            compare(classes[t], classes[all_intervals[k + 1]], same_interval=False)


# ---------------------------------------------------------------------------
# 'exact' strategy
# ---------------------------------------------------------------------------

def _link_exact(records: list[LeakageRecord], uf: UnionFind) -> None:
    """Group records whose tokens compare equal (BSigma-style deterministic tokens)."""
    token_to_root: dict = {}
    for rec in sorted(records, key=lambda r: r.query_id):
        tok = rec.token
        if tok in token_to_root:
            uf.union(token_to_root[tok], rec.query_id)
        else:
            token_to_root[tok] = rec.query_id


# ---------------------------------------------------------------------------
# 'partial' strategy
# ---------------------------------------------------------------------------

def _get_partial_key(token) -> str:
    """Extract the keyword-shared component from a token.

    Supports:
    - Objects with a ``keyword_component`` attribute (MitraToken).
    - Strings split on ``'|'`` (T1-style tokens).
    - Any other type: falls back to ``str(token)``.
    """
    if hasattr(token, "keyword_component"):
        return str(token.keyword_component)
    if isinstance(token, str) and "|" in token:
        return token.split("|")[0]
    return str(token)


def _link_partial(records: list[LeakageRecord], uf: UnionFind) -> None:
    """Group records that share the same keyword-derived token component (Mitra/Janus++)."""
    component_to_root: dict[str, int] = {}
    for rec in sorted(records, key=lambda r: r.query_id):
        key = _get_partial_key(rec.token)
        if key in component_to_root:
            uf.union(component_to_root[key], rec.query_id)
        else:
            component_to_root[key] = rec.query_id


# ---------------------------------------------------------------------------
# 'computational' strategy
# ---------------------------------------------------------------------------

def _link_computational(
    records: list[LeakageRecord],
    uf: UnionFind,
    forward_check,
) -> None:
    """Group records linked by forward_check(new_token, old_token) -> bool.

    All-pairs check: O(n^2) in the number of records.  For each ordered pair
    (newer, older) the callable is invoked; a ``True`` result merges the two
    query_ids into the same group.
    """
    sorted_recs = sorted(records, key=lambda r: (r.timestamp, r.query_id))
    n = len(sorted_recs)
    for i in range(n):
        for j in range(i):  # j < i means j is "older" by sort order
            newer = sorted_recs[i]
            older = sorted_recs[j]
            try:
                if forward_check(newer.token, older.token):
                    uf.union(newer.query_id, older.query_id)
            except Exception:  # pragma: no cover – defensive
                pass


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
