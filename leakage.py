"""Leakage extraction: converts raw events + queries into LeakageRecord objects.

All functions operate on a list of Event objects and a target keyword *w*.
The logical timestamp *mu* acts as a "snapshot point": only events with
event.mu <= mu are considered, matching the server's view at query time.

Leakage algebra identities (from the paper):
    n  = rlen  = a - d
    u  = ulen  = a + d
    a  = (u + n) / 2   (insertions)
    d  = (u - n) / 2   (deletions)
"""

import collections
from typing import Callable

from types_ import Event


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _events_for(
    events: list[Event],
    w: str,
    mu: int,
    op: str | None = None,
) -> list[Event]:
    """Return events matching keyword *w* up to logical time *mu*.

    Args:
        events: Full event log (any order; filtered here).
        w:      Target keyword.
        mu:     Snapshot logical timestamp (inclusive upper bound).
        op:     If given, restrict to 'add' or 'del'; None means both.

    Returns:
        Filtered list of Event objects, chronological order.
    """
    return [
        e for e in events
        if e.keyword == w and e.mu <= mu and (op is None or e.op == op)
    ]


# ---------------------------------------------------------------------------
# Core leakage functions
# ---------------------------------------------------------------------------

def time_db(events: list[Event], w: str, mu: int) -> list[int]:
    """Return the sorted list of logical timestamps for all updates to keyword *w*.

    This is the *update pattern* / time-DB leakage: the adversary learns
    *when* keyword *w* was touched, but not the document details.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        Sorted list of mu values for each update event for *w*.
    """
    return sorted(e.mu for e in _events_for(events, w, mu))


def updates(events: list[Event], w: str, mu: int) -> list[Event]:
    """Return all update events for keyword *w* up to time *mu*.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        List of matching Event objects in chronological order.
    """
    return sorted(_events_for(events, w, mu), key=lambda e: e.mu)


def del_hist(events: list[Event], w: str, mu: int) -> list[int]:
    """Return the doc_ids of documents *deleted* from keyword *w*'s posting list.

    Used by backward-privacy attacks: the server may learn *which* previously
    inserted documents have been removed.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        List of doc_ids for deletion events, in chronological order.
    """
    return [e.doc_id for e in sorted(
        _events_for(events, w, mu, op="del"), key=lambda e: e.mu
    )]


def ulen(events: list[Event], w: str, mu: int) -> int:
    """Compute u = a + d: total number of update events for keyword *w*.

    This is the *update-pattern length* leaked by forward-private schemes.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        Integer count of all add and del events for *w* up to *mu*.
    """
    return len(_events_for(events, w, mu))


def ilen(events: list[Event], w: str, mu: int) -> int:
    """Compute a: number of insertion events for keyword *w* up to time *mu*.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        Integer count of 'add' events.
    """
    return len(_events_for(events, w, mu, op="add"))


def dlen(events: list[Event], w: str, mu: int) -> int:
    """Compute d: number of deletion events for keyword *w* up to time *mu*.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        Integer count of 'del' events.
    """
    return len(_events_for(events, w, mu, op="del"))


def rlen(events: list[Event], w: str, mu: int) -> int:
    """Compute n = a - d: current result-set size for keyword *w*.

    Each insertion adds a document; each deletion removes one.  The net
    count is the number of documents currently matching *w*.

    Args:
        events: Full event log.
        w:      Target keyword.
        mu:     Snapshot logical timestamp.

    Returns:
        Non-negative integer result-set length.
    """
    return ilen(events, w, mu) - dlen(events, w, mu)


def fvol(
    events: list[Event],
    w: str,
    mu: int,
    volume_of: Callable[[int], int],
) -> collections.Counter:
    """Compute the forward-volume pattern for keyword *w*.

    The forward-volume pattern is a multiset (Counter) mapping:
        document volume  ->  number of *currently matching* documents with that volume.

    Only documents still in the result set (inserted but not deleted) are
    counted.  Volume is computed via the caller-supplied *volume_of* function
    so that leakage.py remains independent of the corpus representation.

    Args:
        events:    Full event log.
        w:         Target keyword.
        mu:        Snapshot logical timestamp.
        volume_of: Callable(doc_id) -> int, returns the volume of a document.

    Returns:
        Counter mapping volume (int) -> count of current results with that volume.
    """
    # Build the current result set: inserted minus deleted doc_ids.
    added:   set[int] = {e.doc_id for e in _events_for(events, w, mu, op="add")}
    deleted: set[int] = {e.doc_id for e in _events_for(events, w, mu, op="del")}
    current_docs = added - deleted

    # Tally volumes.
    vol_counter: collections.Counter = collections.Counter()
    for doc_id in current_docs:
        vol_counter[volume_of(doc_id)] += 1
    return vol_counter


# ---------------------------------------------------------------------------
# Response-set similarity (RSP)
# ---------------------------------------------------------------------------

def rsp(
    fvol_a: collections.Counter,
    fvol_b: collections.Counter,
) -> float:
    """Compute response-set similarity between two forward-volume patterns.

    Uses Counter multiset intersection over union (Jaccard-style):

        rsp(A, B) = |A ∩ B| / |A ∪ B|

    where intersection and union are multiset operations:
        (A ∩ B)[v] = min(A[v], B[v])
        (A ∪ B)[v] = max(A[v], B[v])

    Returns 1.0 for identical multisets and 0.0 for disjoint ones.

    Args:
        fvol_a: Forward-volume Counter for the first query observation.
        fvol_b: Forward-volume Counter for the second query observation.

    Returns:
        Float in [0, 1].
    """
    if not fvol_a and not fvol_b:
        # Two empty result sets are treated as identical.
        return 1.0

    intersection = fvol_a & fvol_b   # Counter.__and__ = element-wise min
    union        = fvol_a | fvol_b   # Counter.__or__  = element-wise max

    inter_size = sum(intersection.values())
    union_size = sum(union.values())

    if union_size == 0:
        return 1.0
    return inter_size / union_size


# ---------------------------------------------------------------------------
# Query-equality predicate (T1 – token prefix)
# ---------------------------------------------------------------------------

def qeq(w1: str, w2: str) -> bool:
    """T1 equality: two tokens are equal iff they share the prefix before '|'.

    In Type-1 (T1) forward-private schemes the server receives tokens of the
    form  "<keyword_hash>|<counter>".  Two tokens for the same keyword therefore
    share the same prefix.

    Args:
        w1: First token string.
        w2: Second token string.

    Returns:
        True if the prefix before the first '|' is identical.
    """
    prefix1 = w1.split("|")[0]
    prefix2 = w2.split("|")[0]
    return prefix1 == prefix2


# ---------------------------------------------------------------------------
# Leakage-algebra derivation helpers
# ---------------------------------------------------------------------------

def derive_insert_delete(u: int, n: int) -> tuple[int, int]:
    """Derive (a, d) from the observable pair (u, n).

    From the leakage algebra:
        u = a + d,  n = a - d
    =>  a = (u + n) / 2,  d = (u - n) / 2

    Args:
        u: ulen = total updates (a + d).
        n: rlen = net result-set size (a - d).

    Returns:
        Tuple (a, d) of integers.

    Raises:
        ValueError: If u and n have different parities (non-integer solution).
    """
    if (u + n) % 2 != 0:
        raise ValueError(
            f"u={u} and n={n} have different parities; no integer solution for (a, d)."
        )
    a = (u + n) // 2
    d = (u - n) // 2
    return a, d


def derive_from_n_d(n: int, d: int) -> tuple[int, int]:
    """Derive (a, u) from the observable pair (n, d).

    From the leakage algebra:
        n = a - d  =>  a = n + d
        u = a + d  =>  u = n + 2d

    Args:
        n: rlen = net result-set size (a - d).
        d: dlen = number of deletions.

    Returns:
        Tuple (a, u) of integers.
    """
    a = n + d
    u = a + d   # equivalently u = n + 2*d
    return a, u
