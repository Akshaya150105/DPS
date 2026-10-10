"""Shared data-contract types for the leakage-abuse attack simulator."""

import collections
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Core corpus / update types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """Immutable document record in the corpus.

    Attributes:
        doc_id:    Unique integer identifier.
        keywords:  Frozen set of keyword strings associated with the document.
        timestamp: Wall-clock (or simulated) creation time.

    Note:
        Volume of a document is defined as len(keywords).
    """

    doc_id: int
    keywords: frozenset[str]
    timestamp: datetime
    interval: Optional[int] = None


@dataclass(frozen=True)
class Event:
    """A single database update event (insertion or deletion).

    Attributes:
        mu:      Global logical timestamp (monotonically increasing counter).
        op:      Operation type – 'add' for insertion, 'del' for deletion.
        keyword: The keyword whose posting list is affected.
        doc_id:  The document being added or removed.
    """

    mu: int
    op: str          # 'add' | 'del'
    keyword: str
    doc_id: int


# ---------------------------------------------------------------------------
# Leakage record
# ---------------------------------------------------------------------------

@dataclass
class LeakageRecord:
    """Everything the server learns from a single search token.

    Attributes:
        query_id:  Monotonically increasing query counter (across all epochs).
        token:     Opaque token string emitted by the scheme.
        interval:  The epoch / interval index in which the query was issued.
        timestamp: Logical timestamp (mu) at query time.
        ulen:      Update-pattern length u = a + d (total docs touched for this keyword).
        rlen:      Response length n = a - d (current result-set size).
        fvol:      Forward-volume pattern: Counter mapping document volume ->
                   number of returned documents with that volume.
    """

    query_id: int
    token: str
    interval: int
    timestamp: int
    ulen: int
    rlen: int
    fvol: collections.Counter = field(default_factory=collections.Counter)
    is_probe: bool = False


# ---------------------------------------------------------------------------
# Auxiliary adversary knowledge
# ---------------------------------------------------------------------------

@dataclass
class Aux:
    """Optional auxiliary information available to the adversary.

    All fields are optional; an attack should degrade gracefully when some are
    absent.

    Attributes:
        query_dist:    Adversary's estimate of the query distribution per interval.
                       Mapping: interval -> {keyword: probability}.
        known_docs:    List of documents the adversary has obtained out-of-band.
        data_dist:     Adversary's estimate of keyword frequency per interval.
                       Mapping: interval -> {keyword: count}.
        pre_snapshot:  Snapshot of the corpus *before* the first observed epoch,
                       used by backward-privacy attacks.
    """

    query_dist: Optional[dict[int, dict[str, float]]] = None
    known_docs: Optional[list[Document]] = None
    data_dist: Optional[dict[int, dict[str, int]]] = None
    pre_snapshot: Optional[list[Document]] = None
