"""Attack without Update Knowledge (Section 6.3).

Reference: Section 6.3 of 'Leakage-Abuse Attacks Against Forward and
Backward Private Searchable Symmetric Encryption' (CCS 2023).

Models an adversary who possesses only historical database knowledge before
time mu0 (aux.pre_snapshot) and has no access to subsequent update logs:
  1. For each target query q, find q~: the linked query (in q's equality
     group) occurring closest after mu0. If no such query exists, q~ = q.
  2. Compute insert length of q~: a_q~ = (ulen + rlen) / 2.
  3. Candidate keywords are {w : a_q~ >= L[w]}, where L[w] is the number of
     snapshot documents containing w.
  4. Narrow down candidates by dropping any w for which a volume in
     fvol_q~ is not in V[w] (where V[w] is the set of volumes of snapshot
     documents containing w).
  5. If exactly one candidate remains, assign it to q; otherwise None.
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
    mu0: int = 0,
) -> dict[int, str | None]:
    """Run the attack without update knowledge.

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed across
                  the simulation.
        aux:      Adversary auxiliary knowledge. Must contain ``pre_snapshot``
                  (list of :class:`~types_.Document` objects prior to time mu0).
        equality: :class:`~equality.EqualityResult` containing query groupings.
        mu0:      Logical timestamp cutoff for the pre-snapshot. If None,
                  attempts to read ``aux.mu0`` or defaults to 0.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` when
        unresolved). Every query_id in records is present in the returned dict.
    """
    if mu0 is None:
        mu0 = getattr(aux, "mu0", 0)

    # Initialise every query in records to unresolved (None).
    predictions: dict[int, str | None] = {rec.query_id: None for rec in records}

    if not records or aux.pre_snapshot is None:
        return predictions

    rec_map: dict[int, LeakageRecord] = {r.query_id: r for r in records}
    snapshot_docs = aux.pre_snapshot

    # -----------------------------------------------------------------------
    # Step 1: Build L[w] and V[w] from snapshot documents
    # -----------------------------------------------------------------------
    # L[w]: number of snapshot documents containing w
    # V[w]: set of volumes of snapshot documents containing w
    L: dict[str, int] = defaultdict(int)
    V: dict[str, set[int]] = defaultdict(set)
    vocab_set: set[str] = set()

    for doc in snapshot_docs:
        kws = doc.keywords if hasattr(doc, "keywords") else set()
        vol = len(kws)
        for w in kws:
            L[w] += 1
            V[w].add(vol)
            vocab_set.add(w)

    vocab = sorted(vocab_set)
    if not vocab:
        return predictions

    # -----------------------------------------------------------------------
    # Step 2: For each target query q, find q~ (closest after mu0) and filter
    # -----------------------------------------------------------------------
    for q in records:
        gid = equality.group_of.get(q.query_id)
        if gid is not None and gid in equality.groups:
            linked_recs = [rec_map[qid] for qid in equality.groups[gid] if qid in rec_map]
        else:
            linked_recs = [q]

        # Linked query closest after mu0 (if none, use q)
        after_mu0 = [r for r in linked_recs if r.timestamp >= mu0]
        if after_mu0:
            q_tilde = min(after_mu0, key=lambda r: (r.timestamp - mu0, r.query_id))
        else:
            q_tilde = q

        a_q_tilde = int(round((q_tilde.ulen + q_tilde.rlen) / 2.0))

        # Candidates are {w : a_q~ >= L[w]}
        candidates = [w for w in vocab if a_q_tilde >= L[w]]

        # Drop any w for which a volume in fvol_q~ is not in V[w]
        filtered: list[str] = []
        for w in candidates:
            missing = False
            for vol, cnt in q_tilde.fvol.items():
                if cnt > 0 and vol not in V[w]:
                    missing = True
                    break
            if not missing:
                filtered.append(w)

        # If exactly one candidate remains assign it to q, otherwise None
        if len(filtered) == 1:
            predictions[q.query_id] = filtered[0]
        else:
            predictions[q.query_id] = None

    return predictions
