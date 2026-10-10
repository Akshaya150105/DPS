"""Volumetric Inference Attack (VIA) against forward-private SSE.

Reference: Section 4.2 of 'Leakage-Abuse Attacks Against Forward and
Backward Private Searchable Symmetric Encryption' (CCS 2023).

The attack exploits *volumetric leakage* between pairs of queries linked by
equality inference:
  1. For each linked pair (q, q~):
       a_q  = (ulen + rlen) / 2
       a_q~ = (ulen + rlen) / 2
       delta_a = a_q~ - a_q
  2. Filter auxiliary known documents inserted between q and q~:
       q.timestamp < mu_doc <= q~.timestamp
  3. Build:
       L[w] = count of known docs in this interval containing w
       V[w] = multiset of document volumes for those docs
  4. Candidate keywords are {w : L[w] == delta_a}.
  5. If more than one candidate remains, compute the multiset difference
     (fvol_q~ - fvol_q) and drop any w for which a volume in the difference
     is missing from V[w] (respecting multiplicity).
  6. If one candidate remains, assign it to q; otherwise None.

Queries without a linked pair (e.g. the final query of an equality chain or
singleton queries) cannot form a volumetric comparison pair (q, q~) and are
assigned None.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from equality import EqualityResult
from types_ import Aux, Event, LeakageRecord


def run(
    records: list[LeakageRecord],
    aux: Aux,
    equality: EqualityResult,
    events: list[Event],
) -> dict[int, str | None]:
    """Run the Volumetric Inference Attack and return per-query predictions.

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed across
                  the simulation.
        aux:      Adversary auxiliary knowledge. Must contain ``known_docs``
                  (either a list of :class:`~types_.Document` objects or a list
                  of integer document IDs).
        equality: :class:`~equality.EqualityResult` containing query groupings
                  and ``linked_pairs``.
        events:   Event log from the simulation used to determine insertion
                  timestamps and document compositions.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` when
        unresolved or when the query does not have a linked successor).
    """
    # Initialise all queries to unresolved (None).
    predictions: dict[int, str | None] = {rec.query_id: None for rec in records}

    if not records or aux.known_docs is None or not events:
        return predictions

    rec_map: dict[int, LeakageRecord] = {r.query_id: r for r in records}

    # -----------------------------------------------------------------------
    # Step A: Build known document metadata from aux.known_docs and events
    # -----------------------------------------------------------------------
    # Map doc_id -> insertion timestamp (first 'add' event mu)
    # Map doc_id -> set of keywords
    doc_insert_mu: dict[int, int] = {}
    doc_kws_from_events: dict[int, set[str]] = defaultdict(set)
    for e in events:
        if e.op == "add":
            if e.doc_id not in doc_insert_mu:
                doc_insert_mu[e.doc_id] = e.mu
            doc_kws_from_events[e.doc_id].add(e.keyword)

    # known_docs_info: doc_id -> (keywords_set, volume_int, insert_mu_int)
    known_docs_info: dict[int, tuple[set[str], int, int]] = {}
    for item in aux.known_docs:
        if isinstance(item, int):
            doc_id = item
            kws = doc_kws_from_events.get(doc_id, set())
            vol = len(kws)
            mu = doc_insert_mu.get(doc_id, -1)
            known_docs_info[doc_id] = (kws, vol, mu)
        elif hasattr(item, "doc_id") and hasattr(item, "keywords"):
            doc_id = item.doc_id
            kws = set(item.keywords)
            vol = len(kws)
            mu = doc_insert_mu.get(doc_id, -1)
            known_docs_info[doc_id] = (kws, vol, mu)

    if not known_docs_info:
        return predictions

    # Candidate vocabulary
    vocab: list[str] = sorted(
        {kw for kws, _, _ in known_docs_info.values() for kw in kws}
        | {e.keyword for e in events}
    )
    if not vocab:
        return predictions

    # -----------------------------------------------------------------------
    # Step B: For each linked pair (q, q~), compute delta_a and filter candidates
    # -----------------------------------------------------------------------
    pair_resolved: dict[int, str] = {}
    group_candidate_sets: dict[int, list[set[str]]] = defaultdict(list)

    for q_id, q_tilde_id in equality.linked_pairs.items():
        if q_tilde_id is None:
            # Queries without a linked successor cannot form a comparison pair.
            continue

        q = rec_map.get(q_id)
        q_tilde = rec_map.get(q_tilde_id)
        if q is None or q_tilde is None:
            continue

        # Compute a_q = (ulen + rlen) / 2 and a_q~ likewise; delta_a = a_q~ - a_q
        a_q = (q.ulen + q.rlen) / 2.0
        a_q_tilde = (q_tilde.ulen + q_tilde.rlen) / 2.0
        delta_a = int(round(a_q_tilde - a_q))

        # Using aux.known_docs inserted between the two queries:
        # q.timestamp < mu <= q_tilde.timestamp
        between_docs = [
            (kws, vol)
            for kws, vol, mu in known_docs_info.values()
            if q.timestamp < mu <= q_tilde.timestamp
        ]

        # Build L[w] (count of known docs containing w) and V[w] (multiset of volumes)
        L: dict[str, int] = defaultdict(int)
        V: dict[str, Counter[int]] = defaultdict(Counter)
        for kws, vol in between_docs:
            for w in kws:
                L[w] += 1
                V[w][vol] += 1

        # Candidates are {w : L[w] == delta_a}
        candidates: list[str] = [w for w in vocab if L[w] == delta_a]

        # If more than one remains, compute the multiset difference fvol_q~ - fvol_q
        # and drop any w for which a volume in the difference is missing from V[w]
        # (respecting multiplicity).
        if len(candidates) > 1:
            diff_fvol = q_tilde.fvol - q.fvol
            filtered: list[str] = []
            for w in candidates:
                missing = False
                for vol, cnt in diff_fvol.items():
                    if cnt > 0 and V[w][vol] < cnt:
                        missing = True
                        break
                if not missing:
                    filtered.append(w)
            candidates = filtered

        gid = equality.group_of.get(q_id)
        if candidates and gid is not None:
            group_candidate_sets[gid].append(set(candidates))

        if len(candidates) == 1:
            pair_resolved[q_id] = candidates[0]

    # -----------------------------------------------------------------------
    # Step C: Group-level consolidation & propagation across linked queries
    # -----------------------------------------------------------------------
    group_resolved: dict[int, str] = {}
    for gid, cand_sets in group_candidate_sets.items():
        if not cand_sets:
            continue
        inter = cand_sets[0].copy()
        for cs in cand_sets[1:]:
            inter &= cs
            if not inter:
                inter = min(cand_sets, key=len)
                break
        if len(inter) == 1:
            group_resolved[gid] = next(iter(inter))

    # Assign predicted keyword:
    # A query q with a linked pair receives its resolved keyword (if unique);
    # queries without a linked pair (linked_pairs[qid] is None) receive None.
    for q_id, q_tilde_id in equality.linked_pairs.items():
        if q_tilde_id is not None:
            gid = equality.group_of.get(q_id)
            resolved = pair_resolved.get(q_id) or (
                group_resolved.get(gid) if gid is not None else None
            )
            predictions[q_id] = resolved
        else:
            # Documented behaviour: queries without a linked pair receive None.
            predictions[q_id] = None

    return predictions
