"""Partial-knowledge Volumetric Inference Attack (PVIA).

Reference: Section 6.2 of 'Leakage-Abuse Attacks Against Forward and
Backward Private Searchable Symmetric Encryption' (CCS 2023).

PVIA models an adversary who has obtained only a sampled fraction alpha
(where 0 < alpha < 1) of the inserted files out-of-band:
  1. For each linked pair (q, q~):
       a_q  = (ulen + rlen) / 2
       a_q~ = (ulen + rlen) / 2
       delta_a = a_q~ - a_q
  2. Filter auxiliary known documents inserted between q and q~:
       q.timestamp < mu_doc <= q~.timestamp
  3. Build:
       L[w] = count of known docs in this interval containing w
       V[w] = multiset of document volumes for those docs
  4. Form candidate set using the candidate window:
       alpha * delta_a <= L[w] <= delta_a
  5. Among candidates, choose the keyword with the largest multiset overlap
     between (fvol_q~ - fvol_q) and V[w]; break ties by smaller distance
     to the window.
  6. Assign the winning candidate to q (if unique), otherwise None.

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
    alpha: float | None = None,
) -> dict[int, str | None]:
    """Run the Partial Volumetric Inference Attack and return per-query predictions.

    Args:
        records:  List of :class:`~types_.LeakageRecord` objects observed across
                  the simulation.
        aux:      Adversary auxiliary knowledge containing ``known_docs``
                  (fraction alpha of corpus documents or doc_ids).
        equality: :class:`~equality.EqualityResult` containing query groupings
                  and ``linked_pairs``.
        events:   Simulation event log used to extract insertion timestamps
                  and document keywords.
        alpha:    Fraction of files known to the adversary (must satisfy
                  ``0 < alpha < 1``). If None, extracted from ``aux.alpha``
                  if present.

    Returns:
        Dict mapping ``query_id -> predicted keyword`` (or ``None`` when
        unresolved or when the query does not have a linked successor).

    Raises:
        ValueError: If alpha is not in the open interval (0, 1).
    """
    if alpha is None:
        alpha = getattr(aux, "alpha", None)

    if alpha is None:
        raise ValueError("alpha must be provided as an argument or in aux.alpha.")

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}.")

    # Initialise all queries to unresolved (None).
    predictions: dict[int, str | None] = {rec.query_id: None for rec in records}

    if not records or aux.known_docs is None or not events:
        return predictions

    rec_map: dict[int, LeakageRecord] = {r.query_id: r for r in records}

    # -----------------------------------------------------------------------
    # Step A: Build known document metadata from aux.known_docs and events
    # -----------------------------------------------------------------------
    doc_insert_mu: dict[int, int] = {}
    doc_kws_from_events: dict[int, set[str]] = defaultdict(set)
    for e in events:
        if e.op == "add":
            if e.doc_id not in doc_insert_mu:
                doc_insert_mu[e.doc_id] = e.mu
            doc_kws_from_events[e.doc_id].add(e.keyword)

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

    vocab: list[str] = sorted(
        {kw for kws, _, _ in known_docs_info.values() for kw in kws}
        | {e.keyword for e in events}
    )
    if not vocab:
        return predictions

    # -----------------------------------------------------------------------
    # Step B: For each linked pair (q, q~), apply window filter & score overlap
    # -----------------------------------------------------------------------
    pair_resolved: dict[int, str] = {}
    group_candidates: dict[int, list[set[str]]] = defaultdict(list)

    for q_id, q_tilde_id in equality.linked_pairs.items():
        if q_tilde_id is None:
            # Queries without a linked successor cannot form a comparison pair.
            continue

        q = rec_map.get(q_id)
        q_tilde = rec_map.get(q_tilde_id)
        if q is None or q_tilde is None:
            continue

        a_q = (q.ulen + q.rlen) / 2.0
        a_q_tilde = (q_tilde.ulen + q_tilde.rlen) / 2.0
        delta_a = int(round(a_q_tilde - a_q))

        if delta_a <= 0:
            pair_resolved[q_id] = None
            continue

        between_docs = [
            (kws, vol)
            for kws, vol, mu in known_docs_info.values()
            if q.timestamp < mu <= q_tilde.timestamp
        ]

        L: dict[str, int] = defaultdict(int)
        V: dict[str, Counter[int]] = defaultdict(Counter)
        for kws, vol in between_docs:
            for w in kws:
                L[w] += 1
                V[w][vol] += 1

        # Candidate window: alpha * delta_a <= L[w] <= delta_a
        lo = alpha * delta_a
        hi = float(delta_a)

        candidates = [w for w in vocab if lo <= L[w] <= hi]
        if not candidates:
            # Fallback when sampling variance leaves the window empty
            candidates = list(vocab)

        diff_fvol = q_tilde.fvol - q.fvol

        # Scoring function:
        # 1. Largest multiset overlap between (fvol_q~ - fvol_q) and V[w]
        # 2. Break ties by smaller distance to the window [lo, hi]
        # 3. Further break ties by proximity to the expected count in the window
        def score(w: str) -> tuple[int, float, float]:
            overlap = sum(
                min(diff_fvol[v], V[w][v])
                for v in diff_fvol
                if diff_fvol[v] > 0
            )
            dist_to_window = max(0.0, lo - L[w], L[w] - hi)
            dist_to_center = abs(L[w] - (lo + hi) / 2.0)
            return (overlap, -dist_to_window, -dist_to_center)

        scored = sorted(candidates, key=score, reverse=True)
        best_score = score(scored[0])
        top_cands = [w for w in scored if score(w) == best_score]

        gid = equality.group_of.get(q_id)
        if top_cands and gid is not None:
            group_candidates[gid].append(set(top_cands))

        if len(top_cands) == 1:
            pair_resolved[q_id] = top_cands[0]
        else:
            pair_resolved[q_id] = None

    # -----------------------------------------------------------------------
    # Step C: Group-level consolidation & propagation across linked queries
    # -----------------------------------------------------------------------
    group_resolved: dict[int, str] = {}
    for gid, cand_sets in group_candidates.items():
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

    for q_id, q_tilde_id in equality.linked_pairs.items():
        if q_tilde_id is not None:
            gid = equality.group_of.get(q_id)
            resolved = pair_resolved.get(q_id) or (
                group_resolved.get(gid) if gid is not None else None
            )
            predictions[q_id] = resolved
        else:
            # Queries without a linked pair receive None.
            predictions[q_id] = None

    return predictions
