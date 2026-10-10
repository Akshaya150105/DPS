"""Countermeasures against leakage-abuse attacks and storage cost evaluation.

Reference:
Section 8 and Section 7.4 of 'Leakage-Abuse Attacks Against Forward and Backward
Private Searchable Symmetric Encryption' (CCS 2023).

Techniques:
  1. Volume-hiding padding: Pads response length (rlen) and update length (ulen)
     to discrete bucket boundaries.
  2. File volume padding: Pads every document to uniform volume by adding dummy keywords.
  3. ORAM equality hiding: Replaces deterministic search tokens with pseudorandom
     identifiers so T1 token equality is hidden and only response similarity (T2)
     can infer equality.
  4. Overhead reporting: Evaluates storage and index size overhead under worst-case
     padding countermeasures.
"""

from __future__ import annotations

import collections
import copy
import math
import secrets
from typing import Any

from types_ import Document, LeakageRecord


# ---------------------------------------------------------------------------
# 1. Volume-Hiding Padding (Query & Update Lengths)
# ---------------------------------------------------------------------------

def volume_hiding_padding(
    records: list[LeakageRecord],
    bucket: int,
) -> list[LeakageRecord]:
    """Pad response length (rlen) and update length (ulen) up to bucket size.

    Each observed length is rounded up to the nearest multiple of *bucket*.
    This masks exact result set and update volume counts.

    Args:
        records: List of :class:`~types_.LeakageRecord` objects.
        bucket:  Positive integer bucket granularity.

    Returns:
        New list of :class:`~types_.LeakageRecord` objects with padded rlen and ulen.

    Raises:
        ValueError: If *bucket* < 1.
    """
    if bucket < 1:
        raise ValueError(f"bucket size must be >= 1, got {bucket!r}")

    padded_records: list[LeakageRecord] = []
    for r in records:
        padded_rlen = int(math.ceil(r.rlen / bucket)) * bucket
        padded_ulen = int(math.ceil(r.ulen / bucket)) * bucket

        new_rec = LeakageRecord(
            query_id=r.query_id,
            token=r.token,
            interval=r.interval,
            timestamp=r.timestamp,
            ulen=padded_ulen,
            rlen=padded_rlen,
            fvol=copy.copy(r.fvol),
            is_probe=getattr(r, "is_probe", False),
        )
        padded_records.append(new_rec)

    return padded_records


# ---------------------------------------------------------------------------
# 2. File Volume Padding (Document Level)
# ---------------------------------------------------------------------------

def file_volume_padding(
    dataset: list[Document],
    target_volume: int | None = None,
) -> list[Document]:
    """Pad every document in *dataset* to the exact same keyword volume.

    Adds unique dummy keywords to each document until its volume equals
    *target_volume*. When all files share the same volume, file-volume leakage
    (fvol) patterns provide zero distinguishing information to adversaries.

    Args:
        dataset:       List of :class:`~types_.Document` objects.
        target_volume: Padded volume target. If None, defaults to the maximum
                       volume found across all documents in *dataset*.

    Returns:
        New list of :class:`~types_.Document` objects with identical volume.

    Raises:
        ValueError: If *target_volume* is less than the maximum volume in *dataset*.
    """
    if not dataset:
        return []

    max_vol = max(len(d.keywords) for d in dataset)
    if target_volume is None:
        target_volume = max_vol
    elif target_volume < max_vol:
        raise ValueError(
            f"target_volume ({target_volume}) must be >= maximum document volume ({max_vol})"
        )

    padded_docs: list[Document] = []
    for doc in dataset:
        current_vol = len(doc.keywords)
        diff = target_volume - current_vol
        if diff > 0:
            # Generate deterministic dummy keywords per document
            dummy_kws = {f"__dummy_{doc.doc_id}_{i}__" for i in range(diff)}
            new_kws = frozenset(doc.keywords | dummy_kws)
        else:
            new_kws = doc.keywords

        padded_docs.append(
            Document(
                doc_id=doc.doc_id,
                keywords=new_kws,
                timestamp=doc.timestamp,
            )
        )

    return padded_docs


# ---------------------------------------------------------------------------
# 3. ORAM Equality Hiding
# ---------------------------------------------------------------------------

def oram_equality_hiding(
    records: list[LeakageRecord],
) -> list[LeakageRecord]:
    """Replace search tokens with fresh random values to hide token equality.

    Simulates an ORAM or randomized search pattern countermeasure: token strings
    never repeat and do not share deterministic prefixes. An adversary cannot
    use T1 prefix matching and must rely solely on response-set similarity (T2).

    Args:
        records: List of :class:`~types_.LeakageRecord` objects.

    Returns:
        New list of :class:`~types_.LeakageRecord` objects with randomized tokens.
    """
    hidden_records: list[LeakageRecord] = []
    for r in records:
        random_token = f"oram_{secrets.token_hex(16)}"
        new_rec = LeakageRecord(
            query_id=r.query_id,
            token=random_token,
            interval=r.interval,
            timestamp=r.timestamp,
            ulen=r.ulen,
            rlen=r.rlen,
            fvol=copy.copy(r.fvol),
            is_probe=getattr(r, "is_probe", False),
        )
        hidden_records.append(new_rec)

    return hidden_records


# ---------------------------------------------------------------------------
# 4. Storage Overhead Report
# ---------------------------------------------------------------------------

def overhead_report(
    dataset: list[Document],
    target_volume: int | None = None,
) -> dict[str, Any]:
    """Compute empirical storage overhead for uniform document volume padding.

    Args:
        dataset:       List of :class:`~types_.Document` objects.
        target_volume: Target volume to pad each document to (defaults to max volume).

    Returns:
        Dict containing:
          - num_docs: Number of documents.
          - original_pairs: Total keyword-document entries in original corpus.
          - padded_pairs: Total keyword-document entries after padding.
          - max_volume: Maximum document volume in original corpus.
          - avg_volume: Average document volume in original corpus.
          - target_volume: Document volume after padding.
          - overhead_ratio: Multiplier increase in storage (padded / original).
    """
    if not dataset:
        return {
            "num_docs": 0,
            "original_pairs": 0,
            "padded_pairs": 0,
            "max_volume": 0,
            "avg_volume": 0.0,
            "target_volume": 0,
            "overhead_ratio": 1.0,
        }

    num_docs = len(dataset)
    orig_pairs = sum(len(d.keywords) for d in dataset)
    max_vol = max(len(d.keywords) for d in dataset)
    avg_vol = orig_pairs / num_docs

    tgt_vol = target_volume if target_volume is not None else max_vol
    padded_pairs = num_docs * tgt_vol
    ratio = padded_pairs / orig_pairs if orig_pairs > 0 else 1.0

    return {
        "num_docs": num_docs,
        "original_pairs": orig_pairs,
        "padded_pairs": padded_pairs,
        "max_volume": max_vol,
        "avg_volume": round(avg_vol, 2),
        "target_volume": tgt_vol,
        "overhead_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# 5. Section 8 Cost Estimate Reproduction
# ---------------------------------------------------------------------------

def estimate_section_8_overhead(
    n_docs: int = 30204,
    n_keywords: int = 73991,
    max_doc_freq: int | None = None,
    max_doc_volume: int = 3000,
    avg_doc_volume: float = 25.0,
) -> dict[str, Any]:
    """Reproduce the CCS 2023 Section 8 storage overhead estimate.

    Section 8 analysis:
      Storing encrypted files padded to uniform volume across an encrypted multi-map
      incurs storage cost:
        O(|W| * max_{w in W} |DB[w]| * max_{d in D} #(d))
      For Enron with 30,204 emails and 73,991 keywords:
        - Original index entries ≈ N * avg_doc_volume ≈ 30,204 * 25 ≈ 7.55 * 10^5
        - Worst-case padded entries ≈ |W| * max_doc_freq * max_doc_volume
          ≈ 73,991 * 30,204 * 3,000 ≈ 6.70 * 10^12
        - Overhead ratio increases by approximately ~10^7x (order of 10^7).

    Args:
        n_docs:         Number of emails (default: 30,204).
        n_keywords:     Number of distinct keywords (default: 73,991).
        max_doc_freq:   Worst-case max posting list size (defaults to n_docs).
        max_doc_volume: Worst-case max keywords per document (default: 3,000).
        avg_doc_volume: Average keywords per document (default: 25.0).

    Returns:
        Dict containing original cost, padded cost, overhead ratio, and order of magnitude.
    """
    if max_doc_freq is None:
        max_doc_freq = n_docs

    orig_entries = float(n_docs * avg_doc_volume)
    padded_entries = float(n_keywords * max_doc_freq * max_doc_volume)

    overhead_ratio = padded_entries / orig_entries if orig_entries > 0 else 1.0
    log10_ratio = math.log10(overhead_ratio) if overhead_ratio > 0 else 0.0

    return {
        "n_docs": n_docs,
        "n_keywords": n_keywords,
        "max_doc_freq": max_doc_freq,
        "max_doc_volume": max_doc_volume,
        "avg_doc_volume": avg_doc_volume,
        "original_storage_entries": orig_entries,
        "padded_storage_entries": padded_entries,
        "overhead_ratio": overhead_ratio,
        "order_of_magnitude": f"10^{math.ceil(log10_ratio)}",
        "overhead_approx": "~10^7x",
    }
