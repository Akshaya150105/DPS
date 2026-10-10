"""Simple padding and equality-hiding countermeasures for leakage experiments.

These helpers transform simulator inputs/observations; they do not implement a
production ORAM or encrypted-index layout. The storage model used by
``overhead_report`` counts one posting entry per document-keyword incidence.
"""

from __future__ import annotations

from dataclasses import replace
import secrets
from typing import Iterable

import numpy as np

from types_ import Document, LeakageRecord


PAPER_SECTION_8_EMAILS = 30_204
PAPER_SECTION_8_KEYWORDS = 73_991
PAPER_SECTION_8_OVERHEAD_FACTOR = 107.0


def _ceil_to_bucket(value: int, bucket: int) -> int:
    return ((value + bucket - 1) // bucket) * bucket


def volume_hiding_padding(
    records: Iterable[LeakageRecord],
    bucket: int,
) -> list[LeakageRecord]:
    """Return copied leakage records with ``rlen`` and ``ulen`` bucket-padded.

    ``bucket`` is the bucket width in units of leakage length. Each length is
    rounded independently up to the next multiple (an already aligned value
    is unchanged). The same deterministic rule is used for every observation;
    this is a simplified leakage simulation, not a claim that padding is free.
    """
    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
        raise ValueError(f"bucket must be a positive integer, got {bucket!r}.")

    return [
        replace(
            record,
            rlen=_ceil_to_bucket(record.rlen, bucket),
            ulen=_ceil_to_bucket(record.ulen, bucket),
            fvol=record.fvol.copy(),
        )
        for record in records
    ]


def file_volume_padding(dataset: Iterable[Document]) -> list[Document]:
    """Pad each document with opaque dummy terms to the dataset's max volume.

    A document's volume is ``len(document.keywords)`` in this simulator.
    Unique padding terms preserve every original document/keyword incidence
    and ensure dummy terms do not create shared posting lists across files.
    The returned documents are copies; the input dataset is not mutated.
    """
    docs = list(dataset)
    if not docs:
        return []
    if any(not isinstance(doc, Document) for doc in docs):
        raise TypeError("file_volume_padding expects Document instances.")

    target_volume = max(len(doc.keywords) for doc in docs)
    existing_keywords = {keyword for doc in docs for keyword in doc.keywords}
    namespace = "__dps_volume_padding__"
    while any(keyword.startswith(namespace) for keyword in existing_keywords):
        namespace = f"_{namespace}"

    padded: list[Document] = []
    for doc in docs:
        keywords = set(doc.keywords)
        for index in range(target_volume - len(keywords)):
            keywords.add(f"{namespace}{doc.doc_id}_{index}__")
        padded.append(
            Document(
                doc_id=doc.doc_id,
                keywords=frozenset(keywords),
                timestamp=doc.timestamp,
            )
        )
    return padded


def overhead_report(dataset: Iterable[Document]) -> dict[str, int | float | None]:
    """Estimate posting-entry storage overhead if volume-padding ``dataset``.

    Reports the original incidence count, max-volume padded count, added
    dummy entries, and multiplicative/percentage overhead. This simplified
    estimate ignores document metadata, encryption, and fixed per-file costs.
    """
    docs = list(dataset)
    if any(not isinstance(doc, Document) for doc in docs):
        raise TypeError("overhead_report expects Document instances.")
    original_entries = sum(len(doc.keywords) for doc in docs)
    max_volume = max((len(doc.keywords) for doc in docs), default=0)
    padded_entries = len(docs) * max_volume
    padding_entries = padded_entries - original_entries
    ratio = padded_entries / original_entries if original_entries else None
    percent = (padding_entries / original_entries * 100.0) if original_entries else None
    return {
        "document_count": len(docs),
        "original_posting_entries": original_entries,
        "padded_posting_entries": padded_entries,
        "padding_entries": padding_entries,
        "overhead_ratio": ratio,
        "overhead_percent": percent,
        "target_volume": max_volume,
    }


def paper_section_8_estimate(
    email_count: int = PAPER_SECTION_8_EMAILS,
    keyword_count: int = PAPER_SECTION_8_KEYWORDS,
) -> dict[str, int | float | str | None]:
    """Return the paper's Section 8 Enron estimate and a distinct slot bound.

    Section 8 reports about 107x storage overhead for the Enron corpus with
    30,204 emails and 73,991 keywords. Those two corpus totals are not enough
    to derive that ratio independently: the paper's padding estimate also
    depends on posting-list and maximum file-volume statistics. The reported
    factor is therefore preserved as a paper-reported estimate, not presented
    as a value computed from just the two totals. Separately, the function
    returns ``email_count * keyword_count`` as a simple all-email/all-keyword
    posting-slot bound; it is not the measured padded storage or the derivation
    of the paper's 107x ratio.
    """
    if isinstance(email_count, bool) or not isinstance(email_count, int) or email_count < 0:
        raise ValueError("email_count must be a non-negative integer.")
    if isinstance(keyword_count, bool) or not isinstance(keyword_count, int) or keyword_count < 0:
        raise ValueError("keyword_count must be a non-negative integer.")
    entries = email_count * keyword_count
    is_paper_corpus = (
        email_count == PAPER_SECTION_8_EMAILS
        and keyword_count == PAPER_SECTION_8_KEYWORDS
    )
    return {
        "email_count": email_count,
        "keyword_count": keyword_count,
        "paper_reported_overhead_factor": PAPER_SECTION_8_OVERHEAD_FACTOR if is_paper_corpus else None,
        "paper_reported_overhead": "about 107x storage" if is_paper_corpus else None,
        "worst_case_posting_entries": entries,
        "worst_case_entries_per_email": keyword_count,
    }


def oram_equality_hiding(
    records: Iterable[LeakageRecord],
    seed: int | None = None,
) -> list[LeakageRecord]:
    """Replace every search token with an independent opaque random value.

    This models hiding token equality: each observation receives its own
    unrelated token, so ``build_equality(..., mode="T1")`` cannot match tokens
    and an analyst must instead use response similarity (T2). Supplying a seed
    makes experiments reproducible; the default uses fresh random bytes.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    result: list[LeakageRecord] = []
    seen: set[str] = set()
    for record in records:
        while True:
            token = rng.bytes(16).hex() if rng is not None else secrets.token_hex(16)
            if token not in seen:
                seen.add(token)
                break
        result.append(replace(record, token=token, fvol=record.fvol.copy()))
    return result
