"""Dataset preprocessing for Enron (maildir) and Lucene (mbox) corpora.

Parses email messages, normalises text (lowercase, strip punctuation, remove
quoted replies, tokenize, filter tokens < 3 characters, optional Porter stemming),
computes document frequencies, selects the top-K keywords, and exports:
    1. A pickle file with (docs, vocab, name)
    2. A keyword-to-document-IDs inverted index as JSON
    3. Metadata JSON containing document/keyword counts and date ranges
"""

import argparse
import collections
from datetime import datetime, timezone
import email
import email.message
import email.utils
import json
import os
import pickle
import re
import string
import sys
from typing import Any, Optional

from types_ import Document


# ---------------------------------------------------------------------------
# Self-contained Porter Stemmer (Martin Porter, 1980)
# ---------------------------------------------------------------------------

class PorterStemmer:
    """Standard Martin Porter (1980) stemming algorithm."""

    def __init__(self) -> None:
        self.b: str = ""
        self.k: int = 0
        self.k0: int = 0
        self.j: int = 0

    def _cons(self, i: int) -> bool:
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == self.k0 else not self._cons(i - 1)
        return True

    def _m(self) -> int:
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in "wxy":
            return False
        return True

    def _ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        if self.b[o:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        length = len(s)
        self.b = self.b[:self.j + 1] + s + self.b[self.j + length + 1:]
        self.k = self.j + length

    def _r(self, s: str) -> None:
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.k > self.k0 and self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + "i" + self.b[self.k + 1:]

    def _step2(self) -> None:
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif ch == "c":
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif ch == "e":
            if self._ends("izer"):
                self._r("ize")
        elif ch == "l":
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif ch == "o":
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif ch == "s":
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif ch == "t":
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")

    def _step3(self) -> None:
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif ch == "i":
            if self._ends("iciti"):
                self._r("ic")
        elif ch == "l":
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif ch == "s":
            if self._ends("ness"):
                self._r("")

    def _step4(self) -> None:
        ch = self.b[self.k - 1]
        if ch == "a":
            if not self._ends("al"):
                return
        elif ch == "c":
            if not (self._ends("ance") or self._ends("ence")):
                return
        elif ch == "e":
            if not self._ends("er"):
                return
        elif ch == "i":
            if not self._ends("ic"):
                return
        elif ch == "l":
            if not (self._ends("able") or self._ends("ible")):
                return
        elif ch == "n":
            if not (self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent")):
                return
        elif ch == "o":
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in "st":
                pass
            elif not self._ends("ou"):
                return
        elif ch == "s":
            if not self._ends("ism"):
                return
        elif ch == "t":
            if not (self._ends("ate") or self._ends("iti")):
                return
        elif ch == "u":
            if not self._ends("ous"):
                return
        elif ch == "v":
            if not self._ends("ive"):
                return
        elif ch == "z":
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stem a word using Porter's algorithm."""
        word = word.lower()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


_DEFAULT_STEMMER = PorterStemmer()


def stem_word(w: str) -> str:
    """Return stemmed form of *w*."""
    return _DEFAULT_STEMMER.stem(w)


# ---------------------------------------------------------------------------
# Text cleaning and tokenization helpers
# ---------------------------------------------------------------------------

def remove_quoted_replies(text: str) -> str:
    """Remove quoted reply sections and quote-prefixed lines from email body.

    Filters out standard email forward/reply dividers:
        - -----Original Message-----
        - --- Original Message ---
        - ________________________________
        - 'On <date> ... wrote:'
        - Lines starting with '>'
    """
    reply_split_patterns = [
        r"(?i)\n\s*-----Original Message-----",
        r"(?i)\n\s*----- Original Message -----",
        r"(?i)\n\s*--- Original Message ---",
        r"(?i)\n\s*________________________________",
        r"(?i)\n\s*From:.*?\n\s*Sent:.*?\n\s*To:.*?\n\s*Subject:",
        r"(?i)\n\s*On\s+.*?wrote:\s*\n",
    ]
    for pattern in reply_split_patterns:
        text = re.split(pattern, text, maxsplit=1)[0]

    # Filter out individual lines starting with '>'
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def tokenize_text(text: str, stem: bool = False) -> list[str]:
    """Lowercase, remove punctuation and quoted replies, tokenize, drop < 3 chars.

    Args:
        text: Raw email body text.
        stem: If True, apply Porter stemming to tokens.

    Returns:
        List of cleaned token strings (each of length >= 3).
    """
    # 1. Remove quoted replies
    text = remove_quoted_replies(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove punctuation: replace punctuation characters with whitespace
    for ch in string.punctuation:
        text = text.replace(ch, " ")

    # 4. Tokenize on whitespace
    raw_tokens = text.split()

    # 5. Drop tokens shorter than 3 characters
    tokens = [tok for tok in raw_tokens if len(tok) >= 3]

    # 6. Optional stemming
    if stem:
        tokens = [stem_word(tok) for tok in tokens]
        tokens = [tok for tok in tokens if len(tok) >= 3]

    return tokens


# ---------------------------------------------------------------------------
# Email message payload and date extraction
# ---------------------------------------------------------------------------

def extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from an email Message object."""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition", ""))
            if "attachment" in cdispo:
                continue
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        text = payload.decode(charset, errors="replace")
                    except Exception:
                        text = payload.decode("latin-1", errors="replace")
                    parts.append(text)
        return "\n".join(parts)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except Exception:
                return payload.decode("latin-1", errors="replace")
        payload_str = msg.get_payload()
        if isinstance(payload_str, str):
            return payload_str
        return ""


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse RFC 2822 / 822 Date header into a timezone-aware datetime.

    Returns None if unparseable or year is outside [1970, 2035].
    """
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        dt = email.utils.parsedate_to_datetime(date_str.strip())
        if dt.year < 1970 or dt.year > 2035:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_bound(val: Any, is_end: bool = False) -> datetime:
    """Parse a date boundary into a timezone-aware UTC datetime.

    If is_end is True and val is specified as a date without explicit non-zero
    time components, the time is set to 23:59:59.999999 to cover the entire day.
    """
    if isinstance(val, datetime):
        dt = val
        if is_end and (dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0):
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        dt = datetime(val.year, val.month, val.day)
        if is_end:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif isinstance(val, str):
        val_str = val.strip()
        if len(val_str) == 10 and val_str.count("-") == 2:
            dt = datetime.strptime(val_str, "%Y-%m-%d")
            if is_end:
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            try:
                dt = datetime.fromisoformat(val_str)
            except ValueError:
                dt = email.utils.parsedate_to_datetime(val_str)
    else:
        raise TypeError(f"Invalid date boundary type: {type(val)!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def split_into_intervals(
    docs: list[Document],
    start: Any,
    end: Any,
    freq: str = "month",
) -> list[Document]:
    """Assign each document an interval index by month and drop docs outside range.

    Reference: Section 6 of 'Leakage-Abuse Attacks Against Forward and Backward
    Private Searchable Symmetric Encryption' (CCS 2023).

    Args:
        docs:  List of Document instances (or objects with timestamp, doc_id, keywords).
        start: Start date boundary (str like 'YYYY-MM-DD' or datetime/date object).
        end:   End date boundary (str like 'YYYY-MM-DD' or datetime/date object).
        freq:  Interval frequency ('month' by default; supports 'day', 'week', 'year').

    Returns:
        List of Document instances that fall within [start, end], each with
        an assigned interval index.

    Raises:
        ValueError: If start is after end, or if freq is unsupported.
    """
    if start is None or end is None:
        raise ValueError("start and end boundaries must both be specified.")

    start_dt = _parse_bound(start, is_end=False)
    end_dt = _parse_bound(end, is_end=True)

    if start_dt > end_dt:
        raise ValueError(
            f"Start date ({start!r}) cannot be after end date ({end!r})."
        )

    freq_norm = str(freq).strip().lower()
    if freq_norm not in ("month", "m", "monthly", "day", "d", "daily", "week", "w", "weekly", "year", "y", "yearly"):
        raise ValueError(
            f"Unsupported frequency: {freq!r}. Must be 'month', 'day', 'week', or 'year'."
        )

    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)

    filtered_docs: list[Document] = []
    for doc in docs:
        dt = doc.timestamp
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt.astimezone(timezone.utc)

        if dt_utc < start_utc or dt_utc > end_utc:
            continue

        if freq_norm in ("month", "m", "monthly"):
            interval = (dt_utc.year - start_utc.year) * 12 + (dt_utc.month - start_utc.month)
        elif freq_norm in ("day", "d", "daily"):
            interval = (dt_utc.date() - start_utc.date()).days
        elif freq_norm in ("week", "w", "weekly"):
            interval = (dt_utc.date() - start_utc.date()).days // 7
        elif freq_norm in ("year", "y", "yearly"):
            interval = dt_utc.year - start_utc.year
        else:
            interval = 0

        new_doc = Document(
            doc_id=doc.doc_id,
            keywords=doc.keywords,
            timestamp=doc.timestamp,
            interval=interval,
        )
        filtered_docs.append(new_doc)

    return filtered_docs


# ---------------------------------------------------------------------------
# Source loaders: Enron maildir and Lucene mbox
# ---------------------------------------------------------------------------

def load_enron_messages(path: str) -> tuple[list[tuple[datetime, str]], int]:
    """Walk an Enron maildir directory and parse email files.

    Returns:
        A tuple of (valid_messages, skipped_bad_dates_count), where
        valid_messages is a list of (datetime, body_text).
    """
    messages: list[tuple[datetime, str]] = []
    skipped_bad_dates = 0

    if os.path.isfile(path):
        candidates = [path]
    else:
        candidates = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                candidates.append(os.path.join(root, fname))

    for fpath in candidates:
        try:
            with open(fpath, "rb") as f:
                raw_bytes = f.read()
            msg = email.message_from_bytes(raw_bytes)
        except Exception:
            continue

        dt = parse_date(msg.get("Date"))
        if dt is None:
            skipped_bad_dates += 1
            continue

        body = extract_body(msg)
        messages.append((dt, body))

    return messages, skipped_bad_dates


def load_lucene_messages(path: str) -> tuple[list[tuple[datetime, str]], int]:
    """Parse Lucene mbox archive files or mail directory.

    Returns:
        A tuple of (valid_messages, skipped_bad_dates_count), where
        valid_messages is a list of (datetime, body_text).
    """
    import mailbox

    messages: list[tuple[datetime, str]] = []
    skipped_bad_dates = 0

    if os.path.isfile(path):
        target_files = [path]
    else:
        target_files = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                target_files.append(os.path.join(root, fname))

    for fpath in target_files:
        is_mbox = False
        try:
            box = mailbox.mbox(fpath)
            box_len = len(box)
            if box_len > 0:
                is_mbox = True
                for msg in box:
                    dt = parse_date(msg.get("Date"))
                    if dt is None:
                        skipped_bad_dates += 1
                        continue
                    body = extract_body(msg)
                    messages.append((dt, body))
        except Exception:
            is_mbox = False

        if not is_mbox:
            # Fallback: parse as single RFC 822 email file
            try:
                with open(fpath, "rb") as f:
                    msg = email.message_from_bytes(f.read())
                dt = parse_date(msg.get("Date"))
                if dt is None:
                    skipped_bad_dates += 1
                    continue
                body = extract_body(msg)
                messages.append((dt, body))
            except Exception:
                continue

    return messages, skipped_bad_dates


# ---------------------------------------------------------------------------
# Preprocessing orchestrator
# ---------------------------------------------------------------------------

def preprocess_dataset(
    source: str,
    path: str,
    k: int = 500,
    stem: bool = False,
    name: Optional[str] = None,
    out_dir: str = "datasets/processed",
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    freq: str = "month",
) -> dict[str, Any]:
    """Parse raw messages, select top-K keywords, and save processed artifacts.

    Artifacts saved under `out_dir`:
        - `{name}.pkl`: (docs, vocab, name)
        - `{name}_index.json`: keyword -> list of doc_ids
        - `{name}_metadata.json`: counts and date range
        - `{name}_meta.json`: alias for metadata
        - `{name}_intervals.json`: doc_id -> interval (if intervals applied)

    Args:
        source:  'enron' or 'lucene'.
        path:    Directory or file containing raw email data.
        k:       Number of top keywords to keep by document frequency.
        stem:    Whether to apply Porter stemming.
        name:    Dataset identifier (defaults to source).
        out_dir: Destination folder for output files.
        start:   Optional start date boundary for interval filtering.
        end:     Optional end date boundary for interval filtering.
        freq:    Interval frequency ('month' by default).

    Returns:
        Metadata dict summarizing document counts, vocabulary, and date range.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path does not exist: '{path}'")

    source_lower = source.strip().lower()
    if source_lower == "enron":
        raw_msgs, skipped_bad_dates = load_enron_messages(path)
    elif source_lower == "lucene":
        raw_msgs, skipped_bad_dates = load_lucene_messages(path)
    else:
        raise ValueError(f"Unknown source: {source}. Choose 'enron' or 'lucene'.")

    # Print summary of parsing
    print(f"Parsed {len(raw_msgs)} valid messages from '{path}'.")
    print(f"Skipped {skipped_bad_dates} messages with bad dates.")

    if not name:
        name = source_lower

    # 1. Tokenize messages and compute document frequencies
    doc_entries: list[tuple[datetime, set[str]]] = []
    doc_freq: dict[str, int] = collections.defaultdict(int)

    for dt, body in raw_msgs:
        tokens = tokenize_text(body, stem=stem)
        unique_tokens = set(tokens)
        doc_entries.append((dt, unique_tokens))
        for tok in unique_tokens:
            doc_freq[tok] += 1

    # 2. Select top-K keywords by document frequency (deterministic tie-breaking)
    sorted_keywords = sorted(doc_freq.keys(), key=lambda w: (-doc_freq[w], w))
    vocab = sorted_keywords[:k]
    vocab_set = set(vocab)

    # 3. Sort messages chronologically by timestamp
    doc_entries.sort(key=lambda item: item[0])

    # 4. Construct Document objects
    docs: list[Document] = []
    for doc_id, (dt, unique_tokens) in enumerate(doc_entries):
        filtered_keywords = frozenset(tok for tok in unique_tokens if tok in vocab_set)
        docs.append(Document(doc_id=doc_id, keywords=filtered_keywords, timestamp=dt))

    # Apply date range filtering and interval assignment if requested
    if start is not None or end is not None:
        if start is None:
            start = "1999-12-01" if source_lower == "enron" else "2006-06-01"
        if end is None:
            end = "2001-02-27" if source_lower == "enron" else "2006-11-30"
        docs = split_into_intervals(docs, start=start, end=end, freq=freq)

    # 5. Build keyword to document IDs inverted index
    kw_to_docs: dict[str, list[int]] = {kw: [] for kw in vocab}
    for doc in docs:
        for kw in doc.keywords:
            if kw in kw_to_docs:
                kw_to_docs[kw].append(doc.doc_id)

    # 6. Date range calculation
    min_date = docs[0].timestamp if docs else None
    max_date = docs[-1].timestamp if docs else None

    # 7. Metadata summary
    has_intervals = any(doc.interval is not None for doc in docs)
    distinct_intervals = (
        sorted(set(doc.interval for doc in docs if doc.interval is not None))
        if has_intervals else []
    )
    metadata: dict[str, Any] = {
        "name": name,
        "source": source_lower,
        "stem": stem,
        "k": k,
        "counts": {
            "documents": len(docs),
            "vocab": len(vocab),
            "skipped_bad_dates": skipped_bad_dates,
            "total_messages_seen": len(raw_msgs) + skipped_bad_dates,
        },
        "document_count": len(docs),
        "vocab_size": len(vocab),
        "skipped_bad_dates": skipped_bad_dates,
        "date_range": {
            "start": min_date.isoformat() if min_date else None,
            "end": max_date.isoformat() if max_date else None,
        },
        "start_date": min_date.isoformat() if min_date else None,
        "end_date": max_date.isoformat() if max_date else None,
        "num_intervals": len(distinct_intervals),
        "intervals": distinct_intervals,
        "interval_index": {doc.doc_id: doc.interval for doc in docs if doc.interval is not None},
    }

    # 8. Save artifacts under out_dir
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((docs, vocab, name), f)

    index_path = os.path.join(out_dir, f"{name}_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(kw_to_docs, f, indent=2)

    meta_path = os.path.join(out_dir, f"{name}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Write aliases for convenience
    meta_alias = os.path.join(out_dir, f"{name}_meta.json")
    with open(meta_alias, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if has_intervals:
        intervals_path = os.path.join(out_dir, f"{name}_intervals.json")
        with open(intervals_path, "w", encoding="utf-8") as f:
            json.dump({str(doc.doc_id): doc.interval for doc in docs}, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """CLI driver for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess Enron maildir or Lucene mbox dataset."
    )
    parser.add_argument(
        "--source",
        choices=["enron", "lucene"],
        required=True,
        help="Source type: 'enron' (maildir) or 'lucene' (mbox).",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to raw dataset directory or file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of top keywords to keep by document frequency (default: 500).",
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        default=False,
        help="Enable Porter stemming (default: False).",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/processed",
        help="Directory to save output files (default: datasets/processed).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset name (defaults to --source).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date for filtering (YYYY-MM-DD). Defaults: Enron 1999-12-01, Lucene 2006-06-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for filtering (YYYY-MM-DD). Defaults: Enron 2001-02-27, Lucene 2006-11-30.",
    )
    parser.add_argument(
        "--freq",
        default="month",
        help="Interval frequency (default: 'month').",
    )

    args = parser.parse_args(argv)

    start = args.start
    end = args.end

    if args.source == "enron":
        if start is None:
            start = "1999-12-01"
        if end is None:
            end = "2001-02-27"
    elif args.source == "lucene":
        if start is None:
            start = "2006-06-01"
        if end is None:
            end = "2006-11-30"
        print(
            "Note: The paper omits the year for the Lucene end date; "
            "using default 2006-11-30 (this choice must be documented in the report)."
        )

    preprocess_dataset(
        source=args.source,
        path=args.path,
        k=args.k,
        stem=args.stem,
        name=args.name,
        out_dir=args.out_dir,
        start=start,
        end=end,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()

