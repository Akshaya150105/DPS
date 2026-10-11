"""Google Trends query distribution fetcher using pytrends.

Fetches monthly relative interest for a list of vocabulary keywords over a given
date range, normalizes each month's relative interest into a probability distribution
over the keywords, and saves the result to datasets/trends/trends.csv with columns:
    month, keyword, prob

Includes exponential backoff retries for rate limits and local disk caching to prevent
refetching on rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import pickle
import random
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from pytrends.request import TrendReq
except ImportError:  # pragma: no cover
    TrendReq = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocab file loader
# ---------------------------------------------------------------------------

def load_vocab_file(path: str | pathlib.Path) -> list[str]:
    """Load a list of keywords from a vocab file.

    Supports:
    * .pkl files containing (docs, vocab, name) or a list/set of keywords.
    * .json files containing a list of keywords or a dict with 'vocab' key
      or keyword keys (like inverted index files).
    * .csv files with a 'keyword' column or single column of keywords.
    * Plain text files with one keyword per line (ignoring # comments and blanks).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vocab file not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with path.open("rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, tuple) and len(data) >= 2 and isinstance(data[1], (list, set)):
            return [str(k) for k in data[1]]
        if isinstance(data, (list, set)):
            return [str(k) for k in data]
        if isinstance(data, dict):
            return [str(k) for k in data.keys()]
        raise ValueError(f"Unrecognized data structure in pickle file: {path}")

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(k) for k in data]
        if isinstance(data, dict):
            if "vocab" in data and isinstance(data["vocab"], (list, set)):
                return [str(k) for k in data["vocab"]]
            return [str(k) for k in data.keys()]
        raise ValueError(f"Unrecognized data structure in JSON file: {path}")

    if suffix == ".csv":
        df = pd.read_csv(path)
        if "keyword" in df.columns:
            return [str(k) for k in df["keyword"].dropna().unique()]
        # First column
        return [str(k) for k in df.iloc[:, 0].dropna().unique()]

    # Default to text file
    keywords: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.append(line)
    return keywords


# ---------------------------------------------------------------------------
# Date range helper
# ---------------------------------------------------------------------------

def format_timeframe(start: str, end: str) -> str:
    """Format start and end dates into pytrends timeframe format 'YYYY-MM-DD YYYY-MM-DD'."""
    start_str = start.strip()
    end_str = end.strip()

    # Normalise YYYY-MM to YYYY-MM-01
    if len(start_str) == 7 and start_str.count("-") == 1:
        start_str = f"{start_str}-01"

    if len(end_str) == 7 and end_str.count("-") == 1:
        # End of month: use day 28 or days in month
        parts = end_str.split("-")
        year, month = int(parts[0]), int(parts[1])
        if month in {1, 3, 5, 7, 8, 10, 12}:
            day = 31
        elif month in {4, 6, 9, 11}:
            day = 30
        else:
            day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
        end_str = f"{end_str}-{day:02d}"

    return f"{start_str} {end_str}"


# ---------------------------------------------------------------------------
# Fetching & Caching
# ---------------------------------------------------------------------------

def get_cache_key(batch: list[str], timeframe: str) -> str:
    """Deterministic cache filename for a keyword batch and timeframe."""
    raw = f"{timeframe}::" + ",".join(sorted(batch))
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    slug = "_".join(batch[:2]).replace(" ", "_")[:30]
    return f"trends_{slug}_{h}.csv"


def fetch_batch_with_retry(
    pytrend: Any,
    batch: list[str],
    timeframe: str,
    cache_dir: pathlib.Path | None = None,
    max_retries: int = 5,
    initial_delay: float = 3.0,
    backoff_factor: float = 2.0,
    jitter: float = 1.0,
) -> pd.DataFrame:
    """Fetch relative interest for a batch of up to 5 keywords with retries and caching.

    Args:
        pytrend: An initialised TrendReq instance or mock.
        batch: List of up to 5 keywords.
        timeframe: Date range string in pytrends format (e.g. '2004-01-01 2004-12-31').
        cache_dir: Optional path to directory storing cached CSV responses.
        max_retries: Maximum number of retry attempts on rate limit / network error.
        initial_delay: Initial sleep duration before retry in seconds.
        backoff_factor: Exponential multiplier for backoff delay.
        jitter: Uniform random noise added to delay.

    Returns:
        DataFrame indexed by date containing monthly/periodic counts for the keywords.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / get_cache_key(batch, timeframe)
        if cache_file.exists():
            logger.info("Loading cached trends for batch %s from %s", batch, cache_file)
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

    if pytrend is None:
        if TrendReq is None:
            raise ImportError("pytrends is not installed. Install with 'pip install pytrends'.")
        pytrend = TrendReq(hl="en-US", tz=360)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.info("Fetching Google Trends for %s (timeframe=%s, attempt %d/%d)...",
                        batch, timeframe, attempt + 1, max_retries)
            pytrend.build_payload(batch, timeframe=timeframe)
            df = pytrend.interest_over_time()

            if df is not None and not df.empty:
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

            # Save to cache if cache directory is configured
            if cache_dir is not None and df is not None and not df.empty:
                cache_file = cache_dir / get_cache_key(batch, timeframe)
                df.to_csv(cache_file)

            return df if df is not None else pd.DataFrame()

        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                logger.error("Failed to fetch Google Trends for %s after %d attempts: %s",
                             batch, max_retries, exc)
                raise
            delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, jitter)
            logger.warning("Error fetching %s (attempt %d/%d): %s. Retrying in %.2fs...",
                           batch, attempt + 1, max_retries, exc, delay)
            time.sleep(delay)

    if last_error:
        raise last_error
    return pd.DataFrame()


def fetch_trends(
    keywords: list[str],
    start: str,
    end: str,
    out_path: str | pathlib.Path = "datasets/trends/trends.csv",
    cache_dir: str | pathlib.Path | None = "datasets/trends/.cache",
    max_retries: int = 5,
    initial_delay: float = 3.0,
    request_delay: float = 1.0,
    pytrend: Any | None = None,
    anchor: str | None = None,
) -> pd.DataFrame:
    """Download and normalize monthly Google Trends data for a vocabulary.

    Batches keywords in groups of up to 5, aggregates data to monthly level,
    normalizes each month's relative interest into a probability distribution
    over the keywords, and writes out datasets/trends/trends.csv.

    Google Trends scales every request independently (its most popular term is
    100), so raw values from different batches are not comparable. When
    *anchor* is given, every request carries the anchor keyword plus 4
    vocabulary keywords, and each batch is rescaled so the anchor's monthly
    values match those of the first batch. This puts all keywords on one scale.

    Args:
        keywords: Full list of vocabulary keywords.
        start: Start date string (e.g. '2004-01-01' or '2004-01').
        end: End date string (e.g. '2004-12-31' or '2004-12').
        out_path: Destination path for normalized CSV (default datasets/trends/trends.csv).
        cache_dir: Cache directory for raw pytrends downloads.
        max_retries: Retry attempts on rate limits.
        initial_delay: Initial retry delay in seconds.
        request_delay: Polite sleep duration between batch requests.
        pytrend: Optional TrendReq instance or mock.
        anchor: Optional keyword included in every request to put batches on a
            common scale (recommended for more than 5 keywords). Pick a term of
            moderate, steady popularity so it neither dominates nor rounds to 0.

    Returns:
        DataFrame in long format with columns: month, keyword, prob.
    """
    if not keywords:
        raise ValueError("Vocabulary keywords list cannot be empty.")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_kws: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_kws.append(kw)

    timeframe = format_timeframe(start, end)
    cache_path = pathlib.Path(cache_dir) if cache_dir else None

    # Google Trends allows maximum 5 keywords per request; one slot goes to the anchor.
    batch_kws = [kw for kw in unique_kws if kw != anchor]
    batch_size = 4 if anchor else 5
    batches = [batch_kws[i:i + batch_size] for i in range(0, len(batch_kws), batch_size)]
    if anchor:
        batches = [[anchor] + b for b in batches]

    batch_dfs: list[pd.DataFrame] = []
    anchor_ref: pd.Series | None = None
    for i, batch in enumerate(batches):
        df = fetch_batch_with_retry(
            pytrend=pytrend,
            batch=batch,
            timeframe=timeframe,
            cache_dir=cache_path,
            max_retries=max_retries,
            initial_delay=initial_delay,
        )

        if df is not None and not df.empty:
            # Resample to monthly start (MS) and take mean
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            monthly = df.resample("MS").mean()
            if anchor and anchor in monthly.columns:
                if anchor_ref is None:
                    anchor_ref = monthly[anchor]
                else:
                    batch_anchor = monthly[anchor]
                    if batch_anchor.sum() > 0:
                        # Per-batch scale factor from the anchor's total interest.
                        monthly = monthly * (anchor_ref.sum() / batch_anchor.sum())
                    else:
                        logger.warning("Anchor %r is zero in batch %s; batch left unscaled.", anchor, batch)
                    monthly = monthly.drop(columns=[anchor])
            batch_dfs.append(monthly)

        if i < len(batches) - 1 and request_delay > 0:
            time.sleep(request_delay)

    if not batch_dfs:
        raise RuntimeError("No Google Trends data returned for the requested keywords.")

    # Combine all batch DataFrames across columns
    combined = pd.concat(batch_dfs, axis=1)
    # Deduplicate any duplicate columns if batches overlapped
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Format month strings as YYYY-MM
    combined.index = combined.index.strftime("%Y-%m")

    # Ensure all unique keywords are present as columns (fill missing with 0.0)
    for kw in unique_kws:
        if kw not in combined.columns:
            combined[kw] = 0.0

    combined = combined[unique_kws].fillna(0.0)

    # Normalize each month into a probability distribution over the keywords
    # If a month has sum == 0, distribute uniformly
    rows: list[dict[str, Any]] = []
    for month_str, row in combined.iterrows():
        vals = row.to_numpy(dtype=float)
        total = vals.sum()
        if total > 0:
            probs = vals / total
        else:
            probs = np.full(len(unique_kws), 1.0 / len(unique_kws))

        for kw, p in zip(unique_kws, probs):
            rows.append({
                "month": str(month_str),
                "keyword": kw,
                "prob": float(p),
            })

    result_df = pd.DataFrame(rows, columns=["month", "keyword", "prob"])
    result_df = result_df.sort_values(by=["month", "keyword"]).reset_index(drop=True)

    # Save to out_path
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    logger.info("Saved normalized trends to %s (%d rows)", out, len(result_df))

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize Google Trends monthly relative interest."
    )
    parser.add_argument(
        "--vocab-file", "-v",
        nargs="+",
        help="One or more vocabulary files (.txt, .json, .pkl, or .csv); keywords are merged.",
    )
    parser.add_argument(
        "--keywords", "-k",
        help="Comma-separated list of keywords (alternative to --vocab-file).",
    )
    parser.add_argument(
        "--start", "-s",
        default="2004-01-01",
        help="Start date (YYYY-MM-DD or YYYY-MM). Default: 2004-01-01.",
    )
    parser.add_argument(
        "--end", "-e",
        default="2004-12-31",
        help="End date (YYYY-MM-DD or YYYY-MM). Default: 2004-12-31.",
    )
    parser.add_argument(
        "--out", "-o",
        default="datasets/trends/trends.csv",
        help="Output CSV path (default: datasets/trends/trends.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        default="datasets/trends/.cache",
        help="Cache directory for raw pytrends queries (default: datasets/trends/.cache).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max retries for rate limits (default: 5).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Initial retry delay in seconds (default: 3.0).",
    )

    parser.add_argument(
        "--anchor",
        default=None,
        help="Keyword included in every request to put batches on one scale "
             "(recommended for more than 5 keywords).",
    )

    args = parser.parse_args(argv)

    keywords: list[str] = []
    if args.vocab_file:
        for vf in args.vocab_file:
            keywords.extend(load_vocab_file(vf))
    elif args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    else:
        parser.error("Either --vocab-file or --keywords must be provided.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    fetch_trends(
        keywords=keywords,
        start=args.start,
        end=args.end,
        out_path=args.out,
        cache_dir=args.cache_dir,
        max_retries=args.retries,
        initial_delay=args.delay,
        anchor=args.anchor,
    )


if __name__ == "__main__":
    main()
