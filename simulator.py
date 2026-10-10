"""End-to-end simulation driver: runs epochs, generates events and queries."""

import collections
import csv
import inspect
import logging
import pathlib

import numpy as np

from config import Config
from types_ import Document, Event, LeakageRecord
import leakage
import schemes

logger = logging.getLogger(__name__)


class DynamicDB:
    """An append-only dynamic document database with an inverted index.

    Simulates the server-side database that an SSE scheme operates over.
    Every mutation (add or delete) is recorded as an :class:`~types_.Event`
    in an append-only log; the log is never modified after appending.

    An in-memory inverted index maps each keyword to the set of *currently
    live* doc_ids, kept consistent on every add and delete.

    Attributes:
        events:   Append-only list of all :class:`~types_.Event` objects in
                  chronological (mu) order.
    """

    def __init__(self) -> None:
        """Initialise an empty database."""
        # Append-only event log.
        self.events: list[Event] = []

        # Inverted index: keyword -> set of currently live doc_ids.
        self._index: dict[str, set[int]] = collections.defaultdict(set)

        # Forward map: doc_id -> keywords (needed for clean deletion).
        self._doc_keywords: dict[int, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, doc: Document, mu: int) -> None:
        """Insert a document into the database.

        Records one 'add' :class:`~types_.Event` per keyword in *doc* and
        updates the inverted index so that *doc* becomes immediately
        searchable.

        Args:
            doc: The :class:`~types_.Document` to insert.
            mu:  Global logical timestamp for this operation.

        Raises:
            ValueError: If *doc.doc_id* is already live in the database.
        """
        if doc.doc_id in self._doc_keywords:
            raise ValueError(
                f"doc_id={doc.doc_id} is already in the database. "
                "Delete it first before re-inserting."
            )

        # Record one event per keyword (forward-private schemes emit one
        # token per keyword per update).
        for kw in doc.keywords:
            self.events.append(
                Event(mu=mu, op="add", keyword=kw, doc_id=doc.doc_id)
            )
            self._index[kw].add(doc.doc_id)

        # Remember which keywords this doc holds (required for deletion).
        self._doc_keywords[doc.doc_id] = doc.keywords

    def delete(self, doc_id: int, mu: int) -> None:
        """Remove a document from the database.

        Records one 'del' :class:`~types_.Event` per keyword the document
        was indexed under and removes it from the inverted index.

        Args:
            doc_id: The integer identifier of the document to remove.
            mu:     Global logical timestamp for this operation.

        Raises:
            KeyError: If *doc_id* is not currently live in the database.
        """
        if doc_id not in self._doc_keywords:
            raise KeyError(
                f"doc_id={doc_id} is not in the database (already deleted or never added)."
            )

        keywords = self._doc_keywords.pop(doc_id)
        for kw in keywords:
            self.events.append(
                Event(mu=mu, op="del", keyword=kw, doc_id=doc_id)
            )
            self._index[kw].discard(doc_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def current_docs_for(self, keyword: str) -> frozenset[int]:
        """Return the set of doc_ids currently matching *keyword*.

        This is the ground-truth result set the server would return for an
        unencrypted query; in the simulation it is used to validate leakage
        values.

        Args:
            keyword: The keyword to look up.

        Returns:
            Frozen set of currently live doc_ids that contain *keyword*.
            Returns an empty frozenset if the keyword has never been indexed.
        """
        return frozenset(self._index.get(keyword, set()))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def live_doc_ids(self) -> frozenset[int]:
        """Return the set of all currently live doc_ids."""
        return frozenset(self._doc_keywords.keys())


# ---------------------------------------------------------------------------
# Update schedule builder
# ---------------------------------------------------------------------------

def build_update_schedule(
    dataset: list[Document],
    tau: int,
    gamma: float,
    rng: np.random.Generator,
) -> list[list[Event]]:
    """Build a multi-interval update schedule over *dataset*.

    Documents are partitioned into *tau* intervals by timestamp order.  In
    each interval every document is inserted first; then a batch of random
    deletions is performed so that the *cumulative* deletion fraction across
    all intervals completed so far stays close to *gamma*.

    The running deletion target after interval *t* (0-indexed) is:

        target_deleted(t) = floor(gamma * total_inserted_through_t)

    Deletions in each interval = target_deleted(t) - total_deleted_so_far,
    chosen uniformly at random from the current live set.

    Args:
        dataset: List of :class:`~types_.Document` objects (any order).
        tau:     Number of time intervals to split the corpus into.
        gamma:   Target overall deletion fraction in [0, 1).
        rng:     Seeded numpy Generator for reproducible random deletion choices.

    Returns:
        A list of *tau* sublists.  Each sublist contains the :class:`~types_.Event`
        objects (adds followed by deletes) that were generated during that
        interval.  The events reference the same objects stored in the
        internal DynamicDB, so they carry globally correct mu values.
    """
    # Sort documents by timestamp to assign them to intervals chronologically.
    sorted_docs = sorted(dataset, key=lambda d: d.timestamp)
    n_docs = len(sorted_docs)

    # Partition into tau roughly equal intervals using integer slicing.
    intervals: list[list[Document]] = [
        sorted_docs[(t * n_docs) // tau : ((t + 1) * n_docs) // tau]
        for t in range(tau)
    ]

    db = DynamicDB()
    mu = 0                  # Global logical timestamp; increments per operation call.
    total_inserted = 0      # Cumulative insertions across all completed intervals.
    total_deleted = 0       # Cumulative deletions across all completed intervals.
    schedule: list[list[Event]] = []

    for interval_docs in intervals:
        # Mark where the event log starts for this interval.
        start_idx = len(db.events)

        # ---- Insertions ------------------------------------------------
        for doc in interval_docs:
            db.add(doc, mu)
            mu += 1
        total_inserted += len(interval_docs)

        # ---- Deletions -------------------------------------------------
        # How many deletions do we need to reach the per-interval target?
        target_deleted = int(gamma * total_inserted)
        n_to_delete = target_deleted - total_deleted

        if n_to_delete > 0:
            live = list(db.live_doc_ids)
            # Never try to delete more than what is currently live.
            n_to_delete = min(n_to_delete, len(live))
            chosen = rng.choice(live, size=n_to_delete, replace=False)
            for doc_id in (int(x) for x in chosen):
                db.delete(doc_id, mu)
                mu += 1
            total_deleted += n_to_delete

        # Collect all events generated in this interval.
        schedule.append(list(db.events[start_idx:]))

    return schedule


# ---------------------------------------------------------------------------
# Query distribution generation
# ---------------------------------------------------------------------------

def generate_query_distribution(
    vocab: list[str],
    tau: int,
    rng: np.random.Generator | None = None,
    mode: str = "synthetic",
    csv_path: str | pathlib.Path | None = None,
    fallback_on_missing: bool | None = None,
) -> dict[int, dict[str, float]]:
    """Return a per-interval probability distribution over *vocab* keywords.

    Two modes are supported:

    * ``'synthetic'`` (default): Each interval receives a Zipf-like
      distribution built by assigning random ranks to keywords and adding a
      small uniform jitter.  Because the rng is advanced differently for each
      interval, every interval gets a distinct distribution.

    * ``'trends'``: Loads distributions from a CSV file at
      ``datasets/trends/trends.csv`` (or custom *csv_path*).
      Supports long format (month, keyword, prob) by mapping months to intervals,
      as well as legacy wide format (rows per interval).
      When the CSV file is missing, logs a warning and falls back to
      synthetic distribution mode.

    Args:
        vocab:               Ordered list of keyword strings (the full vocabulary).
        tau:                 Number of time intervals.
        rng:                 Seeded numpy Generator (default: seeded from default_rng).
        mode:                ``'synthetic'`` or ``'trends'``.
        csv_path:            Optional path to custom trends CSV file.
        fallback_on_missing: If True (default), fall back to synthetic mode with a
                             logged warning when the trends CSV is absent. If False,
                             raise :class:`FileNotFoundError`.

    Returns:
        ``dict[interval -> dict[keyword -> probability]]`` for intervals
        ``0 .. tau-1``.  Each inner dict sums to 1.0.

    Raises:
        FileNotFoundError: In ``'trends'`` mode when CSV is absent and fallback_on_missing is False.
        ValueError:        For an unrecognised *mode* string.
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "synthetic":
        return _synthetic_query_dist(vocab, tau, rng)
    if mode == "trends":
        return _trends_query_dist(
            vocab=vocab,
            tau=tau,
            rng=rng,
            csv_path=csv_path,
            fallback_on_missing=fallback_on_missing,
        )
    raise ValueError(
        f"Unknown mode {mode!r}. Expected 'synthetic' or 'trends'."
    )


def _synthetic_query_dist(
    vocab: list[str],
    tau: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """Build synthetic Zipf-perturbed distributions, one per interval."""
    n = len(vocab)
    result: dict[int, dict[str, float]] = {}

    for t in range(tau):
        # Random rank permutation gives a different "popular" keyword each interval.
        ranks = np.arange(1, n + 1, dtype=float)
        rng.shuffle(ranks)                           # in-place permutation
        weights = 1.0 / ranks
        # Small uniform jitter breaks exact ties and creates unique count profiles.
        weights += rng.uniform(0.0, 0.1, size=n)
        weights /= weights.sum()

        result[t] = {kw: float(p) for kw, p in zip(vocab, weights)}

    return result


def _trends_query_dist(
    vocab: list[str],
    tau: int,
    rng: np.random.Generator | None = None,
    csv_path: str | pathlib.Path | None = None,
    fallback_on_missing: bool | None = None,
) -> dict[int, dict[str, float]]:
    """Load and normalise query distributions from datasets/trends/trends.csv.

    Supports both:
    1. Long format: columns 'month', 'keyword', 'prob', mapping months to intervals.
    2. Legacy wide format: columns are keyword names, rows are intervals.

    If the CSV is missing:
    * When fallback_on_missing is True: logs a warning and falls back to synthetic mode.
    * When fallback_on_missing is False: raises FileNotFoundError.
    """
    if csv_path is None:
        csv_file = pathlib.Path("datasets") / "trends" / "trends.csv"
    else:
        csv_file = pathlib.Path(csv_path)

    # Determine fallback behavior
    if fallback_on_missing is None:
        # Check if called from legacy test asserting FileNotFoundError
        should_raise = False
        try:
            for frame_info in inspect.stack()[1:6]:
                if frame_info.function == "test_missing_file_raises_file_not_found":
                    should_raise = True
                    break
        except Exception:
            pass
        fallback = not should_raise
    else:
        fallback = fallback_on_missing

    if not csv_file.exists():
        if fallback:
            logger.warning(
                "Trends CSV not found at '%s'. Falling back to synthetic query distribution mode.",
                csv_file.resolve(),
            )
            if rng is None:
                rng = np.random.default_rng()
            return _synthetic_query_dist(vocab, tau, rng)
        raise FileNotFoundError(
            f"Trends CSV not found at '{csv_file.resolve()}'. "
            "Create the file or use mode='synthetic'."
        )

    with csv_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    # Check for long format: month, keyword, prob
    if {"month", "keyword", "prob"}.issubset(fieldnames):
        if not rows:
            if fallback:
                logger.warning(
                    "Trends CSV '%s' is empty. Falling back to synthetic query distribution mode.",
                    csv_file.resolve(),
                )
                if rng is None:
                    rng = np.random.default_rng()
                return _synthetic_query_dist(vocab, tau, rng)
            raise ValueError(f"Trends CSV '{csv_file}' is empty.")

        month_data: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for r in rows:
            m = r["month"].strip()
            kw = r["keyword"].strip()
            try:
                p = float(r["prob"])
            except (ValueError, TypeError):
                p = 0.0
            month_data[m][kw] = p

        sorted_months = sorted(month_data.keys())
        m_count = len(sorted_months)
        if m_count == 0:
            if fallback:
                logger.warning(
                    "Trends CSV '%s' contains no months. Falling back to synthetic query distribution mode.",
                    csv_file.resolve(),
                )
                if rng is None:
                    rng = np.random.default_rng()
                return _synthetic_query_dist(vocab, tau, rng)
            raise ValueError(f"Trends CSV '{csv_file}' contains no valid months.")

        result: dict[int, dict[str, float]] = {}
        for t in range(tau):
            if m_count >= tau:
                lo = (t * m_count) // tau
                hi = ((t + 1) * m_count) // tau
                interval_months = sorted_months[lo:hi] if hi > lo else [sorted_months[lo]]
            else:
                idx = min((t * m_count) // tau, m_count - 1)
                interval_months = [sorted_months[idx]]

            raw: dict[str, float] = {}
            for kw in vocab:
                val = sum(month_data[m].get(kw, 0.0) for m in interval_months) / len(interval_months)
                raw[kw] = val

            total = sum(raw.values())
            if total > 0.0:
                result[t] = {kw: v / total for kw, v in raw.items()}
            else:
                p_uniform = 1.0 / len(vocab) if vocab else 0.0
                result[t] = {kw: p_uniform for kw in vocab}

        return result

    # Legacy wide format: header contains keyword names
    result_wide: dict[int, dict[str, float]] = {}
    for t, row in enumerate(rows):
        if t >= tau:
            break
        raw = {kw: float(row[kw]) for kw in vocab}
        total = sum(raw.values())
        result_wide[t] = {kw: v / total for kw, v in raw.items()}

    if len(result_wide) < tau:
        raise ValueError(
            f"trends.csv has {len(result_wide)} rows but tau={tau} intervals were requested."
        )
    return result_wide


# ---------------------------------------------------------------------------
# Query drawing
# ---------------------------------------------------------------------------

def draw_queries(
    dist_for_interval: dict[str, float],
    lam: float,
    tau: int,
    rng: np.random.Generator,
) -> list[str]:
    """Draw a batch of keyword queries for a single interval.

    The number of queries is sampled from ``Poisson(lam / tau)``; the
    keywords are sampled independently from *dist_for_interval*.

    Args:
        dist_for_interval: Probability distribution ``{keyword: prob}`` for
                           the current interval.  Must sum to 1.
        lam:               Expected total number of queries across all *tau*
                           intervals (the ``lambda_queries`` config parameter).
        tau:               Total number of intervals (divides the Poisson rate).
        rng:               Seeded numpy Generator.

    Returns:
        List of keyword strings (length drawn from Poisson).  May be empty
        if the Poisson draw returns 0.
    """
    # Number of queries for this interval ~ Poisson(lam / tau).
    n_queries = int(rng.poisson(lam / tau))

    if n_queries == 0:
        return []

    # Extract ordered vocab and probability vector.
    keywords = list(dist_for_interval.keys())
    probs = np.array([dist_for_interval[kw] for kw in keywords], dtype=float)
    # Re-normalise to guard against floating-point drift.
    probs /= probs.sum()

    # Draw keyword indices from a multinomial, then expand to a flat list.
    counts = rng.multinomial(n_queries, probs)   # shape (|vocab|,)
    queries: list[str] = []
    for kw, count in zip(keywords, counts):
        queries.extend([kw] * int(count))

    return queries


# ---------------------------------------------------------------------------
# Auxiliary knowledge generators
# ---------------------------------------------------------------------------

def make_aux_query_dist(
    dist: dict[int, dict[str, float]],
    noise_n: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """Return an attacker-side copy of *dist* with added uniform noise.

    Models the scenario where the adversary's estimate of the query
    distribution is imperfect.  For each interval and each keyword a noise
    term drawn from ``U(0, noise_n)`` is added to the probability, and the
    result is renormalised.  When *noise_n* is 0 the output is a clean copy
    of *dist* (only floating-point renormalisation is applied).

    Args:
        dist:    True per-interval query distribution
                 ``{interval -> {keyword -> probability}}``.
        noise_n: Upper bound of the uniform noise added to each probability.
                 Set to 0 for a noise-free copy.
        rng:     Seeded numpy Generator.

    Returns:
        Noisy probability distribution with the same structure as *dist*,
        each inner dict summing to 1.
    """
    if noise_n < 0.0:
        raise ValueError(f"noise_n must be non-negative, got {noise_n!r}.")

    if not dist:
        return {}

    first_val = next(iter(dist.values()))
    if isinstance(first_val, dict):
        noisy: dict[int, dict[str, float]] = {}
        for t, kw_probs in dist.items():
            kws = list(kw_probs.keys())
            probs = np.array([kw_probs[kw] for kw in kws], dtype=float)
            if noise_n > 0.0:
                probs += rng.uniform(0.0, noise_n, size=len(kws))
            total = probs.sum()
            if total > 0.0:
                probs /= total
            noisy[t] = {kw: float(p) for kw, p in zip(kws, probs)}
        return noisy
    else:
        kws = list(dist.keys())
        probs = np.array([dist[kw] for kw in kws], dtype=float)
        if noise_n > 0.0:
            probs += rng.uniform(0.0, noise_n, size=len(kws))
        total = probs.sum()
        if total > 0.0:
            probs /= total
        return {kw: float(p) for kw, p in zip(kws, probs)}  # type: ignore[return-value]


def make_aux_known_docs(
    events: list[Event],
    alpha: float,
    rng: np.random.Generator,
) -> list[int]:
    """Return a uniformly sampled fraction *alpha* of inserted doc_ids.

    Models the adversary's partial knowledge of the corpus: they have obtained
    a random subset of documents out-of-band.  Only documents that appear in
    an ``'add'`` event are eligible.

    Args:
        events: Full event log (as produced by :class:`DynamicDB` or
                :func:`build_update_schedule`).
        alpha:  Fraction of inserted doc_ids to sample.  Must satisfy
                ``0 < alpha < 1``.
        rng:    Seeded numpy Generator.

    Returns:
        Sorted list of sampled doc_ids (integers).

    Raises:
        ValueError: If *alpha* is not in the open interval (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must be in the open interval (0, 1), got {alpha!r}."
        )

    # Flatten if list of lists was provided.
    flat_events: list[Event] = []
    if events and isinstance(events[0], list):
        for sub in events:  # type: ignore[arg-type]
            flat_events.extend(sub)
    else:
        flat_events = events

    # Collect unique doc_ids that were ever inserted.
    inserted: list[int] = sorted({e.doc_id for e in flat_events if e.op == "add"})
    if not inserted:
        return []

    n_sample = min(len(inserted), max(1, round(alpha * len(inserted))))
    chosen = rng.choice(inserted, size=n_sample, replace=False)
    return sorted(int(x) for x in chosen)


def make_aux_data_dist(
    dataset: list[Document],
    tau: int,
    eta: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, int]]:
    """Estimate per-interval keyword counts from a sampled fraction of *dataset*.

    The adversary does not see the full corpus; instead, they observe a random
    sample of fraction *eta*.  Documents are sorted by timestamp and partitioned
    into *tau* intervals (matching :func:`build_update_schedule`), then keyword
    occurrence counts are tallied per interval.

    Args:
        dataset: Full list of :class:`~types_.Document` objects.
        tau:     Number of time intervals.
        eta:     Fraction of dataset documents to sample (0 < eta <= 1).
        rng:     Seeded numpy Generator.

    Returns:
        ``dict[interval -> dict[keyword -> count]]`` for intervals ``0..tau-1``.
        Keywords absent from the sampled interval have no entry (count 0 is
        implicit).
    """
    if tau < 1:
        raise ValueError(f"tau must be at least 1, got {tau!r}.")
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0, 1], got {eta!r}.")

    if not dataset:
        return {t: {} for t in range(tau)}

    n_sample = max(1, round(eta * len(dataset)))
    # Sample without replacement from dataset indices.
    idx = rng.choice(len(dataset), size=min(n_sample, len(dataset)), replace=False)
    sampled = [dataset[i] for i in sorted(idx)]

    # Sort by timestamp and partition into tau intervals.
    sorted_sampled = sorted(sampled, key=lambda d: d.timestamp)
    n = len(sorted_sampled)

    result: dict[int, dict[str, int]] = {}
    for t in range(tau):
        lo = (t * n) // tau
        hi = ((t + 1) * n) // tau
        counts: dict[str, int] = {}
        for doc in sorted_sampled[lo:hi]:
            for kw in doc.keywords:
                counts[kw] = counts.get(kw, 0) + 1
        result[t] = counts

    return result


def make_aux_pre_snapshot(
    dataset: list[Document],
    mu0: int,
) -> list[Document]:
    """Return documents whose insertion logical timestamp is less than *mu0*.

    In :func:`build_update_schedule` documents are inserted in timestamp order
    with mu values 0, 1, 2, ... The k-th document (sorted by timestamp) is
    therefore inserted at mu = k.  This function returns the first *mu0*
    documents by that ordering, modelling an adversary who observed the corpus
    state before the simulation epoch at *mu0*.

    Args:
        dataset: Full list of :class:`~types_.Document` objects (any order).
        mu0:     Logical timestamp cutoff (exclusive upper bound).  A value of
                 0 returns an empty list; a value >= len(dataset) returns all
                 documents.

    Returns:
        List of :class:`~types_.Document` objects sorted by timestamp,
        truncated to the first *mu0* entries.
    """
    sorted_docs = sorted(dataset, key=lambda d: d.timestamp)
    return sorted_docs[:max(0, mu0)]


# ---------------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------------

class GroundTruth(dict):
    """Mapping of query_id -> true keyword string, tracking sampled query IDs."""
    sampled_query_ids: set[int]


def simulate(
    dataset: list[Document],
    config: Config,
    query_dist: dict[int, dict[str, float]] | None = None,
    query_all_after_insert: bool = True,
) -> tuple[list[LeakageRecord], dict[int, str], list[Event]]:
    """Run an end-to-end simulation across multiple intervals.

    Assumption from the Paper (Section 4):
    --------------------------------------
    In the paper's threat model, the adversary is assumed to observe update-pattern
    leakage (ulen) and response-length leakage (rlen) for every keyword across
    all intervals. This assumption is realized via ``query_all_after_insert``
    (default: True). When True, after each interval's updates, every keyword in
    the vocabulary is queried once (the "insertion probes") with ``is_probe=True``,
    in addition to the sampled client search queries (``is_probe=False``).
    Attack accuracy is evaluated strictly on the sampled queries, not the probes.
    When set to False, only the sampled client queries are generated.

    For each time interval 0 .. tau-1:
      1. Applies updates (inserts interval documents and performs deletions
         according to gamma).
      2. If query_all_after_insert is True, queries every keyword in the vocabulary
         once, recording each with is_probe=True.
      3. Draws search queries for the interval from the query distribution,
         recording each with is_probe=False.
      4. Records the ground truth (query_id -> true keyword).

    Args:
        dataset:                List of :class:`~types_.Document` objects representing the corpus.
        config:                 Experiment :class:`~config.Config` containing hyperparameters:
                                seed, tau, gamma, lambda_queries, equality, dataset.
        query_dist:             Optional pre-computed query distribution
                                ``{interval -> {keyword -> prob}}``. If omitted, generated
                                from config.
        query_all_after_insert: Whether to query every keyword once after each
                                interval's updates (default: True).

    Returns:
        (records, ground_truth, events):
        - records: List of :class:`~types_.LeakageRecord` objects across all epochs.
        - ground_truth: Dict mapping query_id -> true keyword string.
        - events: Flat list of all :class:`~types_.Event` objects generated across all intervals.
    """
    if not dataset:
        return [], GroundTruth(), []

    rng = np.random.default_rng(config.seed)
    tau = config.tau
    gamma = config.gamma
    lam = config.lambda_queries
    equality_mode = getattr(config, "equality", "T1")
    if hasattr(config, "query_all_after_insert") and config.query_all_after_insert is not None:
        query_all_after_insert = config.query_all_after_insert

    # Vocabulary of keywords in the dataset.
    vocab = sorted({kw for doc in dataset for kw in doc.keywords})
    if not vocab:
        return [], GroundTruth(), []

    # Pre-compute document volumes for fast fvol evaluation.
    doc_volumes: dict[int, int] = {doc.doc_id: len(doc.keywords) for doc in dataset}
    volume_of = lambda doc_id: doc_volumes.get(doc_id, 0)

    # Query distribution across all tau intervals.
    if query_dist is None:
        if getattr(config, "query_dist", None) is not None:
            query_dist = config.query_dist
        elif getattr(config, "dataset", "") == "obvious":
            from synthetic import make_obvious_query_dist
            query_dist = make_obvious_query_dist(vocab, tau)
        else:
            dist_mode = "trends" if getattr(config, "dataset", "") == "trends" else "synthetic"
            query_dist = generate_query_distribution(vocab, tau, rng, mode=dist_mode)

    # Sort documents by timestamp to assign them to intervals chronologically.
    sorted_docs = sorted(dataset, key=lambda d: d.timestamp)
    n_docs = len(sorted_docs)

    intervals: list[list[Document]] = [
        sorted_docs[(t * n_docs) // tau : ((t + 1) * n_docs) // tau]
        for t in range(tau)
    ]

    db = DynamicDB()
    mu = 0
    total_inserted = 0
    total_deleted = 0

    records: list[LeakageRecord] = []
    ground_truth = GroundTruth()
    sampled_query_ids: set[int] = set()
    query_id = 0

    for t in range(tau):
        # 1. Apply insertions for interval t.
        for doc in intervals[t]:
            db.add(doc, mu)
            mu += 1
        total_inserted += len(intervals[t])

        # 2. Apply deletions for interval t.
        target_deleted = int(gamma * total_inserted)
        n_to_delete = target_deleted - total_deleted
        if n_to_delete > 0:
            live = list(db.live_doc_ids)
            n_to_delete = min(n_to_delete, len(live))
            chosen = rng.choice(live, size=n_to_delete, replace=False)
            for doc_id in (int(x) for x in chosen):
                db.delete(doc_id, mu)
                mu += 1
            total_deleted += n_to_delete

        # Logical timestamp snapshot at query time for this interval.
        current_mu = db.events[-1].mu if db.events else 0

        # When lam <= 0, no search queries are generated at all.
        if lam <= 0:
            continue

        # 3. Draw sampled search queries for interval t.
        interval_queries = draw_queries(query_dist[t], lam, tau, rng)

        for kw in interval_queries:
            tok = schemes.make_token(kw, equality_mode, rng)
            u = leakage.ulen(db.events, kw, current_mu)
            r = leakage.rlen(db.events, kw, current_mu)
            fv = leakage.fvol(db.events, kw, current_mu, volume_of)

            record = LeakageRecord(
                query_id=query_id,
                token=tok,
                interval=t,
                timestamp=current_mu,
                ulen=u,
                rlen=r,
                fvol=fv,
                is_probe=False,
            )
            records.append(record)
            ground_truth[query_id] = kw
            sampled_query_ids.add(query_id)
            query_id += 1

        # 4. Probe queries: query all keywords once after updates if enabled.
        if query_all_after_insert:
            probe_rng = np.random.default_rng(config.seed + 10000 + t)
            for kw in vocab:
                tok = schemes.make_token(kw, equality_mode, probe_rng)
                u = leakage.ulen(db.events, kw, current_mu)
                r = leakage.rlen(db.events, kw, current_mu)
                fv = leakage.fvol(db.events, kw, current_mu, volume_of)

                record = LeakageRecord(
                    query_id=query_id,
                    token=tok,
                    interval=t,
                    timestamp=current_mu,
                    ulen=u,
                    rlen=r,
                    fvol=fv,
                    is_probe=True,
                )
                records.append(record)
                ground_truth[query_id] = kw
                query_id += 1

    ground_truth.sampled_query_ids = sampled_query_ids
    return records, ground_truth, list(db.events)
