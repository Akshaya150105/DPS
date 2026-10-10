"""End-to-end simulation driver: runs epochs, generates events and queries."""

import collections

import numpy as np

from types_ import Document, Event


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
    rng: np.random.Generator,
    mode: str = "synthetic",
) -> dict[int, dict[str, float]]:
    """Return a per-interval probability distribution over *vocab* keywords.

    Two modes are supported:

    * ``'synthetic'`` (default): Each interval receives a Zipf-like
      distribution built by assigning random ranks to keywords and adding a
      small uniform jitter.  Because the rng is advanced differently for each
      interval, every interval gets a distinct distribution.

    * ``'trends'``: Loads pre-computed distributions from a CSV file at
      ``datasets/trends/trends.csv``.  The CSV must have a header row whose
      columns are keyword names and exactly *tau* data rows (one per
      interval).  Raises :class:`FileNotFoundError` with a descriptive message
      if the file is absent.

    Args:
        vocab: Ordered list of keyword strings (the full vocabulary).
        tau:   Number of time intervals.
        rng:   Seeded numpy Generator (only used in ``'synthetic'`` mode).
        mode:  ``'synthetic'`` or ``'trends'``.

    Returns:
        ``dict[interval -> dict[keyword -> probability]]`` for intervals
        ``0 .. tau-1``.  Each inner dict sums to 1.0.

    Raises:
        FileNotFoundError: In ``'trends'`` mode when the CSV is not present.
        ValueError:        For an unrecognised *mode* string.
    """
    if mode == "synthetic":
        return _synthetic_query_dist(vocab, tau, rng)
    if mode == "trends":
        return _trends_query_dist(vocab, tau)
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
) -> dict[int, dict[str, float]]:
    """Load and normalise query distributions from datasets/trends/trends.csv.

    Expected CSV layout::

        kw0,kw1,...,kw9          # header – keyword names
        12,5,3,...               # row 0: raw counts for interval 0
        8,10,2,...               # row 1: raw counts for interval 1
        ...                      # tau rows total

    Each row is normalised to a probability distribution.
    """
    import pathlib
    import csv

    csv_path = pathlib.Path("datasets") / "trends" / "trends.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Trends CSV not found at '{csv_path.resolve()}'. "
            "Create the file or use mode='synthetic'."
        )

    result: dict[int, dict[str, float]] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for t, row in enumerate(reader):
            if t >= tau:
                break
            raw = {kw: float(row[kw]) for kw in vocab}
            total = sum(raw.values())
            result[t] = {kw: v / total for kw, v in raw.items()}

    if len(result) < tau:
        raise ValueError(
            f"trends.csv has {len(result)} rows but tau={tau} intervals were requested."
        )
    return result


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
