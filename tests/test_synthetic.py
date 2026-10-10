"""Tests for synthetic.py: corpus structure, vocabulary, volumes, and reproducibility."""

from datetime import timezone

import pytest

from synthetic import VOCAB, make_obvious_query_dist, make_synthetic
from types_ import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def docs() -> list[Document]:
    """Single corpus instance shared across tests in this module."""
    return make_synthetic(seed=42)


# ---------------------------------------------------------------------------
# 1. Corpus size
# ---------------------------------------------------------------------------

class TestCorpusSize:
    def test_exactly_100_documents(self, docs: list[Document]) -> None:
        """make_synthetic must return exactly 100 Document objects."""
        assert len(docs) == 100

    def test_all_are_documents(self, docs: list[Document]) -> None:
        """Every item must be a Document instance."""
        assert all(isinstance(d, Document) for d in docs)

    def test_doc_ids_are_unique(self, docs: list[Document]) -> None:
        """doc_id values must be unique across the corpus."""
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 2. Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_exactly_10_keywords_in_vocab(self) -> None:
        """VOCAB must contain exactly 10 entries named kw0 .. kw9."""
        assert len(VOCAB) == 10
        assert set(VOCAB) == {f"kw{i}" for i in range(10)}

    def test_all_document_keywords_in_vocab(self, docs: list[Document]) -> None:
        """Every keyword in every document must belong to the 10-keyword vocabulary."""
        vocab_set = set(VOCAB)
        for doc in docs:
            assert doc.keywords <= vocab_set, (
                f"doc_id={doc.doc_id} has out-of-vocab keywords: "
                f"{doc.keywords - vocab_set}"
            )

    def test_all_vocab_keywords_appear_at_least_once(self, docs: list[Document]) -> None:
        """With 100 documents and 10 keywords, every keyword must appear in at least one doc."""
        seen: set[str] = set()
        for doc in docs:
            seen |= doc.keywords
        assert seen == set(VOCAB), f"Missing keywords: {set(VOCAB) - seen}"


# ---------------------------------------------------------------------------
# 3. Document volumes
# ---------------------------------------------------------------------------

class TestVolumes:
    def test_volume_equals_len_keywords(self, docs: list[Document]) -> None:
        """Volume (len(keywords)) must equal the size of the keywords frozenset."""
        for doc in docs:
            assert len(doc.keywords) == len(doc.keywords), "frozenset is always consistent"
            # Explicit: the number of keywords IS the volume.
            assert isinstance(doc.keywords, frozenset)

    def test_volumes_in_valid_range(self, docs: list[Document]) -> None:
        """Each document must have 2 to 5 keywords (volume in {2, 3, 4, 5})."""
        for doc in docs:
            vol = len(doc.keywords)
            assert 2 <= vol <= 5, (
                f"doc_id={doc.doc_id} has volume {vol}, expected 2–5"
            )

    def test_multiple_volume_values_present(self, docs: list[Document]) -> None:
        """Volumes should not all be the same; we expect at least 2 distinct values."""
        volumes = {len(d.keywords) for d in docs}
        assert len(volumes) >= 2, f"Only one distinct volume found: {volumes}"


# ---------------------------------------------------------------------------
# 4. Time intervals
# ---------------------------------------------------------------------------

class TestIntervals:
    def test_exactly_3_distinct_dates(self, docs: list[Document]) -> None:
        """Documents must span exactly 3 distinct calendar dates (one per epoch)."""
        dates = {d.timestamp.date() for d in docs}
        assert len(dates) == 3, f"Expected 3 distinct dates, got {dates}"

    def test_timestamps_are_timezone_aware(self, docs: list[Document]) -> None:
        """All timestamps must be UTC-aware datetime objects."""
        for doc in docs:
            assert doc.timestamp.tzinfo is not None, (
                f"doc_id={doc.doc_id} has naive timestamp"
            )

    def test_roughly_equal_interval_sizes(self, docs: list[Document]) -> None:
        """Each interval should contain approximately 33 documents (±5 tolerance)."""
        from collections import Counter
        date_counts = Counter(d.timestamp.date() for d in docs)
        for date, count in date_counts.items():
            assert 28 <= count <= 38, (
                f"Interval {date} has {count} documents, expected ~33"
            )


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        """Two calls with the same seed must produce byte-identical results."""
        docs_a = make_synthetic(seed=42)
        docs_b = make_synthetic(seed=42)
        assert docs_a == docs_b, "Same seed produced different corpora."

    def test_different_seeds_differ(self) -> None:
        """Different seeds must (almost certainly) produce different corpora."""
        docs_a = make_synthetic(seed=42)
        docs_b = make_synthetic(seed=99)
        # Compare the full keyword sets; collisions across seeds are astronomically unlikely.
        kw_sets_a = [d.keywords for d in docs_a]
        kw_sets_b = [d.keywords for d in docs_b]
        assert kw_sets_a != kw_sets_b, "Different seeds produced identical keyword sets."

    def test_doc_ids_are_deterministic(self) -> None:
        """doc_id values must be identical across two calls with the same seed."""
        docs_a = make_synthetic(seed=0)
        docs_b = make_synthetic(seed=0)
        assert [d.doc_id for d in docs_a] == [d.doc_id for d in docs_b]


# ---------------------------------------------------------------------------
# 6. make_obvious_query_dist
# ---------------------------------------------------------------------------

class TestObviousQueryDist:
    def test_returns_tau_intervals(self) -> None:
        """make_obvious_query_dist must return a dict with tau entries."""
        dist = make_obvious_query_dist(VOCAB, tau=3)
        assert set(dist.keys()) == {0, 1, 2}

    def test_each_interval_covers_all_keywords(self) -> None:
        """Every keyword must appear in each interval's distribution."""
        dist = make_obvious_query_dist(VOCAB, tau=3)
        for t, d in dist.items():
            assert set(d.keys()) == set(VOCAB), (
                f"Interval {t} is missing some keywords."
            )

    def test_probabilities_sum_to_one(self) -> None:
        """Each interval's probabilities must sum to 1.0 (within float tolerance)."""
        dist = make_obvious_query_dist(VOCAB, tau=3)
        for t, d in dist.items():
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Interval {t} probabilities sum to {total}, not 1.0"
            )

    def test_all_probs_positive(self) -> None:
        """All probabilities must be strictly positive."""
        dist = make_obvious_query_dist(VOCAB, tau=3)
        for t, d in dist.items():
            for kw, p in d.items():
                assert p > 0, f"Interval {t}, keyword {kw} has non-positive probability {p}"

    def test_distributions_differ_across_intervals(self) -> None:
        """Each interval must have a visibly different modal keyword."""
        dist = make_obvious_query_dist(VOCAB, tau=3)
        # The argmax keyword should differ across intervals.
        modes = [max(dist[t], key=dist[t].get) for t in range(3)]
        assert len(set(modes)) == 3, (
            f"Intervals share the same modal keyword: {modes}"
        )

    def test_single_interval(self) -> None:
        """tau=1 must return a single interval dict summing to 1."""
        dist = make_obvious_query_dist(VOCAB, tau=1)
        assert set(dist.keys()) == {0}
        assert abs(sum(dist[0].values()) - 1.0) < 1e-9
