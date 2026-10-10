"""Tests for DynamicDB in simulator.py."""

from datetime import datetime, timezone

import pytest

from simulator import DynamicDB
from types_ import Document
import leakage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)  # fixed timestamp for all docs


def doc(doc_id: int, *keywords: str) -> Document:
    """Convenience factory for test Documents."""
    return Document(doc_id=doc_id, keywords=frozenset(keywords), timestamp=_TS)


def fresh_db() -> DynamicDB:
    """Return a new empty DynamicDB."""
    return DynamicDB()


# ---------------------------------------------------------------------------
# 1. add – index updates
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_makes_doc_retrievable(self) -> None:
        """After add, current_docs_for should return the doc_id for each keyword."""
        db = fresh_db()
        db.add(doc(1, "kw0", "kw1"), mu=0)
        assert 1 in db.current_docs_for("kw0")
        assert 1 in db.current_docs_for("kw1")

    def test_add_appends_one_event_per_keyword(self) -> None:
        """add must append exactly one 'add' Event per keyword."""
        db = fresh_db()
        db.add(doc(1, "kw0", "kw1", "kw2"), mu=0)
        add_events = [e for e in db.events if e.op == "add"]
        assert len(add_events) == 3
        kws = {e.keyword for e in add_events}
        assert kws == {"kw0", "kw1", "kw2"}

    def test_add_event_fields_are_correct(self) -> None:
        """Event fields (mu, op, keyword, doc_id) must match the call arguments."""
        db = fresh_db()
        db.add(doc(7, "kw3"), mu=42)
        e = db.events[0]
        assert e.mu == 42
        assert e.op == "add"
        assert e.keyword == "kw3"
        assert e.doc_id == 7

    def test_add_multiple_docs_same_keyword(self) -> None:
        """Multiple documents sharing a keyword must all appear in its result set."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.add(doc(3, "kw0"), mu=2)
        assert db.current_docs_for("kw0") == frozenset({1, 2, 3})

    def test_add_duplicate_raises(self) -> None:
        """Re-inserting an already-live doc_id must raise ValueError."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        with pytest.raises(ValueError, match="already in the database"):
            db.add(doc(1, "kw0"), mu=1)

    def test_unknown_keyword_returns_empty(self) -> None:
        """current_docs_for an unseen keyword must return an empty frozenset."""
        db = fresh_db()
        assert db.current_docs_for("kw_missing") == frozenset()


# ---------------------------------------------------------------------------
# 2. delete – index removal
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_from_all_keywords(self) -> None:
        """After delete, the doc_id must be absent from every keyword it held."""
        db = fresh_db()
        db.add(doc(1, "kw0", "kw1", "kw2"), mu=0)
        db.delete(1, mu=1)
        assert 1 not in db.current_docs_for("kw0")
        assert 1 not in db.current_docs_for("kw1")
        assert 1 not in db.current_docs_for("kw2")

    def test_delete_appends_one_event_per_keyword(self) -> None:
        """delete must append exactly one 'del' Event per keyword the doc held."""
        db = fresh_db()
        db.add(doc(1, "kw0", "kw1"), mu=0)
        db.delete(1, mu=1)
        del_events = [e for e in db.events if e.op == "del"]
        assert len(del_events) == 2
        assert {e.keyword for e in del_events} == {"kw0", "kw1"}

    def test_delete_event_fields_are_correct(self) -> None:
        """del Event fields must reflect the call arguments."""
        db = fresh_db()
        db.add(doc(5, "kw4"), mu=10)
        db.delete(5, mu=20)
        del_evt = next(e for e in db.events if e.op == "del")
        assert del_evt.mu == 20
        assert del_evt.op == "del"
        assert del_evt.keyword == "kw4"
        assert del_evt.doc_id == 5

    def test_delete_nonexistent_raises(self) -> None:
        """Deleting a doc_id that is not live must raise KeyError."""
        db = fresh_db()
        with pytest.raises(KeyError):
            db.delete(999, mu=0)

    def test_delete_twice_raises(self) -> None:
        """Deleting the same doc_id twice must raise KeyError on the second call."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.delete(1, mu=1)
        with pytest.raises(KeyError):
            db.delete(1, mu=2)

    def test_delete_leaves_other_docs_intact(self) -> None:
        """Deleting one doc must not affect other documents sharing the same keyword."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.delete(1, mu=2)
        assert db.current_docs_for("kw0") == frozenset({2})

    def test_readd_after_delete(self) -> None:
        """A document may be re-added after deletion."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.delete(1, mu=1)
        db.add(doc(1, "kw0"), mu=2)   # should not raise
        assert 1 in db.current_docs_for("kw0")


# ---------------------------------------------------------------------------
# 3. Event log is append-only
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_events_list_grows_monotonically(self) -> None:
        """The events list must only grow; earlier entries must be unchanged."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        first_event = db.events[0]
        db.add(doc(2, "kw1"), mu=1)
        db.delete(1, mu=2)
        # First event should be identical (same object reference).
        assert db.events[0] is first_event

    def test_event_mu_values_match_call_order(self) -> None:
        """Events must appear in the order they were created."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=10)
        db.add(doc(2, "kw1"), mu=20)
        db.delete(1, mu=30)
        mus = [e.mu for e in db.events]
        assert mus == sorted(mus), "Events are not in mu order."

    def test_total_event_count(self) -> None:
        """Total events = sum of keyword counts across all add and delete calls."""
        db = fresh_db()
        db.add(doc(1, "kw0", "kw1"), mu=0)       # 2 events
        db.add(doc(2, "kw2", "kw3", "kw4"), mu=1) # 3 events
        db.delete(1, mu=2)                          # 2 events
        assert len(db.events) == 7


# ---------------------------------------------------------------------------
# 4. rlen consistency with leakage.rlen
# ---------------------------------------------------------------------------

class TestRlenConsistency:
    """rlen from the leakage module must match the live result-set size."""

    def test_rlen_matches_current_docs_after_adds(self) -> None:
        """leakage.rlen == len(current_docs_for) after pure insertions."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.add(doc(3, "kw0"), mu=2)
        mu_now = 2
        assert leakage.rlen(db.events, "kw0", mu_now) == len(db.current_docs_for("kw0"))

    def test_rlen_matches_after_mixed_ops(self) -> None:
        """leakage.rlen == len(current_docs_for) after adds and deletes."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.add(doc(3, "kw0"), mu=2)
        db.delete(2, mu=3)
        mu_now = 3
        assert leakage.rlen(db.events, "kw0", mu_now) == len(db.current_docs_for("kw0"))
        assert leakage.rlen(db.events, "kw0", mu_now) == 2

    def test_rlen_at_historical_snapshot(self) -> None:
        """leakage.rlen at an earlier mu should reflect that historical state."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.delete(1, mu=2)
        # At mu=1 both docs were live.
        assert leakage.rlen(db.events, "kw0", mu=1) == 2
        # At mu=2 only doc 2 is live.
        assert leakage.rlen(db.events, "kw0", mu=2) == 1

    def test_rlen_zero_for_unknown_keyword(self) -> None:
        """leakage.rlen for a keyword with no events must be 0."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        assert leakage.rlen(db.events, "kw_missing", mu=0) == 0

    def test_rlen_zero_after_all_deleted(self) -> None:
        """leakage.rlen must be 0 when every inserted document has been deleted."""
        db = fresh_db()
        db.add(doc(1, "kw0"), mu=0)
        db.add(doc(2, "kw0"), mu=1)
        db.delete(1, mu=2)
        db.delete(2, mu=3)
        assert leakage.rlen(db.events, "kw0", mu=3) == 0
        assert len(db.current_docs_for("kw0")) == 0

    @pytest.mark.parametrize("n_adds,n_dels", [(5, 0), (5, 2), (5, 5), (10, 7)])
    def test_rlen_parametric(self, n_adds: int, n_dels: int) -> None:
        """Parametric check: rlen == n_adds - n_dels for various add/del counts."""
        db = fresh_db()
        for i in range(n_adds):
            db.add(doc(i, "kw0"), mu=i)
        for i in range(n_dels):
            db.delete(i, mu=n_adds + i)
        mu_now = n_adds + n_dels - 1 if (n_adds + n_dels) > 0 else 0
        expected = n_adds - n_dels
        assert leakage.rlen(db.events, "kw0", mu_now) == expected
        assert len(db.current_docs_for("kw0")) == expected
