"""Tests for leakage.py: algebra identities and RSP boundary cases."""

import collections
import pytest
import numpy as np

from types_ import Event
import leakage


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded numpy Generator for reproducible randomness."""
    return np.random.default_rng(seed)


def build_random_log(
    rng: np.random.Generator,
    keyword: str = "kw",
    n_add: int | None = None,
    n_del: int | None = None,
) -> tuple[list[Event], int, int]:
    """Build a random event log for one keyword.

    Args:
        rng:     Seeded numpy Generator.
        keyword: Keyword string to use.
        n_add:   Number of insertions; random in [5, 20] if None.
        n_del:   Number of deletions (must be <= n_add); random if None.

    Returns:
        (events, a, d) – the event list and the true insertion / deletion counts.
    """
    if n_add is None:
        n_add = int(rng.integers(5, 21))
    if n_del is None:
        n_del = int(rng.integers(0, n_add + 1))

    events: list[Event] = []
    mu_counter = 0
    available_docs: list[int] = []

    # Insertions
    doc_ids = list(range(n_add))
    for doc_id in doc_ids:
        events.append(Event(mu=mu_counter, op="add", keyword=keyword, doc_id=doc_id))
        mu_counter += 1
        available_docs.append(doc_id)

    # Deletions (sample without replacement from inserted docs)
    del_docs = rng.choice(available_docs, size=n_del, replace=False).tolist()
    for doc_id in del_docs:
        events.append(Event(mu=mu_counter, op="del", keyword=keyword, doc_id=doc_id))
        mu_counter += 1

    # Snapshot at the very last event
    max_mu = mu_counter - 1
    return events, n_add, n_del


# ---------------------------------------------------------------------------
# 1. n == a - d  and  u == a + d  on random logs
# ---------------------------------------------------------------------------

class TestAlgebraIdentitiesOnRandomLogs:
    """Verify that rlen = a - d and ulen = a + d hold on randomly generated logs."""

    @pytest.mark.parametrize("seed", range(10))
    def test_rlen_equals_a_minus_d(self, seed: int) -> None:
        """rlen (n) should equal a - d for any random event log."""
        rng = make_rng(seed)
        events, a, d = build_random_log(rng)
        mu = max(e.mu for e in events)

        n_observed = leakage.rlen(events, "kw", mu)
        assert n_observed == a - d, (
            f"seed={seed}: rlen={n_observed} but a-d={a-d} (a={a}, d={d})"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_ulen_equals_a_plus_d(self, seed: int) -> None:
        """ulen (u) should equal a + d for any random event log."""
        rng = make_rng(seed)
        events, a, d = build_random_log(rng)
        mu = max(e.mu for e in events)

        u_observed = leakage.ulen(events, "kw", mu)
        assert u_observed == a + d, (
            f"seed={seed}: ulen={u_observed} but a+d={a+d} (a={a}, d={d})"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_ilen_dlen_sum(self, seed: int) -> None:
        """ilen + dlen must equal ulen (every event is either add or del)."""
        rng = make_rng(seed)
        events, _, _ = build_random_log(rng)
        mu = max(e.mu for e in events)

        assert leakage.ilen(events, "kw", mu) + leakage.dlen(events, "kw", mu) \
               == leakage.ulen(events, "kw", mu)


def test_backward_privacy_leakages_recover_their_documented_values() -> None:
    """Type I/II reveal (u,n); Type III reveals (n,d), each deriving a and u."""
    rng = make_rng(909)
    for _ in range(20):
        events, true_a, true_d = build_random_log(rng)
        mu = max(event.mu for event in events)
        observed_u = leakage.ulen(events, "kw", mu)
        observed_n = leakage.rlen(events, "kw", mu)
        observed_d = leakage.dlen(events, "kw", mu)

        type_i_leakage = (observed_u, observed_n)
        type_ii_leakage = (observed_n, observed_u)
        type_iii_leakage = (observed_n, observed_d)
        type_i = leakage.derive_insert_delete(*type_i_leakage)
        type_ii = leakage.derive_insert_delete(type_ii_leakage[1], type_ii_leakage[0])
        type_iii = leakage.derive_from_n_d(*type_iii_leakage)
        assert type_i == type_ii == (true_a, true_d)
        assert type_iii == (true_a, true_a + true_d)


# ---------------------------------------------------------------------------
# 2. derive_insert_delete: (a, d) from (u, n)
# ---------------------------------------------------------------------------

class TestDeriveInsertDelete:
    """Verify derive_insert_delete recovers (a, d) from (u, n)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_roundtrip_from_u_n(self, seed: int) -> None:
        """derive_insert_delete(u, n) should return the original (a, d)."""
        rng = make_rng(seed + 100)
        events, a_true, d_true = build_random_log(rng)
        mu = max(e.mu for e in events)

        u = leakage.ulen(events, "kw", mu)
        n = leakage.rlen(events, "kw", mu)

        a_derived, d_derived = leakage.derive_insert_delete(u, n)
        assert a_derived == a_true, f"seed={seed}: a_derived={a_derived} != a_true={a_true}"
        assert d_derived == d_true, f"seed={seed}: d_derived={d_derived} != d_true={d_true}"

    def test_raises_on_parity_mismatch(self) -> None:
        """derive_insert_delete should raise ValueError when u and n have different parities."""
        with pytest.raises(ValueError, match="parities"):
            leakage.derive_insert_delete(u=5, n=2)  # 5+2=7, odd → no integer solution

    def test_zero_deletions(self) -> None:
        """When d=0, a=u=n."""
        a, d = leakage.derive_insert_delete(u=8, n=8)
        assert a == 8 and d == 0

    def test_all_deleted(self) -> None:
        """When n=0 (all deleted), a == d == u/2."""
        a, d = leakage.derive_insert_delete(u=6, n=0)
        assert a == 3 and d == 3


# ---------------------------------------------------------------------------
# 3. derive_from_n_d: (a, u) from (n, d)
# ---------------------------------------------------------------------------

class TestDeriveFromNAndD:
    """Verify derive_from_n_d recovers (a, u) from (n, d)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_roundtrip_from_n_d(self, seed: int) -> None:
        """derive_from_n_d(n, d) should return the original (a, u)."""
        rng = make_rng(seed + 200)
        events, a_true, d_true = build_random_log(rng)
        mu = max(e.mu for e in events)

        n = leakage.rlen(events, "kw", mu)
        d = leakage.dlen(events, "kw", mu)
        u_true = leakage.ulen(events, "kw", mu)

        a_derived, u_derived = leakage.derive_from_n_d(n, d)
        assert a_derived == a_true, f"seed={seed}: a_derived={a_derived} != a_true={a_true}"
        assert u_derived == u_true, f"seed={seed}: u_derived={u_derived} != u_true={u_true}"

    def test_zero_deletions(self) -> None:
        """When d=0, a=n and u=n."""
        a, u = leakage.derive_from_n_d(n=7, d=0)
        assert a == 7 and u == 7

    def test_known_values(self) -> None:
        """Manual check: n=3, d=2 => a=5, u=7."""
        a, u = leakage.derive_from_n_d(n=3, d=2)
        assert a == 5 and u == 7


# ---------------------------------------------------------------------------
# 4. rsp boundary cases
# ---------------------------------------------------------------------------

class TestRsp:
    """Verify response-set similarity (RSP) on boundary and typical inputs."""

    def test_identical_multisets_give_one(self) -> None:
        """rsp(A, A) must equal 1.0 for any non-empty multiset A."""
        fv = collections.Counter({1: 3, 2: 2, 5: 1})
        assert leakage.rsp(fv, fv) == pytest.approx(1.0)

    def test_identical_single_element(self) -> None:
        """rsp returns 1.0 for two equal single-element Counters."""
        fv = collections.Counter({4: 1})
        assert leakage.rsp(fv, fv) == pytest.approx(1.0)

    def test_disjoint_multisets_give_zero(self) -> None:
        """rsp(A, B) must equal 0.0 when A and B share no volume bucket."""
        fv_a = collections.Counter({1: 5})
        fv_b = collections.Counter({2: 3})
        assert leakage.rsp(fv_a, fv_b) == pytest.approx(0.0)

    def test_both_empty_give_one(self) -> None:
        """Two empty result sets are considered identical (rsp=1)."""
        assert leakage.rsp(collections.Counter(), collections.Counter()) == pytest.approx(1.0)

    def test_one_empty_other_nonempty_give_zero(self) -> None:
        """rsp between empty and non-empty must equal 0.0."""
        fv_a = collections.Counter({1: 2})
        fv_b = collections.Counter()
        assert leakage.rsp(fv_a, fv_b) == pytest.approx(0.0)
        assert leakage.rsp(fv_b, fv_a) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Manual Jaccard computation for a known partially-overlapping pair."""
        # A = {1:2, 2:1}, B = {1:1, 3:1}
        # intersection: {1: min(2,1)} = {1:1}  -> size 1
        # union:        {1: max(2,1), 2:1, 3:1} = {1:2, 2:1, 3:1}  -> size 4
        # rsp = 1/4 = 0.25
        fv_a = collections.Counter({1: 2, 2: 1})
        fv_b = collections.Counter({1: 1, 3: 1})
        assert leakage.rsp(fv_a, fv_b) == pytest.approx(0.25)

    def test_symmetry(self) -> None:
        """rsp(A, B) == rsp(B, A) for arbitrary multisets."""
        rng = make_rng(99)
        volumes = rng.integers(1, 6, size=8).tolist()
        fv_a = collections.Counter(volumes[:4])
        fv_b = collections.Counter(volumes[4:])
        assert leakage.rsp(fv_a, fv_b) == pytest.approx(leakage.rsp(fv_b, fv_a))


# ---------------------------------------------------------------------------
# 5. Snapshot filtering (mu cutoff)
# ---------------------------------------------------------------------------

class TestMuCutoff:
    """Events beyond mu must be invisible to all leakage functions."""

    def test_partial_snapshot(self) -> None:
        """Only events up to mu should be counted."""
        events = [
            Event(mu=0, op="add", keyword="k", doc_id=0),
            Event(mu=1, op="add", keyword="k", doc_id=1),
            Event(mu=2, op="del", keyword="k", doc_id=0),
            Event(mu=3, op="add", keyword="k", doc_id=2),  # beyond mu=2
        ]
        # At mu=2: a=2, d=1 => n=1, u=3
        assert leakage.rlen(events, "k", mu=2) == 1
        assert leakage.ulen(events, "k", mu=2) == 3
        # At mu=3: a=3, d=1 => n=2, u=4
        assert leakage.rlen(events, "k", mu=3) == 2
        assert leakage.ulen(events, "k", mu=3) == 4

    def test_no_events_gives_zero(self) -> None:
        """With no matching events, all counts are zero."""
        events: list[Event] = []
        assert leakage.rlen(events, "k", mu=100) == 0
        assert leakage.ulen(events, "k", mu=100) == 0
        assert leakage.ilen(events, "k", mu=100) == 0
        assert leakage.dlen(events, "k", mu=100) == 0
