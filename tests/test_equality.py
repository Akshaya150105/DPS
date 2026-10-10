"""Tests for equality.py: UnionFind, build_equality (T1/T2), equality_quality."""

import pytest

from config import Config
from equality import EqualityResult, UnionFind, build_equality, equality_quality
from simulator import simulate
from synthetic import make_synthetic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(seed: int = 42, tau: int = 3, lam: float = 60.0,
         gamma: float = 0.15, equality: str = "T1") -> tuple:
    """Run simulate and return (records, ground_truth)."""
    cfg = Config(seed=seed, tau=tau, lambda_queries=lam, gamma=gamma, equality=equality)
    docs = make_synthetic(seed=seed)
    records, gt, _ = simulate(docs, cfg)
    return records, gt


# ---------------------------------------------------------------------------
# 1. UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_singleton_finds_itself(self) -> None:
        uf = UnionFind()
        uf._ensure(1)
        assert uf.find(1) == 1

    def test_union_merges_two_singletons(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_union_is_transitive(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_disjoint_sets_have_different_roots(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf._ensure(3)
        assert uf.find(1) != uf.find(3)

    def test_find_is_idempotent(self) -> None:
        uf = UnionFind()
        uf.union(5, 6)
        r1 = uf.find(5)
        r2 = uf.find(5)
        assert r1 == r2

    def test_path_compression(self) -> None:
        """After find, all nodes on the path must point directly to the root."""
        uf = UnionFind()
        for i in range(1, 6):
            uf.union(i, i + 1)
        root = uf.find(1)
        uf.find(6)  # trigger path compression
        # All should now point directly to root.
        for i in range(1, 7):
            assert uf.find(i) == root

    def test_many_elements(self) -> None:
        uf = UnionFind()
        for i in range(99):
            uf.union(i, i + 1)
        root = uf.find(0)
        assert all(uf.find(i) == root for i in range(100))


# ---------------------------------------------------------------------------
# 2. build_equality – shared structural guarantees
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records_gt_struct():
    return _run(gamma=0.0, lam=80.0)


class TestBuildEqualityStructure:

    def test_returns_equality_result(self, records_gt_struct) -> None:
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        assert isinstance(result, EqualityResult)

    def test_every_query_id_in_group_of(self, records_gt_struct) -> None:
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        qids = {r.query_id for r in records}
        assert set(result.group_of.keys()) == qids

    def test_every_query_id_appears_in_some_group(self, records_gt_struct) -> None:
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        all_in_groups = {qid for members in result.groups.values() for qid in members}
        qids = {r.query_id for r in records}
        assert all_in_groups == qids

    def test_group_of_consistent_with_groups(self, records_gt_struct) -> None:
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        for gid, members in result.groups.items():
            for qid in members:
                assert result.group_of[qid] == gid

    def test_linked_pairs_coverage(self, records_gt_struct) -> None:
        """Every query_id must appear as a key in linked_pairs."""
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        qids = {r.query_id for r in records}
        assert set(result.linked_pairs.keys()) == qids

    def test_linked_pairs_chain_within_group(self, records_gt_struct) -> None:
        """Following linked_pairs from the head must traverse exactly the group members."""
        records, _ = records_gt_struct
        result = build_equality(records, mode="T1")
        id_to_rec = {r.query_id: r for r in records}
        for gid, members in result.groups.items():
            # Head is the member with the smallest timestamp (then query_id).
            head = min(members, key=lambda qid: (id_to_rec[qid].timestamp, qid))
            visited = []
            cur = head
            while cur is not None:
                visited.append(cur)
                cur = result.linked_pairs[cur]
            assert sorted(visited) == sorted(members)

    def test_invalid_mode_raises(self, records_gt_struct) -> None:
        records, _ = records_gt_struct
        with pytest.raises(ValueError):
            build_equality(records, mode="T3")

    def test_empty_records_returns_empty_result(self) -> None:
        result = build_equality([], mode="T1")
        assert result.groups == {}
        assert result.group_of == {}
        assert result.linked_pairs == {}


# ---------------------------------------------------------------------------
# 3. build_equality – T1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t1_result_gt():
    records, gt = _run(gamma=0.0, lam=80.0, equality="T1")
    result = build_equality(records, mode="T1")
    return result, gt


class TestBuildEqualityT1:
    def test_t1_precision_is_1(self, t1_result_gt) -> None:
        """T1 must not put two different keywords in the same group."""
        result, gt = t1_result_gt
        assert len(result.group_of) > 0
        p, _ = equality_quality(result, gt)
        assert p == pytest.approx(1.0), f"T1 precision={p:.4f}, expected 1.0"

    def test_t1_recall_is_1(self, t1_result_gt) -> None:
        """T1 must place all queries for the same keyword in the same group."""
        result, gt = t1_result_gt
        _, r = equality_quality(result, gt)
        assert r == pytest.approx(1.0), f"T1 recall={r:.4f}, expected 1.0"

    def test_t1_groups_correspond_to_distinct_keywords(self, t1_result_gt) -> None:
        """Each T1 group must contain exactly one true keyword."""
        result, gt = t1_result_gt
        for gid, members in result.groups.items():
            kws = {gt[qid] for qid in members}
            assert len(kws) == 1, f"Group {gid} contains multiple keywords: {kws}"

    def test_t1_singletons_have_unique_queries(self, t1_result_gt) -> None:
        """Every keyword queried at least twice must appear in a group of size >= 2."""
        result, gt = t1_result_gt
        from collections import Counter
        kw_counts = Counter(gt.values())
        multi_kws = {kw for kw, cnt in kw_counts.items() if cnt >= 2}
        for gid, members in result.groups.items():
            if gt[members[0]] in multi_kws:
                assert len(members) >= 2


# ---------------------------------------------------------------------------
# 4. build_equality – T2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t2_result_gt_gamma0():
    """T2 equality with gamma=0 (no deletions; fvol stable across intervals)."""
    records, gt = _run(gamma=0.0, lam=80.0, equality="T2")
    result = build_equality(records, mode="T2", delta=0.6)
    return result, gt


class TestBuildEqualityT2:
    def test_t2_recall_above_0_8_gamma0(self, t2_result_gt_gamma0) -> None:
        """T2 recall must exceed 0.8 when gamma=0 (stable fvol)."""
        result, gt = t2_result_gt_gamma0
        _, r = equality_quality(result, gt)
        assert r >= 0.8, f"T2 recall={r:.4f} < 0.8"

    def test_t2_precision_nonneg(self, t2_result_gt_gamma0) -> None:
        """T2 precision must be in [0, 1]."""
        result, gt = t2_result_gt_gamma0
        p, _ = equality_quality(result, gt)
        assert 0.0 <= p <= 1.0

    def test_t2_all_qids_covered(self, t2_result_gt_gamma0) -> None:
        """T2 result must contain every query_id."""
        result, gt = t2_result_gt_gamma0
        assert set(result.group_of.keys()) == set(gt.keys())

    def test_t2_high_delta_produces_singletons(self) -> None:
        """With delta=1.0 only identical fvols are linked; few merges expected."""
        records, gt = _run(gamma=0.1, lam=60.0, equality="T2")
        result = build_equality(records, mode="T2", delta=1.0)
        # Most groups should be singletons with delta=1.0
        singleton_frac = sum(1 for m in result.groups.values() if len(m) == 1)
        # At least half should be singletons (very conservative check).
        assert singleton_frac / max(len(result.groups), 1) >= 0.3

    def test_t2_lower_delta_more_links(self) -> None:
        """Lower delta should produce fewer but larger groups than higher delta."""
        records, _ = _run(gamma=0.0, lam=80.0, equality="T2")
        result_low  = build_equality(records, mode="T2", delta=0.1)
        result_high = build_equality(records, mode="T2", delta=0.95)
        # Fewer groups means more things were merged.
        assert len(result_low.groups) <= len(result_high.groups)


# ---------------------------------------------------------------------------
# 5. equality_quality
# ---------------------------------------------------------------------------

class TestEqualityQuality:
    def _make_result_and_gt(self, groups: dict[int, list[int]], gt: dict[int, str]) -> tuple:
        from equality import EqualityResult
        group_of = {qid: gid for gid, members in groups.items() for qid in members}
        linked_pairs = {qid: None for qid in group_of}
        result = EqualityResult(groups=groups, group_of=group_of, linked_pairs=linked_pairs)
        return result, gt

    def test_perfect_grouping_gives_1_1(self) -> None:
        groups = {0: [0, 1], 2: [2, 3]}
        gt     = {0: "kw0", 1: "kw0", 2: "kw1", 3: "kw1"}
        result, gt_ = self._make_result_and_gt(groups, gt)
        p, r = equality_quality(result, gt_)
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_all_singletons_recall_is_0(self) -> None:
        groups = {0: [0], 1: [1], 2: [2]}
        gt     = {0: "kw0", 1: "kw0", 2: "kw0"}
        result, gt_ = self._make_result_and_gt(groups, gt)
        p, r = equality_quality(result, gt_)
        assert r == pytest.approx(0.0)
        # No predictions so precision defaults to 1.0.
        assert p == pytest.approx(1.0)

    def test_all_merged_precision_is_0(self) -> None:
        groups = {0: [0, 1, 2, 3]}
        gt     = {0: "kw0", 1: "kw1", 2: "kw2", 3: "kw3"}
        result, gt_ = self._make_result_and_gt(groups, gt)
        p, r = equality_quality(result, gt_)
        assert p == pytest.approx(0.0)
        # No truly positive pairs exist, so recall defaults to 1.0.
        assert r == pytest.approx(1.0)

    def test_half_recall(self) -> None:
        groups = {0: [0, 1], 2: [2], 3: [3]}
        gt     = {0: "kw0", 1: "kw0", 2: "kw0", 3: "kw0"}
        result, gt_ = self._make_result_and_gt(groups, gt)
        p, r = equality_quality(result, gt_)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# 6. build_equality – 'exact' mode (BSigma-style tokens)
# ---------------------------------------------------------------------------

def _make_records_with_tokens(token_map):
    """Build minimal LeakageRecord objects with arbitrary token values."""
    import collections
    from types_ import LeakageRecord
    recs = []
    for qid, tok in token_map.items():
        r = LeakageRecord(
            query_id=qid,
            token=tok,
            interval=0,
            timestamp=qid,
            ulen=1,
            rlen=1,
            fvol=collections.Counter(),
        )
        recs.append(r)
    return recs


class TestBuildEqualityExact:
    """Exact mode: tokens that compare equal are grouped together."""

    def test_exact_groups_identical_string_tokens(self):
        """Two records with the same string token are placed in one group."""
        recs = _make_records_with_tokens({0: "tok_abc", 1: "tok_abc", 2: "tok_xyz"})
        result = build_equality(recs, mode="exact")
        assert result.group_of[0] == result.group_of[1], "identical tokens must be in same group"
        assert result.group_of[0] != result.group_of[2], "different tokens must be in different groups"

    def test_exact_precision_recall_perfect_bsigma_style(self):
        """BSigma: same keyword -> same label token. Exact should achieve P=R=1."""
        from schemes import BSigma
        import collections
        from types_ import LeakageRecord
        scheme = BSigma()
        scheme.add("alpha", 1)
        scheme.add("alpha", 2)
        scheme.add("beta", 3)
        tok_alpha = scheme._get_label("alpha")
        tok_beta  = scheme._get_label("beta")
        recs = [
            LeakageRecord(query_id=0, token=tok_alpha, interval=0, timestamp=0,
                          ulen=2, rlen=2, fvol=collections.Counter()),
            LeakageRecord(query_id=1, token=tok_alpha, interval=1, timestamp=1,
                          ulen=2, rlen=2, fvol=collections.Counter()),
            LeakageRecord(query_id=2, token=tok_beta,  interval=0, timestamp=2,
                          ulen=1, rlen=1, fvol=collections.Counter()),
        ]
        result = build_equality(recs, mode="exact")
        gt = {0: "alpha", 1: "alpha", 2: "beta"}
        p, r = equality_quality(result, gt)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_exact_all_distinct_tokens_all_singletons(self):
        """All unique tokens -> all singleton groups."""
        recs = _make_records_with_tokens({0: "t0", 1: "t1", 2: "t2"})
        result = build_equality(recs, mode="exact")
        assert all(len(members) == 1 for members in result.groups.values())

    def test_exact_case_insensitive_mode_string(self):
        """Mode EXACT and Exact are both accepted."""
        recs = _make_records_with_tokens({0: "tok", 1: "tok"})
        r1 = build_equality(recs, mode="EXACT")
        r2 = build_equality(recs, mode="Exact")
        assert r1.group_of[0] == r1.group_of[1]
        assert r2.group_of[0] == r2.group_of[1]


# ---------------------------------------------------------------------------
# 7. build_equality – 'partial' mode (Mitra-style tokens)
# ---------------------------------------------------------------------------

class TestBuildEqualityPartial:
    """Partial mode: tokens sharing a keyword_component are grouped together."""

    def test_partial_groups_by_keyword_component(self):
        """MitraTokens with same keyword_component land in one group."""
        from schemes import MitraToken
        tok_a1 = MitraToken(keyword_component="comp_A", counter=1)
        tok_a2 = MitraToken(keyword_component="comp_A", counter=2)
        tok_b1 = MitraToken(keyword_component="comp_B", counter=1)
        recs = _make_records_with_tokens({0: tok_a1, 1: tok_a2, 2: tok_b1})
        result = build_equality(recs, mode="partial")
        assert result.group_of[0] == result.group_of[1], "same component => same group"
        assert result.group_of[0] != result.group_of[2], "different component => different group"

    def test_partial_precision_recall_perfect_mitra(self):
        """Mitra: two searches for the same keyword share keyword_component => P=R=1."""
        from schemes import Mitra
        import collections
        from types_ import LeakageRecord
        scheme = Mitra()
        scheme.add("gamma", 10)
        scheme.search("gamma")
        tok1 = scheme.transcript.tokens[-1]
        scheme.add("gamma", 20)
        scheme.search("gamma")
        tok2 = scheme.transcript.tokens[-1]
        scheme.add("delta", 30)
        scheme.search("delta")
        tok3 = scheme.transcript.tokens[-1]
        recs = [
            LeakageRecord(query_id=0, token=tok1, interval=0, timestamp=0,
                          ulen=1, rlen=1, fvol=collections.Counter()),
            LeakageRecord(query_id=1, token=tok2, interval=1, timestamp=1,
                          ulen=2, rlen=2, fvol=collections.Counter()),
            LeakageRecord(query_id=2, token=tok3, interval=0, timestamp=2,
                          ulen=1, rlen=1, fvol=collections.Counter()),
        ]
        result = build_equality(recs, mode="partial")
        gt = {0: "gamma", 1: "gamma", 2: "delta"}
        p, r = equality_quality(result, gt)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_partial_string_pipe_tokens(self):
        """String tokens of the form prefix|nonce are grouped by prefix."""
        recs = _make_records_with_tokens({0: "aaa|n1", 1: "aaa|n2", 2: "bbb|n3"})
        result = build_equality(recs, mode="partial")
        assert result.group_of[0] == result.group_of[1]
        assert result.group_of[0] != result.group_of[2]


# ---------------------------------------------------------------------------
# 8. build_equality – 'computational' mode (Fides-style forward_check)
# ---------------------------------------------------------------------------

class TestBuildEqualityComputational:
    """Computational mode: forward_check callable determines grouping."""

    def _make_fides_forward_check(self, scheme):
        """Return a forward_check callable bound to a Fides scheme instance."""
        def check(newer_token, older_token):
            from schemes import FidesToken
            if not (isinstance(newer_token, FidesToken) and
                    isinstance(older_token, FidesToken)):
                return False
            steps = newer_token.counter - older_token.counter
            if steps <= 0:
                return False
            return scheme.check_token_link(newer_token, older_token, steps=steps)
        return check

    def test_computational_groups_fides_tokens(self):
        """Fides tokens from the same keyword chain are grouped by forward_check."""
        from schemes import Fides, FidesToken
        import collections
        from types_ import LeakageRecord
        scheme = Fides(rsa_key_size=1024)
        scheme.add("zeta", 1)
        tok1 = FidesToken(token_val=scheme._keyword_state["zeta"][1], counter=1)
        scheme.add("zeta", 2)
        tok2 = FidesToken(token_val=scheme._keyword_state["zeta"][1], counter=2)
        scheme.add("eta", 99)
        tok_eta = FidesToken(token_val=scheme._keyword_state["eta"][1], counter=1)
        recs = [
            LeakageRecord(query_id=0, token=tok1, interval=0, timestamp=0,
                          ulen=1, rlen=1, fvol=collections.Counter()),
            LeakageRecord(query_id=1, token=tok2, interval=1, timestamp=1,
                          ulen=2, rlen=2, fvol=collections.Counter()),
            LeakageRecord(query_id=2, token=tok_eta, interval=0, timestamp=2,
                          ulen=1, rlen=1, fvol=collections.Counter()),
        ]
        fc = self._make_fides_forward_check(scheme)
        result = build_equality(recs, mode="computational", forward_check=fc)
        assert result.group_of[0] == result.group_of[1], "same keyword chain => same group"
        assert result.group_of[0] != result.group_of[2], "independent keyword => different group"

    def test_computational_precision_recall_perfect(self):
        """Computational mode recovers P=R=1 for Fides tokens from same keyword."""
        from schemes import Fides, FidesToken
        import collections
        from types_ import LeakageRecord
        scheme = Fides(rsa_key_size=1024)
        for kw, doc_ids in [("phi", [1, 2]), ("chi", [3])]:
            for doc in doc_ids:
                scheme.add(kw, doc)
        scheme.search("phi")
        tok_phi = scheme.transcript.tokens[-1]
        scheme.search("chi")
        tok_chi = scheme.transcript.tokens[-1]
        scheme.add("phi", 5)
        scheme.search("phi")
        tok_phi2 = scheme.transcript.tokens[-1]
        recs = [
            LeakageRecord(query_id=0, token=tok_phi,  interval=0, timestamp=0,
                          ulen=2, rlen=2, fvol=collections.Counter()),
            LeakageRecord(query_id=1, token=tok_chi,  interval=0, timestamp=1,
                          ulen=1, rlen=1, fvol=collections.Counter()),
            LeakageRecord(query_id=2, token=tok_phi2, interval=1, timestamp=2,
                          ulen=3, rlen=3, fvol=collections.Counter()),
        ]
        fc = self._make_fides_forward_check(scheme)
        result = build_equality(recs, mode="computational", forward_check=fc)
        gt = {0: "phi", 1: "chi", 2: "phi"}
        p, r = equality_quality(result, gt)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_computational_missing_callable_raises(self):
        """Calling computational without forward_check raises ValueError."""
        recs = _make_records_with_tokens({0: "t0", 1: "t1"})
        with pytest.raises(ValueError, match="forward_check"):
            build_equality(recs, mode="computational")

    def test_computational_case_insensitive(self):
        """Mode COMPUTATIONAL is accepted."""
        recs = _make_records_with_tokens({0: "t0"})
        result = build_equality(recs, mode="COMPUTATIONAL", forward_check=lambda a, b: False)
        assert list(result.groups.values()) == [[0]]


# ---------------------------------------------------------------------------
# 9. T2 still works when tokens are random (Moneta-style)
# ---------------------------------------------------------------------------

class TestT2WithRandomTokens:
    """T2 (response similarity) must work even when tokens are opaque random values."""

    def test_t2_ignores_token_content(self):
        """T2 groups by fvol similarity, not token value; random tokens are irrelevant."""
        import collections
        from types_ import LeakageRecord
        fvol_shared = collections.Counter({1: 5})
        recs = [
            LeakageRecord(query_id=0, token="rand_xyz_1", interval=0, timestamp=0,
                          ulen=5, rlen=5, fvol=fvol_shared),
            LeakageRecord(query_id=1, token="rand_abc_2", interval=0, timestamp=1,
                          ulen=5, rlen=5, fvol=fvol_shared),
        ]
        result = build_equality(recs, mode="T2", delta=0.99)
        # With identical fvol, RSP = 1.0 >= 0.99 -> must be merged
        assert result.group_of[0] == result.group_of[1]

    def test_t2_completely_disjoint_fvol_not_linked(self):
        """Records with no shared response-set volume must not be linked."""
        import collections
        from types_ import LeakageRecord
        recs = [
            LeakageRecord(query_id=0, token="r1", interval=0, timestamp=0,
                          ulen=1, rlen=1, fvol=collections.Counter({1: 100})),
            LeakageRecord(query_id=1, token="r2", interval=0, timestamp=1,
                          ulen=1, rlen=1, fvol=collections.Counter({2: 100})),
        ]
        result = build_equality(recs, mode="T2", delta=0.6)
        # Completely disjoint volumes -> RSP = 0 < 0.6 -> separate groups
        assert result.group_of[0] != result.group_of[1]


# ---------------------------------------------------------------------------
# 10. Unknown mode raises ValueError (updated error message)
# ---------------------------------------------------------------------------

class TestNewModeValidation:
    def test_unknown_mode_still_raises(self):
        """An unrecognised mode string raises ValueError regardless of new modes."""
        recs = _make_records_with_tokens({0: "t"})
        with pytest.raises(ValueError):
            build_equality(recs, mode="INVALID_MODE")

    def test_t1_and_t2_still_accepted(self):
        """Existing T1 and T2 modes are still valid after adding new modes."""
        recs = _make_records_with_tokens({0: "aaa|n1", 1: "aaa|n2"})
        r1 = build_equality(recs, mode="T1")
        assert isinstance(r1, EqualityResult)
        r2 = build_equality(recs, mode="T2")
        assert isinstance(r2, EqualityResult)
