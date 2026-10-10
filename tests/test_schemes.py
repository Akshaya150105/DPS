"""Tests for SSE schemes and token generation in schemes.py."""

import numpy as np
import pytest

from schemes import generate_table2, make_token, reproduce_fig2


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestMakeTokenT1:
    def test_format_has_single_pipe_separator(self) -> None:
        """T1 token must contain exactly one '|' separator: prefix|nonce."""
        token = make_token("kw0", "T1", make_rng())
        parts = token.split("|")
        assert len(parts) == 2, f"Token {token!r} does not have exactly two parts."
        assert len(parts[0]) > 0 and len(parts[1]) > 0

    def test_same_keyword_same_prefix(self) -> None:
        """Two tokens for the same keyword must share the same prefix."""
        rng = make_rng(123)
        tok1 = make_token("kw1", "T1", rng)
        tok2 = make_token("kw1", "T1", rng)
        prefix1, nonce1 = tok1.split("|")
        prefix2, nonce2 = tok2.split("|")
        assert prefix1 == prefix2
        assert nonce1 != nonce2, "Nonces must be fresh and distinct."

    def test_different_keywords_different_prefixes(self) -> None:
        """Different keywords must produce different prefixes."""
        rng = make_rng(42)
        tok1 = make_token("kw0", "T1", rng)
        tok2 = make_token("kw1", "T1", rng)
        assert tok1.split("|")[0] != tok2.split("|")[0]

    def test_case_insensitive_mode_t1(self) -> None:
        """Mode string 't1' (lowercase) should be accepted."""
        tok = make_token("kw0", "t1", make_rng(42))
        assert "|" in tok

    def test_reproducible_same_seed(self) -> None:
        """Same seed must produce identical tokens."""
        tok1 = make_token("kw0", "T1", make_rng(99))
        tok2 = make_token("kw0", "T1", make_rng(99))
        assert tok1 == tok2


class TestMakeTokenT2:
    def test_no_separator(self) -> None:
        """T2 tokens must be opaque random strings without '|'."""
        token = make_token("kw0", "T2", make_rng())
        assert "|" not in token
        assert len(token) > 0

    def test_same_keyword_different_tokens(self) -> None:
        """In T2 mode, multiple queries for the same keyword yield distinct tokens."""
        rng = make_rng(42)
        tok1 = make_token("kw0", "T2", rng)
        tok2 = make_token("kw0", "T2", rng)
        assert tok1 != tok2

    def test_case_insensitive_mode_t2(self) -> None:
        """Mode string 't2' (lowercase) should be accepted."""
        tok = make_token("kw0", "t2", make_rng(42))
        assert "|" not in tok

    def test_reproducible_same_seed(self) -> None:
        """Same seed must produce identical tokens in T2 mode."""
        tok1 = make_token("kw0", "T2", make_rng(55))
        tok2 = make_token("kw0", "T2", make_rng(55))
        assert tok1 == tok2


class TestMakeTokenInvalidMode:
    def test_invalid_mode_raises_value_error(self) -> None:
        """Unknown mode string must raise ValueError."""
        with pytest.raises(ValueError, match="Expected 'T1' or 'T2'"):
            make_token("kw0", "T3", make_rng())
        with pytest.raises(ValueError, match="Expected 'T1' or 'T2'"):
            make_token("kw0", "invalid", make_rng())


class TestPaperReproductions:
    def test_table2_contains_paper_leakage_exceptions(self, capsys) -> None:
        table = generate_table2()
        capsys.readouterr()

        assert "| Orion | RSP | yes | no | no | no | yes | Type I | irrelevant |" in table
        assert "| 2chFB | token | no | no | no | no | yes | Type II | partially same |" in table
        assert table.count("| Type III |") == 4

    def test_fig2_exposes_lengths_and_forward_volume(self, capsys) -> None:
        values = reproduce_fig2()
        output = capsys.readouterr().out

        assert values == {
            "ulen": 4,
            "rlen": 2,
            "insertions": 3,
            "deletions": 1,
            "fvol": {2: 2},
        }
        assert "fvol={2: 2}" in output


# ---------------------------------------------------------------------------
# Tests for Minimal Schemes (BSigma, Fides, Mitra) & Transcript
# ---------------------------------------------------------------------------

from schemes import BSigma, Fides, FidesToken, Mitra, Transcript, compute_a_and_d


class TestBSigma:
    def test_adds_and_deletes_correctness(self) -> None:
        """BSigma correctly returns live doc_ids after sequence of adds and deletes."""
        scheme = BSigma()
        scheme.add("crypto", 10)
        scheme.add("crypto", 20)
        scheme.add("crypto", 30)
        scheme.delete("crypto", 20)

        results = scheme.search("crypto")
        assert results == [10, 30]

        # Search unindexed keyword
        assert scheme.search("nonexistent") == []

        # Further additions and deletions
        scheme.add("crypto", 40)
        scheme.delete("crypto", 10)
        assert scheme.search("crypto") == [30, 40]

    def test_bsigma_transcript_computes_a_and_d(self) -> None:
        """From BSigma transcript, ulen and rlen yield accurate (a, d)."""
        scheme = BSigma()
        # 4 additions, 1 deletion -> a=4, d=1 -> ulen=5, rlen=3
        scheme.add("security", 1)
        scheme.add("security", 2)
        scheme.add("security", 3)
        scheme.add("security", 4)
        scheme.delete("security", 2)

        res = scheme.search("security")
        assert res == [1, 3, 4]

        # Verify recorded transcript entries
        assert len(scheme.transcript.entries) == 1
        entry = scheme.transcript.entries[0]
        assert len(entry.returned_index_entries) == 5  # ulen = a + d = 5
        assert len(entry.returned_file_ids) == 3       # rlen = a - d = 3

        a, d = scheme.transcript.compute_a_and_d()
        assert a == 4
        assert d == 1

        # Also test via standalone helper
        a2, d2 = compute_a_and_d(scheme.transcript)
        assert (a2, d2) == (4, 1)

        # Another search with only adds: a=2, d=0 -> ulen=2, rlen=2
        scheme.add("privacy", 100)
        scheme.add("privacy", 200)
        scheme.search("privacy")
        a_priv, d_priv = scheme.transcript.compute_a_and_d(query_index=-1)
        assert a_priv == 2
        assert d_priv == 0


class TestFides:
    def test_adds_and_deletes_correctness(self) -> None:
        """Fides correctly returns matching doc_ids across multiple updates."""
        scheme = Fides(rsa_key_size=1024)
        scheme.add("database", 5)
        scheme.add("database", 6)
        scheme.delete("database", 5)
        scheme.add("database", 7)

        res = scheme.search("database")
        assert res == [6, 7]

        # Different keyword is independent
        scheme.add("network", 99)
        assert scheme.search("network") == [99]
        assert scheme.search("database") == [6, 7]

    def test_fides_forward_check_links_old_and_new_tokens(self) -> None:
        """Public forward check links an older token from a newer token using exponent e."""
        scheme = Fides(rsa_key_size=1024)

        # Update 1
        scheme.add("audit", 1)
        tok1 = FidesToken(token_val=scheme._keyword_state["audit"][1], counter=1)

        # Update 2 & 3
        scheme.add("audit", 2)
        scheme.add("audit", 3)
        tok3 = FidesToken(token_val=scheme._keyword_state["audit"][1], counter=3)

        # Forward check: tok1 (counter 1) can be computed from tok3 (counter 3) across 2 steps
        assert scheme.check_token_link(tok3, tok1, steps=2) is True
        assert Fides.forward_check(tok3, tok1, steps=2, n=scheme.n, e=scheme.e) is True

        # Incorrect step count must fail
        assert scheme.check_token_link(tok3, tok1, steps=1) is False
        assert scheme.check_token_link(tok3, tok1, steps=3) is False

        # Independent keyword tokens must not link
        scheme.add("finance", 1)
        tok_finance = FidesToken(token_val=scheme._keyword_state["finance"][1], counter=1)
        assert scheme.check_token_link(tok3, tok_finance, steps=2) is False

    def test_fides_transcript_recorded(self) -> None:
        """Fides records search token, returned index entries, and returned file IDs."""
        scheme = Fides(rsa_key_size=1024)
        scheme.add("query", 11)
        scheme.add("query", 22)
        res = scheme.search("query")
        assert res == [11, 22]

        assert len(scheme.transcript.entries) == 1
        entry = scheme.transcript.entries[0]
        assert isinstance(entry.token, FidesToken)
        assert entry.token.counter == 2
        assert len(entry.returned_index_entries) == 2
        assert entry.returned_file_ids == [11, 22]


class TestMitra:
    def test_adds_and_deletes_correctness(self) -> None:
        """Mitra returns correct live doc_ids after sequence of additions and deletions."""
        scheme = Mitra()
        scheme.add("sse", 1)
        scheme.add("sse", 2)
        scheme.add("sse", 3)
        scheme.delete("sse", 2)
        scheme.add("sse", 4)

        res = scheme.search("sse")
        assert res == [1, 3, 4]

        scheme.delete("sse", 1)
        res2 = scheme.search("sse")
        assert res2 == [3, 4]

    def test_mitra_tokens_share_keyword_derived_component(self) -> None:
        """Tokens for the same keyword share a persistent keyword-derived component."""
        scheme = Mitra()
        scheme.add("protocol", 1)
        scheme.search("protocol")
        tok1 = scheme.transcript.tokens[-1]

        scheme.add("protocol", 2)
        scheme.search("protocol")
        tok2 = scheme.transcript.tokens[-1]

        # Both tokens must share the same keyword component
        assert tok1.keyword_component == tok2.keyword_component
        # But their counters differ as new updates were added
        assert tok1.counter == 1
        assert tok2.counter == 2

        # A different keyword must have a different keyword component
        scheme.add("cipher", 10)
        scheme.search("cipher")
        tok_diff = scheme.transcript.tokens[-1]
        assert tok_diff.keyword_component != tok1.keyword_component
