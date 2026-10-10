"""Forward/backward-private SSE schemes, leakage profiles, and minimal implementations.

Includes:
    - make_token: Simulated token generator for T1 (prefix) and T2 (random).
    - Transcript: Recording search tokens, returned index entries, and file IDs.
    - BSigma: Basic Sigma scheme storing (label, AES-GCM(id || op)).
    - Fides: Forward-private scheme using an RSA trapdoor permutation token chain.
    - Mitra: HMAC-SHA256 based scheme where tokens share a keyword-derived component.
"""

import collections
from dataclasses import dataclass, field
import hashlib
import hmac
import os
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np

_DEFAULT_SECRET = b"dps_token_secret_seed"


# ---------------------------------------------------------------------------
# Simulated token generation (T1 / T2)
# ---------------------------------------------------------------------------

def make_token(
    keyword: str,
    mode: str,
    rng: np.random.Generator,
    secret: bytes = _DEFAULT_SECRET,
) -> str:
    """Generate an SSE search token for *keyword* under the specified mode.

    Modes:
        'T1': Prefix-revealing token. Returns a keyword-derived prefix
              (HMAC of the keyword with a secret) + '|' + a fresh random nonce.
              Enables deterministic query-equality inference via token prefix.
        'T2': Fully random opaque token. Returns a random hex string with no
              keyword-identifying prefix. Query equality must be inferred via
              response-set similarity (RSP).

    Args:
        keyword: The search keyword string.
        mode:    'T1' or 'T2' (case-insensitive).
        rng:     Seeded numpy Generator for fresh random nonces / tokens.
        secret:  Secret key for HMAC generation in 'T1' mode.

    Returns:
        Opaque token string.

    Raises:
        ValueError: If *mode* is not 'T1' or 'T2'.
    """
    mode_upper = mode.strip().upper()
    if mode_upper == "T1":
        prefix = hmac.new(secret, keyword.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        nonce = rng.bytes(8).hex()
        return f"{prefix}|{nonce}"
    elif mode_upper == "T2":
        return rng.bytes(16).hex()
    else:
        raise ValueError(f"Unknown token mode {mode!r}. Expected 'T1' or 'T2'.")


# ---------------------------------------------------------------------------
# Search Transcript
# ---------------------------------------------------------------------------

@dataclass
class TranscriptEntry:
    """Represents the observed transcript from a single search query."""

    token: Any
    returned_index_entries: list[Any]
    returned_file_ids: list[int]


class Transcript:
    """Records server-observed tokens, index entries, and requested file IDs."""

    def __init__(self) -> None:
        self.entries: list[TranscriptEntry] = []
        self.tokens: list[Any] = []
        self.returned_index_entries: list[list[Any]] = []
        self.returned_file_ids: list[list[int]] = []

    def record(self, token: Any, index_entries: list[Any], file_ids: list[int]) -> None:
        """Record an executed search query."""
        entry = TranscriptEntry(
            token=token,
            returned_index_entries=list(index_entries),
            returned_file_ids=list(file_ids),
        )
        self.entries.append(entry)
        self.tokens.append(token)
        self.returned_index_entries.append(list(index_entries))
        self.returned_file_ids.append(list(file_ids))

    def compute_a_and_d(self, query_index: int = -1) -> tuple[int, int]:
        """Compute (a, d) from ulen and rlen recorded in the transcript.

        Identities:
            u = ulen = a + d (number of returned index entries)
            n = rlen = a - d (number of returned file IDs)
            a = (u + n) // 2 (number of additions)
            d = (u - n) // 2 (number of deletions)
        """
        if not self.entries:
            raise ValueError("Transcript is empty; no queries recorded.")
        u = len(self.returned_index_entries[query_index])
        n = len(self.returned_file_ids[query_index])
        a = (u + n) // 2
        d = (u - n) // 2
        return a, d

    def clear(self) -> None:
        """Reset the transcript."""
        self.entries.clear()
        self.tokens.clear()
        self.returned_index_entries.clear()
        self.returned_file_ids.clear()


def compute_a_and_d(transcript: Transcript, query_index: int = -1) -> tuple[int, int]:
    """Helper function to compute (a, d) from a Transcript."""
    return transcript.compute_a_and_d(query_index)


# ---------------------------------------------------------------------------
# Scheme (a): BSigma
# ---------------------------------------------------------------------------

class BSigma:
    """Minimal implementation of BSigma.

    - Server stores: (label, AES-GCM(id || op)) for each keyword update.
    - Search:
        1. Client sends label; server returns all matching index entries (reveals ulen = a + d).
        2. Client decrypts, removes deleted IDs, and requests the surviving file IDs (reveals rlen = a - d).
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        transcript: Optional[Transcript] = None,
    ) -> None:
        self.key: bytes = key or os.urandom(32)
        self.aes_key: bytes = hashlib.sha256(self.key + b":aes").digest()[:16]
        self.server_db: dict[str, list[bytes]] = collections.defaultdict(list)
        self.transcript: Transcript = transcript if transcript is not None else Transcript()

    def _get_label(self, keyword: str) -> str:
        return hmac.new(self.key, f"lbl:{keyword}".encode("utf-8"), hashlib.sha256).hexdigest()

    def add(self, keyword: str, doc_id: int) -> None:
        """Add document id under keyword."""
        self._update(keyword, doc_id, "add")

    def delete(self, keyword: str, doc_id: int) -> None:
        """Delete document id under keyword."""
        self._update(keyword, doc_id, "del")

    def _update(self, keyword: str, doc_id: int, op: str) -> None:
        label = self._get_label(keyword)
        plaintext = f"{doc_id}:{op}".encode("utf-8")
        nonce = os.urandom(12)
        aesgcm = AESGCM(self.aes_key)
        ciphertext = nonce + aesgcm.encrypt(nonce, plaintext, None)
        self.server_db[label].append(ciphertext)

    def search(self, keyword: str) -> list[int]:
        """Search for keyword, decrypt results, reconcile updates, and record transcript."""
        token = self._get_label(keyword)

        # 1. Server returns all index entries for this label (reveals ulen)
        returned_entries = list(self.server_db.get(token, []))

        # 2. Client decrypts and reconciles additions and deletions
        aesgcm = AESGCM(self.aes_key)
        live_doc_ids: set[int] = set()

        for ct in returned_entries:
            nonce = ct[:12]
            raw_ct = ct[12:]
            pt = aesgcm.decrypt(nonce, raw_ct, None).decode("utf-8")
            doc_id_str, op = pt.split(":")
            doc_id = int(doc_id_str)
            if op == "add":
                live_doc_ids.add(doc_id)
            elif op == "del":
                live_doc_ids.discard(doc_id)

        # 3. Client requests the surviving files (reveals rlen)
        result_files = sorted(live_doc_ids)

        # 4. Record interaction in transcript
        self.transcript.record(
            token=token,
            index_entries=returned_entries,
            file_ids=result_files,
        )

        return result_files


# ---------------------------------------------------------------------------
# Scheme (b): Fides (RSA Trapdoor Permutation Token Chain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidesToken:
    """Search token for Fides: current chain head and update counter."""

    token_val: int
    counter: int

    def __repr__(self) -> str:
        return f"FidesToken(val={hex(self.token_val)[:10]}..., c={self.counter})"


class Fides:
    """Forward-private SSE scheme based on an RSA trapdoor permutation.

    - Updates move forward in the chain using the private trapdoor exponent d:
          T_{c} = (T_{c-1})^d mod N
    - Server uses the public exponent e to trace backward from newer tokens to older tokens:
          T_{prev} = (T_{curr})^e mod N
    - Forward check verifies that an older token can be computed from a newer token.
    """

    def __init__(
        self,
        rsa_key_size: int = 1024,
        key: Optional[bytes] = None,
        transcript: Optional[Transcript] = None,
    ) -> None:
        self.key: bytes = key or os.urandom(32)
        self.transcript: Transcript = transcript if transcript is not None else Transcript()

        # Generate RSA keypair
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=rsa_key_size)
        nums = self._private_key.private_numbers()
        self.n: int = nums.public_numbers.n
        self.e: int = nums.public_numbers.e
        self._d: int = nums.d

        # Client state: keyword -> (counter c, current token T_c)
        self._keyword_state: dict[str, tuple[int, int]] = {}

        # Server storage: label -> (nonce, ciphertext)
        self.server_db: dict[str, bytes] = {}

    def _get_root_token(self, keyword: str) -> int:
        """Derive initial root token T_0 for a keyword."""
        h = hmac.new(self.key, f"root:{keyword}".encode("utf-8"), hashlib.sha256).digest()
        val = int.from_bytes(h, "big") % (self.n - 2) + 2
        return val

    def _derive_entry_key(self, token_val: int) -> bytes:
        return hashlib.sha256(token_val.to_bytes((self.n.bit_length() + 7) // 8, "big")).digest()[:16]

    def _derive_label(self, token_val: int) -> str:
        return hashlib.sha256(b"label:" + token_val.to_bytes((self.n.bit_length() + 7) // 8, "big")).hexdigest()

    def add(self, keyword: str, doc_id: int) -> None:
        """Add document id under keyword."""
        self._update(keyword, doc_id, "add")

    def delete(self, keyword: str, doc_id: int) -> None:
        """Delete document id under keyword."""
        self._update(keyword, doc_id, "del")

    def _update(self, keyword: str, doc_id: int, op: str) -> None:
        if keyword not in self._keyword_state:
            curr_c = 0
            curr_token = self._get_root_token(keyword)
        else:
            curr_c, curr_token = self._keyword_state[keyword]

        # Compute next token using private trapdoor exponent d
        next_c = curr_c + 1
        next_token = pow(curr_token, self._d, self.n)
        self._keyword_state[keyword] = (next_c, next_token)

        # Encrypt update entry
        label = self._derive_label(next_token)
        entry_key = self._derive_entry_key(next_token)
        aesgcm = AESGCM(entry_key)
        nonce = os.urandom(12)
        plaintext = f"{doc_id}:{op}".encode("utf-8")
        ciphertext = nonce + aesgcm.encrypt(nonce, plaintext, None)
        self.server_db[label] = ciphertext

    def search(self, keyword: str) -> list[int]:
        """Issue search token; server traces backward along the chain using public exponent e."""
        if keyword not in self._keyword_state:
            token = FidesToken(token_val=self._get_root_token(keyword), counter=0)
            self.transcript.record(token, [], [])
            return []

        c, current_t = self._keyword_state[keyword]
        token = FidesToken(token_val=current_t, counter=c)

        # Server steps backward using public exponent e
        curr = current_t
        returned_entries: list[bytes] = []
        recovered_tokens: list[int] = []

        for _ in range(c):
            lbl = self._derive_label(curr)
            if lbl in self.server_db:
                returned_entries.append(self.server_db[lbl])
                recovered_tokens.append(curr)
            # Step backward: T_{prev} = (T_{curr})^e mod N
            curr = pow(curr, self.e, self.n)

        # Client decrypts entries in chronological order (oldest to newest)
        live_doc_ids: set[int] = set()
        for tok_val, ct in zip(reversed(recovered_tokens), reversed(returned_entries)):
            entry_key = self._derive_entry_key(tok_val)
            aesgcm = AESGCM(entry_key)
            nonce = ct[:12]
            raw_ct = ct[12:]
            pt = aesgcm.decrypt(nonce, raw_ct, None).decode("utf-8")
            doc_id_str, op = pt.split(":")
            doc_id = int(doc_id_str)
            if op == "add":
                live_doc_ids.add(doc_id)
            elif op == "del":
                live_doc_ids.discard(doc_id)

        result_files = sorted(live_doc_ids)

        self.transcript.record(
            token=token,
            index_entries=returned_entries,
            file_ids=result_files,
        )

        return result_files

    @staticmethod
    def forward_check(
        newer_token: int | FidesToken,
        older_token: int | FidesToken,
        steps: int,
        n: Optional[int] = None,
        e: int = 65537,
    ) -> bool:
        """Verify that an older token can be derived from a newer token via public exponent e.

        Args:
            newer_token: Token value at a higher counter.
            older_token: Token value at an earlier counter.
            steps:       Counter difference (newer_counter - older_counter).
            n:           RSA modulus N (optional if called on an instance).
            e:           Public RSA exponent (default: 65537).

        Returns:
            True if older_token == newer_token^(e^steps) mod N.
        """
        t_new = newer_token.token_val if isinstance(newer_token, FidesToken) else newer_token
        t_old = older_token.token_val if isinstance(older_token, FidesToken) else older_token

        if steps < 0:
            return False
        curr = t_new
        for _ in range(steps):
            curr = pow(curr, e, n)
        return curr == t_old

    def check_token_link(
        self,
        newer_token: int | FidesToken,
        older_token: int | FidesToken,
        steps: int,
    ) -> bool:
        """Instance method for forward_check using the instance's modulus."""
        return self.forward_check(newer_token, older_token, steps, n=self.n, e=self.e)


# ---------------------------------------------------------------------------
# Scheme (c): Mitra (HMAC-SHA256 Based)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MitraToken:
    """Search token for Mitra: contains a shared keyword-derived component."""

    keyword_component: str
    counter: int

    def __repr__(self) -> str:
        return f"MitraToken(kw={self.keyword_component[:8]}..., c={self.counter})"


class Mitra:
    """Minimal implementation of Mitra.

    - Tokens share a keyword-derived HMAC-SHA256 component.
    - Each update increments a per-keyword counter c.
    - Search provides the keyword component and counter; server derives all labels up to c.
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        transcript: Optional[Transcript] = None,
    ) -> None:
        self.key: bytes = key or os.urandom(32)
        self.transcript: Transcript = transcript if transcript is not None else Transcript()
        self._keyword_counters: dict[str, int] = collections.defaultdict(int)
        self.server_db: dict[str, bytes] = {}

    def _get_keyword_component(self, keyword: str) -> bytes:
        """Derive the secret keyword-specific HMAC component."""
        return hmac.new(self.key, f"mitra:{keyword}".encode("utf-8"), hashlib.sha256).digest()

    def add(self, keyword: str, doc_id: int) -> None:
        """Add document id under keyword."""
        self._update(keyword, doc_id, "add")

    def delete(self, keyword: str, doc_id: int) -> None:
        """Delete document id under keyword."""
        self._update(keyword, doc_id, "del")

    def _update(self, keyword: str, doc_id: int, op: str) -> None:
        kw_comp = self._get_keyword_component(keyword)
        self._keyword_counters[keyword] += 1
        c = self._keyword_counters[keyword]

        # Label and entry key derived from keyword component and counter
        label = hmac.new(kw_comp, f"lbl:{c}".encode("utf-8"), hashlib.sha256).hexdigest()
        entry_key = hmac.new(kw_comp, f"enc:{c}".encode("utf-8"), hashlib.sha256).digest()[:16]

        aesgcm = AESGCM(entry_key)
        nonce = os.urandom(12)
        plaintext = f"{doc_id}:{op}".encode("utf-8")
        ciphertext = nonce + aesgcm.encrypt(nonce, plaintext, None)
        self.server_db[label] = ciphertext

    def search(self, keyword: str) -> list[int]:
        """Search for keyword; search token reveals the keyword component and current counter."""
        kw_comp = self._get_keyword_component(keyword)
        c = self._keyword_counters[keyword]
        token = MitraToken(keyword_component=kw_comp.hex(), counter=c)

        if c == 0:
            self.transcript.record(token, [], [])
            return []

        # Server generates all labels from 1 to c using the shared keyword component
        returned_entries: list[bytes] = []
        entry_keys: list[bytes] = []

        for i in range(1, c + 1):
            label = hmac.new(kw_comp, f"lbl:{i}".encode("utf-8"), hashlib.sha256).hexdigest()
            if label in self.server_db:
                returned_entries.append(self.server_db[label])
                e_key = hmac.new(kw_comp, f"enc:{i}".encode("utf-8"), hashlib.sha256).digest()[:16]
                entry_keys.append(e_key)

        # Client decrypts and reconciles live document IDs
        live_doc_ids: set[int] = set()
        for e_key, ct in zip(entry_keys, returned_entries):
            aesgcm = AESGCM(e_key)
            nonce = ct[:12]
            raw_ct = ct[12:]
            pt = aesgcm.decrypt(nonce, raw_ct, None).decode("utf-8")
            doc_id_str, op = pt.split(":")
            doc_id = int(doc_id_str)
            if op == "add":
                live_doc_ids.add(doc_id)
            elif op == "del":
                live_doc_ids.discard(doc_id)

        result_files = sorted(live_doc_ids)

        self.transcript.record(
            token=token,
            index_entries=returned_entries,
            file_ids=result_files,
        )

        return result_files
