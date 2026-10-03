"""Split a secret key so that any k of n people can rebuild it, and fewer cannot.

This is Asmuth-Bloom threshold secret sharing, which is built on the Chinese
Remainder Theorem: each person holds the remainder of one big number modulo
their own prime. It is the job CRT is actually suited to in this project.

Asmuth-Bloom hides the secret statistically rather than perfectly; the gap
shrinks with MARGIN_BITS. Shamir's scheme is the usual production choice.
"""

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import count
from math import prod

SCHEME = "asmuth-bloom"
# How much larger each person's prime is than the secret, in bits.
MARGIN_BITS = 64

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


@dataclass
class Share:
    scheme: str
    k: int
    n: int
    index: int
    secret_bytes: int
    modulus: str       # hex, public
    residue: str       # hex, the secret part of the share
    fingerprint: str   # first bytes of SHA-256(secret), to detect a wrong rebuild

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))


def _is_prime(n):
    """Miller-Rabin with fixed bases; ample for the non-adversarial numbers used here."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL_PRIMES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(n):
    return next(c for c in count(n + 1) if _is_prime(c))


@lru_cache(maxsize=None)
def _parameters(secret_bytes, k, n):
    """Public numbers: m0 bounds the secret, moduli are n primes just above it.

    They depend only on the sizes, not on the secret, so they are found once.
    """
    secret_bits = 8 * secret_bytes
    m0 = _next_prime(1 << secret_bits)
    moduli = []
    candidate = 1 << (secret_bits + MARGIN_BITS)
    for _ in range(n):
        candidate = _next_prime(candidate)
        moduli.append(candidate)
    # The Asmuth-Bloom condition: k shares pin the number down, k-1 leave it open.
    if m0 * prod(moduli[n - k + 1:]) >= prod(moduli[:k]):
        raise ValueError("moduli do not satisfy the Asmuth-Bloom condition")
    return m0, moduli


def _fingerprint(secret):
    return hashlib.sha256(secret).hexdigest()[:16]


def split(secret, k, n):
    """Return n shares of `secret` (bytes); any k of them rebuild it."""
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    m0, moduli = _parameters(len(secret), k, n)
    value = int.from_bytes(secret, "big")
    # Blind the secret with a random multiple of m0, staying below the k smallest moduli.
    blinded = value + secrets.randbelow(prod(moduli[:k]) // m0) * m0
    return [
        Share(SCHEME, k, n, i + 1, len(secret), format(m, "x"), format(blinded % m, "x"), _fingerprint(secret))
        for i, m in enumerate(moduli)
    ]


def _crt(residues, moduli):
    big_m = prod(moduli)
    total = 0
    for residue, m in zip(residues, moduli):
        partial = big_m // m
        total += residue * partial * pow(partial, -1, m)
    return total % big_m


def reconstruct_unchecked(shares):
    """CRT over whatever shares are given, with no checks. Used to show k-1 shares fail."""
    first = shares[0]
    m0, _ = _parameters(first.secret_bytes, first.k, first.n)
    blinded = _crt([int(s.residue, 16) for s in shares], [int(s.modulus, 16) for s in shares])
    return (blinded % m0).to_bytes(first.secret_bytes + 1, "big")[-first.secret_bytes:]


def combine(shares):
    """Rebuild the secret from at least k distinct shares."""
    if not shares:
        raise ValueError("no shares given")
    first = shares[0]
    distinct = {s.index: s for s in shares}
    if any((s.scheme, s.k, s.n, s.secret_bytes, s.fingerprint) != (SCHEME, first.k, first.n, first.secret_bytes, first.fingerprint) for s in shares):
        raise ValueError("these shares do not belong to the same secret")
    if len(distinct) < first.k:
        raise ValueError(f"need {first.k} different shares, got {len(distinct)}")
    secret = reconstruct_unchecked(list(distinct.values())[:first.k])
    if _fingerprint(secret) != first.fingerprint:
        raise ValueError("shares are damaged: the rebuilt key does not match")
    return secret
