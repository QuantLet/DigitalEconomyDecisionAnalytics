"""
Additive secret sharing primitives for a 3-party semi-honest MPC protocol.

Each secret x is split into three shares (s0, s1, s2) over a finite field
Z_p such that s0 + s1 + s2 ≡ x (mod p).  Any single share is uniformly
random and reveals nothing about x; reconstruction requires all three.

Arithmetic on shares is linear: add/subtract locally, multiply via a
Beaver triple.  This stub uses a trusted dealer for Beaver triples to keep
the focus on the genomic ML application rather than the cryptographic setup.
"""

import numpy as np

PRIME = (1 << 61) - 1  # Mersenne prime for the finite field


def _mod(x: np.ndarray) -> np.ndarray:
    return x % PRIME


def share(secret: np.ndarray, n_parties: int = 3) -> list[np.ndarray]:
    """Split secret into n_parties additive shares."""
    secret = _mod(np.asarray(secret, dtype=np.int64))
    shares = [_mod(np.random.randint(0, PRIME, secret.shape, dtype=np.int64))
              for _ in range(n_parties - 1)]
    last = _mod(secret - sum(shares))
    return shares + [last]


def reconstruct(shares: list[np.ndarray]) -> np.ndarray:
    """Reconstruct the secret from all shares."""
    return _mod(sum(shares))


def secure_add(shares_a: list[np.ndarray],
               shares_b: list[np.ndarray]) -> list[np.ndarray]:
    """Element-wise addition in secret-shared form (local, no communication)."""
    return [_mod(a + b) for a, b in zip(shares_a, shares_b)]


def secure_scale(shares: list[np.ndarray], scalar: int) -> list[np.ndarray]:
    """Multiply secret-shared value by a public scalar (local)."""
    return [_mod(s * scalar) for s in shares]


# ---------------------------------------------------------------------------
# Beaver-triple based multiplication (simplified: trusted dealer)
# ---------------------------------------------------------------------------

def _beaver_triple(shape) -> tuple:
    """Generate a Beaver triple (a, b, c=a*b) as shares."""
    a = np.random.randint(0, PRIME >> 8, shape, dtype=np.int64)
    b = np.random.randint(0, PRIME >> 8, shape, dtype=np.int64)
    c = _mod(a * b)
    return share(a), share(b), share(c)


def secure_mul(shares_x: list[np.ndarray],
               shares_y: list[np.ndarray]) -> list[np.ndarray]:
    """
    Element-wise multiplication via Beaver triples.
    In a real deployment each party communicates masked values;
    here we simulate the full protocol with a trusted dealer.
    """
    shape = shares_x[0].shape
    sa, sb, sc = _beaver_triple(shape)

    # Each party computes d = x - a, e = y - b then broadcasts
    d_shares = [_mod(shares_x[i] - sa[i]) for i in range(3)]
    e_shares = [_mod(shares_y[i] - sb[i]) for i in range(3)]

    # Reconstruct d and e (simulates the broadcast step)
    d = reconstruct(d_shares)
    e = reconstruct(e_shares)

    # Each party updates its share of c to get xy = de + d*b + e*a + c
    result = []
    for i in range(3):
        zi = sc[i]
        zi = _mod(zi + d * sb[i])
        zi = _mod(zi + e * sa[i])
        if i == 0:
            zi = _mod(zi + d * e)
        result.append(zi)
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.integers(0, 1000, size=(4,), dtype=np.int64)
    y = rng.integers(0, 1000, size=(4,), dtype=np.int64)

    sx, sy = share(x), share(y)

    print("x         :", x)
    print("y         :", y)
    print("x + y     :", (x + y))
    print("MPC add   :", reconstruct(secure_add(sx, sy)))

    print("x * y     :", _mod(x * y))
    print("MPC mul   :", reconstruct(secure_mul(sx, sy)))
