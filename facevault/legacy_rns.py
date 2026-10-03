"""The 2024 project's "RNS encryption", rebuilt from the report so it can be measured.

Every pixel value x is replaced by its remainders modulo a few small numbers
(3, 5 and 17 in the report), giving one "encrypted" image per modulus.
The Chinese Remainder Theorem (CRT) turns the remainders back into x.

This module is kept as a baseline. It is NOT secure and must not be used to
protect anything; see attacks.py for why.
"""

from math import gcd, prod

import numpy as np

# The moduli quoted in the report. 3 * 5 * 17 = 255, so only 0..254 fit.
LEGACY_MODULI = (3, 5, 17)

# A set that does cover 0..255 (7 * 8 * 9 = 504). It fixes the lost pixels,
# but there is still no key, so it is still not encryption.
RANGE_FIXED_MODULI = (7, 8, 9)


def pairwise_coprime(moduli):
    return all(gcd(a, b) == 1 for i, a in enumerate(moduli) for b in moduli[i + 1:])


def encode(image, moduli=LEGACY_MODULI):
    """Split an 8-bit image into one residue image per modulus."""
    return [(image % m).astype(np.uint8) for m in moduli]


def crt(residues, moduli):
    """Combine residue arrays into the unique value in 0 .. prod(moduli) - 1."""
    if not pairwise_coprime(moduli):
        raise ValueError("moduli must be pairwise coprime")
    big_m = prod(moduli)
    total = np.zeros(residues[0].shape, dtype=np.int64)
    for residue, m in zip(residues, moduli):
        partial = big_m // m
        total += residue.astype(np.int64) * partial * pow(partial, -1, m)
    return total % big_m


def decode(residues, moduli=LEGACY_MODULI):
    """Rebuild the image. Values at or above prod(moduli) cannot come back."""
    return (crt(residues, moduli) % 256).astype(np.uint8)
