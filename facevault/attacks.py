"""Attacks on the legacy residue scheme. None of them needs a key, because there is none."""

from math import prod

import numpy as np

from . import legacy_rns


def infer_moduli(shares):
    """Read each modulus straight off its residue image.

    Residues mod m take the values 0 .. m-1, so on any natural image the
    largest value seen is m-1. Keeping the moduli secret therefore does not help.
    """
    return tuple(int(share.max()) + 1 for share in shares)


def recover_without_key(shares):
    """Rebuild the image from the residue files alone. Returns (image, moduli used)."""
    moduli = infer_moduli(shares)
    return legacy_rns.decode(shares, moduli), moduli


def stretch_share(share):
    """Brighten one residue image so a person (or a face detector) can look at it."""
    top = max(int(share.max()), 1)
    return (share.astype(np.float32) * (255.0 / top)).round().astype(np.uint8)


def lost_pixel_fraction(image, moduli=legacy_rns.LEGACY_MODULI):
    """Share of pixel values the moduli cannot represent (255 for the report's set)."""
    return float(np.mean(image >= prod(moduli)))
