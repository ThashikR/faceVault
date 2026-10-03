"""Standard image-cipher measurements."""

import math

import numpy as np


def entropy(data):
    """Shannon entropy in bits per byte. 8.0 is the ideal for ciphertext."""
    counts = np.bincount(np.asarray(data, dtype=np.uint8).ravel(), minlength=256)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def adjacent_correlation(image, direction="horizontal"):
    """Correlation between neighbouring pixels. Near 1 for photos, near 0 for good ciphertext."""
    a = np.asarray(image, dtype=np.float64)
    if direction == "horizontal":
        x, y = a[:, :-1], a[:, 1:]
    elif direction == "vertical":
        x, y = a[:-1, :], a[1:, :]
    elif direction == "diagonal":
        x, y = a[:-1, :-1], a[1:, 1:]
    else:
        raise ValueError(f"unknown direction: {direction}")
    x, y = x.ravel(), y.ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def npcr(a, b):
    """Number of Pixels Change Rate, in percent. About 99.61 for two random images."""
    return float(np.mean(np.asarray(a) != np.asarray(b)) * 100.0)


def uaci(a, b):
    """Unified Average Changing Intensity, in percent. About 33.46 for two random images."""
    diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    return float(diff.mean() / 255.0 * 100.0)


def mse(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff * diff))


def psnr(a, b):
    """Peak signal-to-noise ratio in dB. Infinite when the images are identical."""
    error = mse(a, b)
    return math.inf if error == 0 else float(10.0 * math.log10(255.0 * 255.0 / error))
