"""The usual ways of hiding a face: blur, pixelation, a black box.

They are here as baselines to measure FaceVault against. None of them can be
undone, so the original face is lost even to someone who is entitled to see it.
"""

import cv2


def blur(image, boxes, kernel):
    """Gaussian-blur each (x, y, w, h) box. `kernel` is the blur width in pixels (odd)."""
    out = image.copy()
    for x, y, w, h in boxes:
        out[y:y + h, x:x + w] = cv2.GaussianBlur(image[y:y + h, x:x + w], (kernel, kernel), 0)
    return out


def pixelate(image, boxes, block):
    """Replace each box with square blocks of `block` pixels, each one flat colour."""
    out = image.copy()
    for x, y, w, h in boxes:
        small = cv2.resize(image[y:y + h, x:x + w], (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_AREA)
        out[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def black_box(image, boxes):
    out = image.copy()
    for x, y, w, h in boxes:
        out[y:y + h, x:x + w] = 0
    return out
