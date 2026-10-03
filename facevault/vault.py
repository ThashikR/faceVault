"""Lock the face regions of an image and unlock them again, bit for bit.

Each face rectangle is encrypted with AES-256-GCM and the ciphertext is written
back into the same pixels, so the protected image shows noise where the faces
were and the rest of the scene is untouched. Everything needed to unlock it,
except the vault's private key, travels inside the PNG file.

The key for each image comes from the vault's PUBLIC key (see hybrid.py), so
the device that locks faces cannot unlock them.
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import hybrid

VERSION = 1
ALGORITHM = "X25519-HKDF-SHA256+AES-256-GCM"
PNG_KEY = "facevault"
_INFO = b"facevault/v1/image"
_NONCE_BYTES = 12
_TAG_BYTES = 16


class TamperedError(Exception):
    """The protected image or its header was changed, or the wrong key was used."""


@dataclass
class RevealResult:
    image: np.ndarray
    # False when pixels OUTSIDE the face regions were edited after protection.
    background_intact: bool


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _unb64(text):
    return base64.b64decode(text.encode("ascii"))


def canonical_json(obj):
    """One fixed byte form per object, so both sides authenticate the same bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def prepare_boxes(boxes, shape, margin=0.15):
    """Grow each (x, y, w, h) box by a margin, clip it to the image, merge overlaps."""
    height, width = shape[:2]
    grown = []
    for x, y, w, h in boxes:
        dx, dy = w * margin, h * margin
        x0, y0 = max(int(round(x - dx)), 0), max(int(round(y - dy)), 0)
        x1, y1 = min(int(round(x + w + dx)), width), min(int(round(y + h + dy)), height)
        if x1 > x0 and y1 > y0:
            grown.append([x0, y0, x1 - x0, y1 - y0])

    merged = True
    while merged:
        merged = False
        for i in range(len(grown)):
            for j in range(i + 1, len(grown)):
                a, b = grown[i], grown[j]
                if _overlap(a, b):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    grown[i] = [x0, y0, x1 - x0, y1 - y0]
                    del grown[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(grown, key=lambda box: (box[1], box[0]))


def _background_hash(image, boxes):
    """SHA-256 of everything outside the face regions."""
    blanked = image.copy()
    for x, y, w, h in boxes:
        blanked[y:y + h, x:x + w] = 0
    digest = hashlib.sha256(repr(blanked.shape).encode("ascii"))
    digest.update(blanked.tobytes())
    return digest.hexdigest()


def _aad(core, index):
    return canonical_json(core) + b"|" + str(index).encode("ascii")


def protect(image, boxes, vault_public, context=None, margin=0.15):
    """Encrypt the given face boxes in place. Returns (protected_image, header).

    `context` (camera id, time, ...) is stored readable in the header and is
    bound to the ciphertext, so it cannot be swapped without detection.
    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    boxes = prepare_boxes(boxes, image.shape, margin)
    ephemeral_public, key = hybrid.sender_key(vault_public, _INFO)

    core = {
        "v": VERSION,
        "alg": ALGORITHM,
        "shape": list(image.shape),
        "eph": _b64(ephemeral_public),
        "context": context or {},
        "bg_sha256": _background_hash(image, boxes),
        "boxes": boxes,
    }

    aes = AESGCM(key)
    protected = image.copy()
    rois = []
    for index, (x, y, w, h) in enumerate(boxes):
        nonce = os.urandom(_NONCE_BYTES)
        region = image[y:y + h, x:x + w]
        sealed = aes.encrypt(nonce, region.tobytes(), _aad(core, index))
        ciphertext, tag = sealed[:-_TAG_BYTES], sealed[-_TAG_BYTES:]
        protected[y:y + h, x:x + w] = np.frombuffer(ciphertext, dtype=np.uint8).reshape(region.shape)
        rois.append({"nonce": _b64(nonce), "tag": _b64(tag)})

    header = dict(core)
    header["rois"] = rois
    return protected, header


def reveal(protected, header, vault_private):
    """Decrypt every face region. Raises TamperedError if anything was altered."""
    protected = np.ascontiguousarray(protected, dtype=np.uint8)
    try:
        core = {name: header[name] for name in ("v", "alg", "shape", "eph", "context", "bg_sha256", "boxes")}
        rois = header["rois"]
        if core["v"] != VERSION or list(protected.shape) != core["shape"] or len(rois) != len(core["boxes"]):
            raise TamperedError("header does not match this image")
        key = hybrid.recipient_key(vault_private, _unb64(core["eph"]), _INFO)
    except (KeyError, TypeError, ValueError) as error:
        raise TamperedError(f"unreadable header: {error}") from error

    aes = AESGCM(key)
    image = protected.copy()
    for index, ((x, y, w, h), roi) in enumerate(zip(core["boxes"], rois)):
        region = protected[y:y + h, x:x + w]
        try:
            plain = aes.decrypt(_unb64(roi["nonce"]), region.tobytes() + _unb64(roi["tag"]), _aad(core, index))
        except (InvalidTag, ValueError) as error:
            raise TamperedError(f"face region {index} failed authentication") from error
        image[y:y + h, x:x + w] = np.frombuffer(plain, dtype=np.uint8).reshape(region.shape)

    intact = _background_hash(protected, core["boxes"]) == core["bg_sha256"]
    return RevealResult(image=image, background_intact=intact)


def save_png(path, image, header):
    """Write the protected image with its header in a PNG text chunk (lossless)."""
    info = PngInfo()
    info.add_text(PNG_KEY, json.dumps(header, sort_keys=True))
    Image.fromarray(image).save(path, format="PNG", pnginfo=info)


def load_png(path):
    """Read a protected PNG. Returns (image, header)."""
    with Image.open(path) as picture:
        text = getattr(picture, "text", {}).get(PNG_KEY)
        if text is None:
            raise ValueError(f"{path} is not a FaceVault image (no header found)")
        return np.array(picture.convert("RGB")), json.loads(text)


def load_image(path):
    """Read any ordinary image file as an RGB array."""
    with Image.open(path) as picture:
        return np.array(picture.convert("RGB"))
