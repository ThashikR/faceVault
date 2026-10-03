"""Face detection (YuNet) and watch-list matching (SFace), both run by OpenCV on the CPU."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine score above which OpenCV's SFace documentation treats two faces as the same person.
MATCH_THRESHOLD = 0.363

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Face:
    box: tuple          # (x, y, w, h) in pixels
    score: float
    row: np.ndarray     # YuNet's raw output row (box, 5 landmarks, score), needed for alignment


def _bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _require(path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run: python scripts/fetch_models.py")
    return str(path)


class FaceDetector:
    def __init__(self, score_threshold=0.8):
        self._net = cv2.FaceDetectorYN.create(_require(YUNET), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, rgb):
        """Return the faces found in an RGB image, most confident first."""
        height, width = rgb.shape[:2]
        self._net.setInputSize((width, height))
        _, rows = self._net.detect(_bgr(rgb))
        if rows is None:
            return []
        faces = [Face(tuple(int(round(v)) for v in row[:4]), float(row[-1]), row) for row in rows]
        return sorted(faces, key=lambda face: face.score, reverse=True)


class FaceMatcher:
    def __init__(self):
        self._net = cv2.FaceRecognizerSF.create(_require(SFACE), "")

    def embed(self, rgb, face):
        """Turn one detected face into a feature vector."""
        aligned = self._net.alignCrop(_bgr(rgb), face.row)
        return self._net.feature(aligned).copy()

    def similarity(self, a, b):
        return float(self._net.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


class Watchlist:
    """People to raise an alert for. One image per person; the file name is the label."""

    def __init__(self, detector, matcher, threshold=MATCH_THRESHOLD):
        self.detector, self.matcher, self.threshold = detector, matcher, threshold
        self.entries = []   # (label, feature)

    def add(self, label, rgb):
        faces = self.detector.detect(rgb)
        if not faces:
            return False
        self.entries.append((label, self.matcher.embed(rgb, faces[0])))
        return True

    def add_folder(self, folder):
        from .vault import load_image

        skipped = []
        for path in sorted(Path(folder).iterdir()):
            if path.suffix.lower() in _IMAGE_SUFFIXES and not self.add(path.stem, load_image(path)):
                skipped.append(path.name)
        return skipped

    def match(self, rgb, face):
        """Return (label, score) for the best match above the threshold, or None."""
        if not self.entries:
            return None
        feature = self.matcher.embed(rgb, face)
        label, score = max(((name, self.matcher.similarity(feature, known)) for name, known in self.entries), key=lambda item: item[1])
        return (label, score) if score >= self.threshold else None
