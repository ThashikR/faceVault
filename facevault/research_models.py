"""Stronger, research-only face models: SCRFD-10G (detector) and ArcFace R50 (recogniser).

They come from InsightFace's "buffalo_l" pack and are run with onnxruntime.
InsightFace licenses these model files for non-commercial research only, so
they are optional: FaceVault itself works without them, with YuNet and SFace.
Fetch them with scripts/fetch_research_models.py.
"""

import cv2
import numpy as np

from .detect import MODEL_DIR, Face

# SCRFD's scores run lower than YuNet's. On the WIDER FACE validation set this
# confidence locked 95.0% of faces 32 px and taller for 9.4% of the image area
# (YuNet at its lock confidence: 87.2% for 8.5%). It was chosen on that set.
SCRFD_LOCK_THRESHOLD = 0.2

SCRFD = MODEL_DIR / "buffalo_l" / "det_10g.onnx"
ARCFACE = MODEL_DIR / "buffalo_l" / "w600k_r50.onnx"

# Where the five landmarks sit in ArcFace's 112 x 112 aligned crop.
_ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _session(path):
    import onnxruntime

    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run: python scripts/fetch_research_models.py")
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class ScrfdDetector:
    """SCRFD-10G. `long_side` is the size the image's longer side is scaled to before detection."""

    _STRIDES = (8, 16, 32)
    _ANCHORS = 2

    def __init__(self, score_threshold=SCRFD_LOCK_THRESHOLD, long_side=640, nms_threshold=0.4):
        self._net = _session(SCRFD)
        self._input = self._net.get_inputs()[0].name
        self.score_threshold, self.long_side, self.nms_threshold = score_threshold, long_side, nms_threshold

    def detect(self, rgb):
        """Return the faces found, most confident first, in the same form as FaceDetector.detect."""
        height, width = rgb.shape[:2]
        scale = self.long_side / max(height, width)
        new_w, new_h = max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)
        # The network needs sides that are multiples of its coarsest stride.
        canvas = np.zeros((-(-new_h // 32) * 32, -(-new_w // 32) * 32, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(rgb, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (canvas.shape[1], canvas.shape[0]), (127.5, 127.5, 127.5), swapRB=False)
        outputs = self._net.run(None, {self._input: blob})

        boxes, scores, points = [], [], []
        for index, stride in enumerate(self._STRIDES):
            score = outputs[index].reshape(-1)
            distance = outputs[index + 3].reshape(-1, 4) * stride
            offsets = outputs[index + 6].reshape(-1, 5, 2) * stride
            rows, cols = canvas.shape[0] // stride, canvas.shape[1] // stride
            centres = np.stack(np.mgrid[:rows, :cols][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            centres = np.repeat(centres, self._ANCHORS, axis=0)
            keep = score >= self.score_threshold
            c, d = centres[keep], distance[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
            scores.append(score[keep])
            points.append(c[:, None, :] + offsets[keep])
        boxes, scores, points = np.concatenate(boxes) / scale, np.concatenate(scores), np.concatenate(points) / scale
        if not len(scores):
            return []

        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.score_threshold, self.nms_threshold)
        faces = []
        for i in np.asarray(kept).reshape(-1):
            # Same layout as a YuNet row: box, five landmarks (image-left eye first), score.
            row = np.concatenate([xywh[i], points[i].reshape(-1), scores[i:i + 1]]).astype(np.float32)
            faces.append(Face(tuple(int(round(v)) for v in xywh[i]), float(scores[i]), row))
        return sorted(faces, key=lambda face: face.score, reverse=True)


class ArcFaceMatcher:
    """ArcFace ResNet-50 trained on WebFace600K. A second recogniser, independent of SFace."""

    def __init__(self):
        self._net = _session(ARCFACE)
        self._input = self._net.get_inputs()[0].name

    def embed(self, rgb, face):
        """Align the face by its five landmarks and return a unit-length feature vector."""
        from skimage.transform import SimilarityTransform

        landmarks = np.asarray(face.row[4:14], dtype=np.float32).reshape(5, 2)
        transform = SimilarityTransform.from_estimate(landmarks, _ARCFACE_TEMPLATE)
        aligned = cv2.warpAffine(rgb, transform.params[:2], (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
        feature = self._net.run(None, {self._input: blob})[0][0]
        return feature / np.linalg.norm(feature)

    def similarity(self, a, b):
        return float(np.dot(a, b))
