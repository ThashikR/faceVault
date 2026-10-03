"""Download the two OpenCV face models into models/ and check their SHA-256.

    python scripts/fetch_models.py

Source: https://github.com/opencv/opencv_zoo (YuNet: MIT licence, SFace: Apache-2.0).
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

# name -> (url, expected SHA-256). The hashes are those of the files this
# project was developed and measured with.
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    failed = False
    for name, (url, expected) in MODELS.items():
        target = MODEL_DIR / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)
        actual = sha256(target)
        if expected and actual != expected:
            print(f"  HASH MISMATCH for {name}\n    expected {expected}\n    got      {actual}")
            failed = True
        else:
            print(f"  ok  {name}  {target.stat().st_size / 1e6:.2f} MB  sha256={actual}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
