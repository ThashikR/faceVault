"""Download the optional research-only face models (about 290 MB) into models/buffalo_l/.

    pip install onnxruntime
    python scripts/fetch_research_models.py

Source: InsightFace, https://github.com/deepinsight/insightface (model pack "buffalo_l").
Licence of the model files: non-commercial research use only. They are used by
the experiments to compare a stronger detector and a second recogniser; the
FaceVault app does not need them.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# The two files the experiments use, with the SHA-256 of the copies they were measured with.
EXPECTED = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    target = MODEL_DIR / "buffalo_l"
    archive = MODEL_DIR / "buffalo_l.zip"
    target.mkdir(parents=True, exist_ok=True)
    if not all((target / name).exists() for name in EXPECTED):
        if not archive.exists():
            print("downloading buffalo_l.zip ...")
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    failed = False
    for name, expected in EXPECTED.items():
        actual = sha256(target / name)
        if expected and actual != expected:
            print(f"  HASH MISMATCH for {name}\n    expected {expected}\n    got      {actual}")
            failed = True
        else:
            print(f"  ok  {name}  {(target / name).stat().st_size / 1e6:.1f} MB  sha256={actual}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
