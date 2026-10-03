"""Download the WIDER FACE validation set (about 365 MB) into data/wider_face/.

    python scripts/fetch_wider_face.py

Used only by the crowd-scene step of experiments/run_all.py.
Source: the dataset authors' copy at https://huggingface.co/datasets/CUHK-CSE/wider_face
"""

import urllib.request
import zipfile
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "data" / "wider_face"
BASE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
FILES = ("wider_face_split.zip", "WIDER_val.zip")


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        archive = TARGET / name
        if not archive.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE}/{name}", archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(TARGET)
        print(f"  ok  {name}  {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
