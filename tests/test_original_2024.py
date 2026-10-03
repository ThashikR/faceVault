"""The original 2024 code (original-2024/my_app.py), run exactly as written.

my_app.py imports face_recognition at the top, which needs dlib, so it cannot
be imported here. Its encryption and decryption functions do not use
face_recognition, so they are lifted out of the file unchanged and run.
"""

import ast
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facevault import attacks, legacy_rns

APP = Path(__file__).resolve().parent.parent / "original-2024" / "my_app.py"
WANTED = {"multiplicative_inverse", "crt", "decrypt_image", "decrypt_folder_for_filename", "encrypt_image"}


class _Screen:
    """Stands in for Streamlit: the original functions report success through st."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture(scope="module")
def original():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in WANTED]
    assert {f.name for f in functions} == WANTED
    namespace = {"os": os, "np": np, "Image": Image, "st": _Screen()}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace


@pytest.fixture
def photo(tmp_path):
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)   # 0..254
    pixels[30:, 40:] = 255                                             # a white patch, well past pixel 500
    path = tmp_path / "grp-2.png"
    Image.fromarray(pixels).save(path)
    return path, pixels


def _files(folder):
    return [np.array(Image.open(folder / f"encrypted_{k}.png")) for k in range(3)]


def test_original_encryption_writes_three_residue_images(original, photo, tmp_path):
    path, pixels = photo
    out = tmp_path / "output_folder" / "grp-2"
    original["encrypt_image"](str(path), str(out))
    flat = pixels.reshape(-1, 3)
    for file, modulus in zip(_files(out), (3, 5, 17)):
        stored = file.reshape(-1, 3)
        assert np.array_equal(stored[:500], flat[:500])          # the first 500 pixels are unaltered
        assert np.array_equal(stored[500:], flat[500:] % modulus)


def test_original_decryption_restores_everything_except_white(original, photo, tmp_path):
    path, pixels = photo
    original["encrypt_image"](str(path), str(tmp_path / "output_folder" / "grp-2"))
    original["decrypt_folder_for_filename"](str(tmp_path / "output_folder"), "grp-2")
    decrypted = np.array(Image.open(tmp_path / "output_folder" / "grp-2" / "decrypted_image.png"))
    white = pixels == 255
    assert np.array_equal(decrypted[~white], pixels[~white])
    assert np.all(decrypted[white] == 0)   # the defect behind Fig 7.1.6 of the report


def test_the_baseline_used_in_the_experiments_matches_the_original_code(original, photo, tmp_path):
    """facevault/legacy_rns.py must behave exactly like the 2024 code it stands in for."""
    path, pixels = photo
    out = tmp_path / "output_folder" / "grp-2"
    original["encrypt_image"](str(path), str(out))
    original["decrypt_folder_for_filename"](str(tmp_path / "output_folder"), "grp-2")

    ours = legacy_rns.encode(pixels, unaltered=legacy_rns.UNALTERED_PIXELS)
    for theirs, mine in zip(_files(out), ours):
        assert np.array_equal(theirs, mine)
    assert np.array_equal(np.array(Image.open(out / "decrypted_image.png")), legacy_rns.decode(ours))


def test_the_attack_rebuilds_the_original_output_without_knowing_the_moduli(original, photo, tmp_path):
    path, _ = photo
    out = tmp_path / "output_folder" / "grp-2"
    original["encrypt_image"](str(path), str(out))
    recovered, moduli = attacks.recover_without_key(_files(out))
    assert moduli == (3, 5, 17)
    original["decrypt_folder_for_filename"](str(tmp_path / "output_folder"), "grp-2")
    assert np.array_equal(recovered, np.array(Image.open(out / "decrypted_image.png")))
