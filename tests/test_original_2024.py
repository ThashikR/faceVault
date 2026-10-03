"""The reconstructed 2024 app: its encryption and decryption behave as the report describes.

The face-matching functions need dlib and are not exercised here.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

APP = Path(__file__).resolve().parent.parent / "original-2024" / "app.py"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("original_2024_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # main() does not run: the module is not __main__
    return module


@pytest.fixture
def photo(tmp_path):
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)   # 0..254
    pixels[30:, 40:] = 255                                             # a white patch, well past pixel 500
    path = tmp_path / "grp-2.png"
    Image.fromarray(pixels).save(path)
    return path, pixels


def test_multiplicative_inverse(app):
    assert (85 * app.multiplicative_inverse(85, 3)) % 3 == 1
    assert (15 * app.multiplicative_inverse(15, 17)) % 17 == 1


def test_encrypt_writes_three_images_in_a_folder_named_after_the_image(app, photo, tmp_path):
    path, _ = photo
    written = app.encrypt_image(path, tmp_path / "output_folder")
    assert [p.name for p in written] == ["encrypted_0.png", "encrypted_1.png", "encrypted_2.png"]
    assert all(p.parent == tmp_path / "output_folder" / "grp-2" for p in written)


def test_first_500_pixels_are_left_unaltered(app, photo, tmp_path):
    path, pixels = photo
    flat = pixels.reshape(-1, 3)
    for index, encrypted_path in enumerate(app.encrypt_image(path, tmp_path / "out")):
        encrypted = np.array(Image.open(encrypted_path)).reshape(-1, 3)
        assert np.array_equal(encrypted[:500], flat[:500])
        assert encrypted[500:].max() < app.MODULI[index]


def test_decryption_restores_everything_except_white(app, photo, tmp_path):
    path, pixels = photo
    app.encrypt_image(path, tmp_path / "out")
    decrypted_path = app.decrypt_folder_for_filename(tmp_path / "out", "grp-2")
    assert decrypted_path.name == "decrypted_image.png"
    decrypted = np.array(Image.open(decrypted_path))
    white = pixels == 255
    assert np.array_equal(decrypted[~white], pixels[~white])
    assert np.all(decrypted[white] == 0)   # the defect behind Fig 7.1.6 of the report


def test_app_decryption_screen(app, photo, tmp_path):
    from streamlit.testing.v1 import AppTest

    path, _ = photo
    out = tmp_path / "out"
    app.encrypt_image(path, out)

    screen = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not screen.exception
    assert screen.title[0].value == "Surveillance System for Criminal Detection"

    screen.sidebar.radio[0].set_value("Decryption").run()
    screen.text_input[0].set_value(str(out))
    screen.text_input[1].set_value("grp-2")
    screen.button[0].click().run()
    assert not screen.exception
    assert screen.success[0].value.startswith("Decryption successful")
    assert (out / "grp-2" / "decrypted_image.png").exists()
