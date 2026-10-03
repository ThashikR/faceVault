"""Tests that use the real face models. Skipped until scripts/fetch_models.py has been run."""

import numpy as np
import pytest

from facevault import detect, hybrid, vault

pytestmark = pytest.mark.skipif(not detect.YUNET.exists() or not detect.SFACE.exists(), reason="face models not downloaded")


@pytest.fixture(scope="module")
def astronaut():
    # NASA portrait of Eileen Collins, public domain, bundled with scikit-image.
    from skimage import data

    return np.ascontiguousarray(data.astronaut())


@pytest.fixture(scope="module")
def detector():
    return detect.FaceDetector()


def test_detector_finds_the_face(astronaut, detector):
    faces = detector.detect(astronaut)
    assert len(faces) == 1
    x, y, w, h = faces[0].box
    assert w > 50 and h > 50


def test_enlarged_pass_reports_boxes_in_original_pixels(astronaut, detector):
    normal = detector.detect(astronaut)[0].box
    enlarged = detector.detect_enlarged(astronaut, 2)[0].box
    assert all(abs(a - b) <= 12 for a, b in zip(normal, enlarged))


def test_protected_image_hides_the_face_and_reveal_restores_it(astronaut, detector):
    private, public = hybrid.generate_keypair()
    faces = detector.detect(astronaut)
    protected, header = vault.protect(astronaut, [face.box for face in faces], public)
    assert detector.detect(protected) == []
    restored = vault.reveal(protected, header, private).image
    assert np.array_equal(restored, astronaut)
    assert len(detector.detect(restored)) == 1


def test_watchlist_matches_the_same_person_only(astronaut, detector):
    watchlist = detect.Watchlist(detector, detect.FaceMatcher())
    assert watchlist.add("collins", astronaut)
    face = detector.detect(astronaut)[0]
    label, score = watchlist.match(astronaut, face)
    assert label == "collins" and score > 0.9

    mirrored = np.ascontiguousarray(astronaut[:, ::-1])
    hit = watchlist.match(mirrored, detector.detect(mirrored)[0])
    assert hit is not None and hit[0] == "collins"
