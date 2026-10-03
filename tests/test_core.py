"""Tests that need no model files or datasets."""

import itertools
import json

import numpy as np
import pytest

from facevault import alerts, attacks, hybrid, legacy_rns, metrics, obfuscate, pipeline, sharing, vault
from facevault.audit import AuditLog


@pytest.fixture
def image():
    rng = np.random.default_rng(7)
    picture = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    picture[:20, :20] = 255   # a blown-out patch, like a bright window
    return picture


@pytest.fixture
def keypair():
    return hybrid.generate_keypair()


# --- the legacy scheme and why it fails -------------------------------------

def test_legacy_scheme_loses_white_pixels(image):
    restored = legacy_rns.decode(legacy_rns.encode(image))
    assert np.all(restored[image == 255] == 0)
    assert np.array_equal(restored[image != 255], image[image != 255])
    assert attacks.lost_pixel_fraction(image) > 0


def test_range_fixed_moduli_are_lossless(image):
    moduli = legacy_rns.RANGE_FIXED_MODULI
    assert np.array_equal(legacy_rns.decode(legacy_rns.encode(image, moduli), moduli), image)


def test_legacy_scheme_is_recovered_without_any_key(image):
    recovered, moduli = attacks.recover_without_key(legacy_rns.encode(image))
    assert moduli == legacy_rns.LEGACY_MODULI
    assert np.array_equal(recovered, legacy_rns.decode(legacy_rns.encode(image)))


def test_crt_rejects_moduli_that_share_a_factor():
    with pytest.raises(ValueError):
        legacy_rns.crt([np.zeros(1), np.zeros(1)], (4, 6))


# --- face-region encryption -------------------------------------------------

BOXES = [(30, 20, 40, 50), (100, 60, 30, 30)]


def test_protect_then_reveal_is_bit_exact(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, BOXES, public, {"camera": "t"})
    result = vault.reveal(protected, header, private)
    assert np.array_equal(result.image, image)
    assert result.background_intact


def test_only_face_regions_change(image, keypair):
    _, public = keypair
    protected, header = vault.protect(image, BOXES, public)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for x, y, w, h in header["boxes"]:
        mask[y:y + h, x:x + w] = True
        assert not np.array_equal(protected[y:y + h, x:x + w], image[y:y + h, x:x + w])
    assert np.array_equal(protected[~mask], image[~mask])


def test_wrong_key_is_rejected(image, keypair):
    _, public = keypair
    protected, header = vault.protect(image, BOXES, public)
    other_private, _ = hybrid.generate_keypair()
    with pytest.raises(vault.TamperedError):
        vault.reveal(protected, header, other_private)


def test_one_flipped_bit_in_a_face_is_detected(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, BOXES, public)
    x, y, _, _ = header["boxes"][0]
    protected[y, x, 0] ^= 1
    with pytest.raises(vault.TamperedError):
        vault.reveal(protected, header, private)


def test_changed_context_is_detected(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, BOXES, public, {"camera": "gate-1"})
    header["context"]["camera"] = "gate-2"
    with pytest.raises(vault.TamperedError):
        vault.reveal(protected, header, private)


def test_edited_background_is_reported(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, BOXES, public)
    protected[0, 0, 0] ^= 1   # outside every face box
    result = vault.reveal(protected, header, private)
    assert not result.background_intact


def test_overlapping_boxes_are_merged(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, [(10, 10, 40, 40), (30, 30, 40, 40)], public)
    assert len(header["boxes"]) == 1
    assert np.array_equal(vault.reveal(protected, header, private).image, image)


def test_image_without_faces_round_trips(image, keypair):
    private, public = keypair
    protected, header = vault.protect(image, [], public)
    assert np.array_equal(protected, image)
    assert np.array_equal(vault.reveal(protected, header, private).image, image)


def test_png_file_carries_everything_but_the_key(image, keypair, tmp_path):
    private, public = keypair
    protected, header = vault.protect(image, BOXES, public)
    path = tmp_path / "protected.png"
    vault.save_png(path, protected, header)
    loaded, loaded_header = vault.load_png(path)
    assert np.array_equal(vault.reveal(loaded, loaded_header, private).image, image)


def test_ciphertext_looks_random(keypair):
    _, public = keypair
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)   # worst case: a featureless image
    protected, _ = vault.protect(flat, [(0, 0, 200, 200)], public, margin=0)
    assert metrics.entropy(protected) > 7.99
    assert abs(metrics.adjacent_correlation(protected[:, :, 0])) < 0.05


# --- baselines: blur, pixelation, black box ----------------------------------

@pytest.mark.parametrize("hide", [
    lambda image, boxes: obfuscate.blur(image, boxes, 15),
    lambda image, boxes: obfuscate.pixelate(image, boxes, 8),
    obfuscate.black_box,
])
def test_baselines_change_only_the_boxes(image, hide):
    hidden = hide(image, BOXES)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for x, y, w, h in BOXES:
        mask[y:y + h, x:x + w] = True
        assert not np.array_equal(hidden[y:y + h, x:x + w], image[y:y + h, x:x + w])
    assert np.array_equal(hidden[~mask], image[~mask])


def test_pixelation_makes_flat_blocks(image):
    x, y, w, h = 32, 24, 48, 40
    region = obfuscate.pixelate(image, [(x, y, w, h)], 8)[y:y + h, x:x + w]
    assert np.all(region[:8, :8] == region[0, 0])


# --- images signed by the camera ---------------------------------------------

def test_signed_image_is_accepted_from_the_trusted_camera(image, keypair):
    private, public = keypair
    camera_private, camera_public = alerts.generate_signing_keypair()
    protected, header = vault.protect(image, BOXES, public, {"camera": "gate-1"}, signing_private=camera_private)
    assert vault.signed_by(header, camera_public)
    assert np.array_equal(vault.reveal(protected, header, private, trusted_camera=camera_public).image, image)


def test_image_made_without_the_camera_key_is_refused(image, keypair):
    """Anyone holding the public vault key can make a protected image; only the camera can sign one."""
    private, public = keypair
    _, camera_public = alerts.generate_signing_keypair()
    forger_private, _ = alerts.generate_signing_keypair()
    for signing_private in (None, forger_private):
        protected, header = vault.protect(image, BOXES, public, signing_private=signing_private)
        with pytest.raises(vault.TamperedError):
            vault.reveal(protected, header, private, trusted_camera=camera_public)


def test_signature_cannot_be_stripped_or_moved(image, keypair):
    private, public = keypair
    camera_private, camera_public = alerts.generate_signing_keypair()
    protected, header = vault.protect(image, BOXES, public, {"camera": "gate-1"}, signing_private=camera_private)

    stripped = {name: value for name, value in header.items() if name not in ("signature", "camera")}
    with pytest.raises(vault.TamperedError):
        vault.reveal(protected, stripped, private)            # the camera field is bound to the ciphertext

    edited = {**header, "context": {"camera": "gate-2"}}
    assert not vault.signed_by(edited, camera_public)


# --- capture log: a deleted or swapped image shows up --------------------------

def test_capture_log_reports_missing_and_altered_images(image, tmp_path):
    keys, store = tmp_path / "keys", tmp_path / "store"
    pipeline.setup_keys(keys)
    store.mkdir()
    public = pipeline.read_key(keys / pipeline.VAULT_PUBLIC)
    camera_private = pipeline.read_key(keys / pipeline.CAMERA_KEY)
    for name in ("a.png", "b.png", "c.png"):
        protected, header = vault.protect(image, BOXES, public, signing_private=camera_private)
        vault.save_png(store / name, protected, header)
        pipeline.record_capture(keys, store / name, "gate-1")

    log_ok, rows = pipeline.check_captures(keys, store)
    assert log_ok and [status for _, status in rows] == ["ok", "ok", "ok"]

    (store / "b.png").unlink()                                  # someone deletes an image
    other, other_header = vault.protect(image, BOXES, public)   # someone swaps another
    vault.save_png(store / "c.png", other, other_header)
    log_ok, rows = pipeline.check_captures(keys, store)
    assert log_ok and {entry["target"]: status for entry, status in rows} == {"a.png": "ok", "b.png": "missing", "c.png": "altered"}


def test_capture_log_cannot_be_rewritten_without_the_camera_key(image, tmp_path):
    keys, store = tmp_path / "keys", tmp_path / "store"
    pipeline.setup_keys(keys)
    store.mkdir()
    public = pipeline.read_key(keys / pipeline.VAULT_PUBLIC)
    for name in ("a.png", "b.png"):
        protected, header = vault.protect(image, BOXES, public)
        vault.save_png(store / name, protected, header)
        pipeline.record_capture(keys, store / name, "gate-1")
    assert pipeline.check_captures(keys, store)[0]

    # An insider removes the first entry and rebuilds a consistent chain, but cannot sign it.
    (keys / pipeline.CAPTURE_LOG).unlink()
    AuditLog(keys / pipeline.CAPTURE_LOG).append("capture", ["gate-1"], "b.png", "sha256:" + pipeline.file_digest(store / "b.png"))
    assert AuditLog(keys / pipeline.CAPTURE_LOG).verify() == (True, None)      # the chain alone looks fine
    assert not pipeline.check_captures(keys, store)[0]                          # the signature check does not


# --- threshold key sharing --------------------------------------------------

@pytest.mark.parametrize("k,n", [(2, 3), (3, 5), (4, 4)])
def test_any_k_shares_rebuild_the_key(k, n):
    secret = hybrid.generate_keypair()[0]
    shares = sharing.split(secret, k, n)
    for subset in itertools.combinations(shares, k):
        assert sharing.combine(list(subset)) == secret


def test_fewer_than_k_shares_do_not_rebuild_the_key():
    secret = hybrid.generate_keypair()[0]
    shares = sharing.split(secret, 3, 5)
    with pytest.raises(ValueError):
        sharing.combine(shares[:2])
    with pytest.raises(ValueError):
        sharing.combine([shares[0], shares[0], shares[0]])   # the same share three times
    assert sharing.reconstruct_unchecked(shares[:2]) != secret


def test_shares_from_different_keys_do_not_mix():
    a = sharing.split(hybrid.generate_keypair()[0], 2, 3)
    b = sharing.split(hybrid.generate_keypair()[0], 2, 3)
    with pytest.raises(ValueError):
        sharing.combine([a[0], b[1]])


def test_share_survives_json(tmp_path):
    secret = hybrid.generate_keypair()[0]
    shares = [sharing.Share.from_json(share.to_json()) for share in sharing.split(secret, 2, 3)]
    assert sharing.combine(shares[1:]) == secret


# --- alerts -----------------------------------------------------------------

ALERT = {"event": "watchlist-match", "matches": [{"label": "person-a", "score": 0.71}]}


def test_alert_round_trip(keypair):
    private, public = keypair
    sign_private, sign_public = alerts.generate_signing_keypair()
    envelope = alerts.seal_alert(ALERT, public, sign_private)
    assert "person-a" not in json.dumps(envelope)
    assert alerts.open_alert(envelope, private, sign_public) == ALERT


def test_alert_from_unknown_sender_is_rejected(keypair):
    private, public = keypair
    forger_private, _ = alerts.generate_signing_keypair()
    _, trusted_public = alerts.generate_signing_keypair()
    with pytest.raises(alerts.AlertError):
        alerts.open_alert(alerts.seal_alert(ALERT, public, forger_private), private, trusted_public)


def test_altered_alert_is_rejected(keypair):
    private, public = keypair
    sign_private, sign_public = alerts.generate_signing_keypair()
    envelope = alerts.seal_alert(ALERT, public, sign_private)
    envelope["nonce"] = envelope["nonce"][::-1]
    with pytest.raises(alerts.AlertError):
        alerts.open_alert(envelope, private, sign_public)


def test_alert_for_someone_else_cannot_be_read(keypair):
    _, public = keypair
    other_private, _ = hybrid.generate_keypair()
    sign_private, sign_public = alerts.generate_signing_keypair()
    with pytest.raises(alerts.AlertError):
        alerts.open_alert(alerts.seal_alert(ALERT, public, sign_private), other_private, sign_public)


# --- audit log --------------------------------------------------------------

def test_audit_log_detects_edits(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("reveal", ["officer-1", "officer-2"], "a.png", "case 1")
    log.append("reveal", ["officer-1", "officer-3"], "b.png", "case 2")
    log.append("reveal-denied", ["officer-2"], "c.png", "case 3")
    assert log.verify() == (True, None)

    lines = log.path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("case 2", "case 9")
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify() == (False, 2)


def test_audit_log_detects_deleted_entry(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for case in range(3):
        log.append("reveal", ["officer-1", "officer-2"], f"{case}.png", "case")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    log.path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert log.verify()[0] is False


# --- the whole workflow -----------------------------------------------------

class _FixedDetector:
    """Stands in for the neural detector so this test needs no model file."""

    def detect(self, rgb):
        from facevault.detect import Face

        return [Face(box, 0.99, None) for box in BOXES]


def test_full_workflow(image, tmp_path):
    keys = tmp_path / "keys"
    share_paths = pipeline.setup_keys(keys, threshold=2, officers=3)
    assert not list(keys.rglob("vault.key"))   # the private key is never stored whole
    with pytest.raises(FileExistsError):
        pipeline.setup_keys(keys)

    outcome = pipeline.protect_image(image, pipeline.read_key(keys / pipeline.VAULT_PUBLIC), _FixedDetector(), {"camera": "t"})
    outcome.matches = [{"label": "person-a", "score": 0.7, "box": list(BOXES[0])}]
    alert_path = pipeline.send_alert(pipeline.make_alert(outcome, "p.png"), keys, tmp_path / "outbox")
    assert pipeline.read_alert(alert_path, keys)["matches"][0]["label"] == "person-a"

    shares = [sharing.Share.from_json(path.read_text(encoding="utf-8")) for path in share_paths]
    audit = keys / pipeline.AUDIT_LOG
    with pytest.raises(ValueError):
        pipeline.reveal_image(outcome.protected, outcome.header, shares[:1], "curious", audit, "p.png")
    result = pipeline.reveal_image(outcome.protected, outcome.header, shares[1:], "case 42", audit, "p.png")
    assert np.array_equal(result.image, image)

    entries = AuditLog(audit).entries()
    assert [entry["action"] for entry in entries] == ["reveal-denied", "reveal"]
    assert entries[1]["actors"] == ["officer-2", "officer-3"]
    assert AuditLog(audit).verify() == (True, None)
