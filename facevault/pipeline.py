"""The whole workflow in four steps: set up keys, protect, alert, reveal."""

import base64
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import alerts, hybrid, sharing, signing, vault
from .audit import AuditLog

VAULT_PUBLIC = "vault.pub"
CAMERA_KEY = "camera-signing.key"
CAMERA_PUBLIC = "camera-signing.pub"
RECIPIENT_KEY = "alert-recipient.key"
RECIPIENT_PUBLIC = "alert-recipient.pub"
SHARES_DIR = "shares"
AUDIT_LOG = "audit.jsonl"
CAPTURE_LOG = "capture-log.jsonl"
DELETION_LOG = "deletion-log.jsonl"
QUORUM_PUBLIC = "officers-signing.pub"


def write_key(path, data):
    Path(path).write_text(base64.b64encode(data).decode("ascii") + "\n", encoding="ascii")


def read_key(path):
    return base64.b64decode(Path(path).read_text(encoding="ascii").strip())


def officers_signing_key(vault_private):
    """A signing key that exists only while the officers' shares are combined.

    It is derived from the vault's private key, so a record signed with it
    proves that enough officers acted together. Its public half is saved at
    setup for anyone to check such records.
    """
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"facevault/v1/officers-signing").derive(vault_private)


def setup_keys(directory, threshold=2, officers=3):
    """Create every key the system needs. Returns the list of share files written.

    The vault's private key is split into shares and then thrown away: after
    this call it exists nowhere in one piece.
    """
    directory = Path(directory)
    shares_dir = directory / SHARES_DIR
    if (directory / VAULT_PUBLIC).exists():
        raise FileExistsError(f"{directory} already holds a vault; refusing to overwrite its keys")
    shares_dir.mkdir(parents=True, exist_ok=True)

    vault_private, vault_public = hybrid.generate_keypair()
    write_key(directory / VAULT_PUBLIC, vault_public)
    write_key(directory / QUORUM_PUBLIC, signing.public_from_private(officers_signing_key(vault_private)))
    paths = []
    for share in sharing.split(vault_private, threshold, officers):
        path = shares_dir / f"officer-{share.index}.json"
        path.write_text(share.to_json(), encoding="utf-8")
        paths.append(path)

    camera_private, camera_public = alerts.generate_signing_keypair()
    write_key(directory / CAMERA_KEY, camera_private)
    write_key(directory / CAMERA_PUBLIC, camera_public)

    recipient_private, recipient_public = hybrid.generate_keypair()
    write_key(directory / RECIPIENT_KEY, recipient_private)
    write_key(directory / RECIPIENT_PUBLIC, recipient_public)
    return paths


@dataclass
class ProtectOutcome:
    protected: object
    header: dict
    faces: list
    matches: list = field(default_factory=list)   # [{"label", "score", "box"}]


def protect_image(rgb, vault_public, detector, context=None, watchlist=None, signing_private=None):
    """Find the faces, check them against the watch-list, then lock them.

    With the camera's `signing_private` key the protected image is signed.
    """
    faces = detector.detect(rgb)
    matches = []
    if watchlist is not None:
        for face in faces:
            hit = watchlist.match(rgb, face)
            if hit:
                matches.append({"label": hit[0], "score": round(hit[1], 4), "box": list(face.box)})
    protected, header = vault.protect(rgb, [face.box for face in faces], vault_public, context, signing_private=signing_private)
    return ProtectOutcome(protected, header, faces, matches)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_capture(keys_dir, image_path, camera):
    """Add a saved protected image to the camera's signed, chained capture log.

    The log is what makes a later deletion or swap of the file show up.
    """
    keys_dir = Path(keys_dir)
    log = AuditLog(keys_dir / CAPTURE_LOG)
    return log.append("capture", [camera], Path(image_path).name, "sha256:" + file_digest(image_path),
                      signing_private=read_key(keys_dir / CAMERA_KEY))


def check_captures(keys_dir, folder):
    """Compare the capture log with the files in `folder`.

    Returns (log_ok, rows): log_ok is False if the log itself was edited or
    holds an entry the camera did not sign; rows are (entry, status, deletion)
    for the newest entry of each file. status is "ok", "altered", "deleted"
    (removed by the officers together; `deletion` is their record) or
    "missing" (removed by someone else).
    """
    keys_dir, folder = Path(keys_dir), Path(folder)
    log = AuditLog(keys_dir / CAPTURE_LOG)
    log_ok, _ = log.verify(read_key(keys_dir / CAMERA_PUBLIC))
    newest = {entry["target"]: entry for entry in log.entries()}

    # Deletion records count only if the whole deletion log is signed by the officers' key.
    deletions = {}
    deletion_log = AuditLog(keys_dir / DELETION_LOG)
    if (keys_dir / QUORUM_PUBLIC).exists() and deletion_log.verify(read_key(keys_dir / QUORUM_PUBLIC))[0]:
        deletions = {(record["target"], record["reason"].split(" | ")[0]): record for record in deletion_log.entries()}

    rows = []
    for entry in newest.values():
        path = folder / entry["target"]
        deletion = deletions.get((entry["target"], entry["reason"]))
        if not path.exists() and deletion is not None:
            rows.append((entry, "deleted", deletion))
            continue
        if not path.exists():
            status = "missing"
        elif "sha256:" + file_digest(path) != entry["reason"]:
            status = "altered"
        else:
            status = "ok"
        rows.append((entry, status, None))
    return log_ok, rows


def delete_image(image_path, shares, reason, keys_dir):
    """Delete a protected image with the agreement of ALL the officers, and record it.

    Viewing needs k officers; deleting evidence is more serious, so it needs
    every one of them. The record is signed with a key only they can rebuild
    together. This cannot stop someone deleting the file by other means; it
    lets `check_captures` tell an agreed deletion from an unexplained one.
    """
    keys_dir, image_path = Path(keys_dir), Path(image_path)
    audit = AuditLog(keys_dir / AUDIT_LOG)
    actors = sorted({f"officer-{share.index}" for share in shares})
    try:
        if not shares or len(actors) != shares[0].n:
            raise ValueError(f"deleting needs all {shares[0].n if shares else 'the'} officers, got {len(actors)}")
        # Every officer's share must be genuine: each group of k must rebuild the same key.
        keys = {sharing.combine(list(group)) for group in itertools.combinations(shares, shares[0].k)}
        if len(keys) != 1:
            raise ValueError("the shares do not agree")
        signing_private = officers_signing_key(keys.pop())
        if signing.public_from_private(signing_private) != read_key(keys_dir / QUORUM_PUBLIC):
            raise ValueError("these shares belong to a different vault")
        if not image_path.exists():
            raise ValueError(f"{image_path.name} does not exist")
    except (ValueError, OSError) as error:
        audit.append("delete-denied", actors, image_path.name, f"{reason} [{error}]")
        raise ValueError(str(error)) from error

    record = AuditLog(keys_dir / DELETION_LOG).append(
        "delete", actors, image_path.name, f"sha256:{file_digest(image_path)} | {reason}", signing_private=signing_private)
    image_path.unlink()
    audit.append("delete", actors, image_path.name, reason)
    return record


def make_alert(outcome, protected_name):
    """The message sent when a watch-list face is seen. It names the file, not the pixels."""
    return {
        "event": "watchlist-match",
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": outcome.header["context"],
        "matches": outcome.matches,
        "protected_image": protected_name,
    }


def send_alert(alert, keys_dir, outbox):
    """Seal the alert and drop it in the outbox folder. Returns the file written."""
    keys_dir, outbox = Path(keys_dir), Path(outbox)
    envelope = alerts.seal_alert(alert, read_key(keys_dir / RECIPIENT_PUBLIC), read_key(keys_dir / CAMERA_KEY))
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = outbox / f"{stamp}.alert.json"
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


def read_alert(path, keys_dir):
    keys_dir = Path(keys_dir)
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    return alerts.open_alert(envelope, read_key(keys_dir / RECIPIENT_KEY), read_key(keys_dir / CAMERA_PUBLIC))


def reveal_image(protected, header, shares, reason, audit_path, target, trusted_camera=None):
    """Rebuild the vault key from the officers' shares, unlock the faces, and log it.

    Failed attempts are logged too. The rebuilt key lives only inside this call.
    With `trusted_camera` the image must carry that camera's signature.
    """
    log = AuditLog(audit_path)
    actors = sorted(f"officer-{share.index}" for share in shares)
    try:
        result = vault.reveal(protected, header, sharing.combine(shares), trusted_camera)
    except (ValueError, vault.TamperedError) as error:
        log.append("reveal-denied", actors, target, f"{reason} [{error}]")
        raise
    log.append("reveal", actors, target, reason)
    return result
