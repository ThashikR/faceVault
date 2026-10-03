"""Tamper-evident logs: who unlocked what, and which images the camera made.

Each entry stores the hash of the entry before it, so changing or deleting an
old entry breaks every hash after it. Entries can also be signed, so that
nobody without the signing key can add or rewrite one. Removing entries from
the END of a log still leaves a valid, shorter chain, so the latest hash
should also be kept somewhere the log's owner cannot edit.
"""

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import signing
from .vault import canonical_json

GENESIS = "0" * 64


def _entry_hash(entry):
    body = {name: value for name, value in entry.items() if name not in ("hash", "signature")}
    return hashlib.sha256(canonical_json(body)).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)

    def entries(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append(self, action, actors, target, reason="", signing_private=None):
        """Add one entry and return it. With `signing_private` the entry is signed."""
        existing = self.entries()
        entry = {
            "seq": len(existing) + 1,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "actors": list(actors),
            "target": target,
            "reason": reason,
            "prev": existing[-1]["hash"] if existing else GENESIS,
        }
        entry["hash"] = _entry_hash(entry)
        if signing_private is not None:
            signature = signing.sign(signing_private, bytes.fromhex(entry["hash"]))
            entry["signature"] = base64.b64encode(signature).decode("ascii")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self, signer_public=None):
        """Return (ok, first_bad_seq). first_bad_seq is None when the chain is intact.

        With `signer_public` every entry must also carry that signer's signature.
        """
        previous = GENESIS
        for position, entry in enumerate(self.entries(), start=1):
            if entry.get("seq") != position or entry.get("prev") != previous or entry.get("hash") != _entry_hash(entry):
                return False, position
            if signer_public is not None:
                try:
                    signature = base64.b64decode(entry["signature"])
                except (KeyError, ValueError):
                    return False, position
                if not signing.verify(signer_public, signature, bytes.fromhex(entry["hash"])):
                    return False, position
            previous = entry["hash"]
        return True, None
