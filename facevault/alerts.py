"""Encrypted, signed alerts.

An alert is sealed so only the intended recipient can read it
(X25519 + ChaCha20-Poly1305) and signed by the camera (Ed25519) so the
recipient knows where it came from and that nobody changed it.
"""

import base64
import json
import os

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from . import hybrid
from .vault import canonical_json

VERSION = 1
KIND = "facevault-alert"
_INFO = b"facevault/v1/alert"

_RAW = serialization.Encoding.Raw


class AlertError(Exception):
    """The alert is forged, damaged, from an untrusted sender, or not for this recipient."""


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _unb64(text):
    return base64.b64decode(text.encode("ascii"))


def generate_signing_keypair():
    """Return (private_bytes, public_bytes) for the device that sends alerts."""
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(_RAW, serialization.PrivateFormat.Raw, serialization.NoEncryption()),
        private.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw),
    )


def seal_alert(alert, recipient_public, signing_private):
    """Encrypt `alert` (a dict) to the recipient and sign the result. Returns an envelope dict."""
    signer = Ed25519PrivateKey.from_private_bytes(signing_private)
    sender_public = signer.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw)
    ephemeral_public, key = hybrid.sender_key(recipient_public, _INFO)
    nonce = os.urandom(12)
    # Binding both identities stops someone re-signing a stolen ciphertext as their own.
    aad = _INFO + sender_public + recipient_public
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, canonical_json(alert), aad)
    envelope = {
        "v": VERSION,
        "type": KIND,
        "sender": _b64(sender_public),
        "eph": _b64(ephemeral_public),
        "nonce": _b64(nonce),
        "ciphertext": _b64(ciphertext),
    }
    envelope["signature"] = _b64(signer.sign(canonical_json(envelope)))
    return envelope


def open_alert(envelope, recipient_private, trusted_sender_public):
    """Check the signature against the sender we trust, then decrypt. Returns the alert dict."""
    try:
        unsigned = {name: envelope[name] for name in ("v", "type", "sender", "eph", "nonce", "ciphertext")}
        signature = _unb64(envelope["signature"])
        if unsigned["v"] != VERSION or unsigned["type"] != KIND:
            raise AlertError("not a FaceVault alert")
        if _unb64(unsigned["sender"]) != trusted_sender_public:
            raise AlertError("alert is not from the trusted sender")
        Ed25519PublicKey.from_public_bytes(trusted_sender_public).verify(signature, canonical_json(unsigned))
        key = hybrid.recipient_key(recipient_private, _unb64(unsigned["eph"]), _INFO)
        aad = _INFO + trusted_sender_public + hybrid.public_from_private(recipient_private)
        plain = ChaCha20Poly1305(key).decrypt(_unb64(unsigned["nonce"]), _unb64(unsigned["ciphertext"]), aad)
    except InvalidSignature as error:
        raise AlertError("signature check failed") from error
    except InvalidTag as error:
        raise AlertError("alert cannot be decrypted with this key") from error
    except (KeyError, TypeError, ValueError) as error:
        raise AlertError(f"unreadable alert: {error}") from error
    return json.loads(plain)
