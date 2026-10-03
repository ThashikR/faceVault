"""Public-key key agreement (X25519 + HKDF-SHA256).

The sender only needs the recipient's PUBLIC key to produce a fresh 32-byte
encryption key; only the matching PRIVATE key can produce the same key again.
This is what lets a camera lock faces without being able to unlock them.
"""

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_BYTES = 32

_RAW = serialization.Encoding.Raw
_RAW_PUBLIC = serialization.PublicFormat.Raw
_RAW_PRIVATE = serialization.PrivateFormat.Raw


def _public_bytes(private):
    return private.public_key().public_bytes(_RAW, _RAW_PUBLIC)


def generate_keypair():
    """Return (private_bytes, public_bytes)."""
    private = X25519PrivateKey.generate()
    return private.private_bytes(_RAW, _RAW_PRIVATE, serialization.NoEncryption()), _public_bytes(private)


def public_from_private(private_bytes):
    return _public_bytes(X25519PrivateKey.from_private_bytes(private_bytes))


def _derive(shared, ephemeral_public, recipient_public, info):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=ephemeral_public + recipient_public, info=info)
    return hkdf.derive(shared)


def sender_key(recipient_public, info):
    """Make a one-time key for this recipient. Returns (ephemeral_public, key)."""
    ephemeral = X25519PrivateKey.generate()
    ephemeral_public = _public_bytes(ephemeral)
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient_public))
    return ephemeral_public, _derive(shared, ephemeral_public, recipient_public, info)


def recipient_key(recipient_private, ephemeral_public, info):
    """Rebuild the sender's one-time key from the recipient's private key."""
    private = X25519PrivateKey.from_private_bytes(recipient_private)
    shared = private.exchange(X25519PublicKey.from_public_bytes(ephemeral_public))
    return _derive(shared, ephemeral_public, _public_bytes(private), info)
