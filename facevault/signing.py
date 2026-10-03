"""Ed25519 signatures: the camera signs, anyone with its public key can check."""

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_RAW = serialization.Encoding.Raw


def public_from_private(private_bytes):
    return Ed25519PrivateKey.from_private_bytes(private_bytes).public_key().public_bytes(_RAW, serialization.PublicFormat.Raw)


def sign(private_bytes, message):
    return Ed25519PrivateKey.from_private_bytes(private_bytes).sign(message)


def verify(public_bytes, signature, message):
    """True if `signature` on `message` was made by the holder of the matching private key."""
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True
