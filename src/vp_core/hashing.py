"""Content hashes for files.

Dataset tables are hashed by :func:`vp_core.dataset.dataset_hash`, which
canonicalises before hashing. Everything else — weights, in particular — is
hashed byte-for-byte here, because a weights file is opaque and any change to
it is a change to the version.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["sha256_bytes", "sha256_file"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()
