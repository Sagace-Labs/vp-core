"""The ``metrics.json`` file: one entry per protocol a version was measured under.

A protocol fixes how a number was produced, not what the model is, so measuring
a version under a second protocol does not make it a new version. Entries are
keyed by protocol id, which is what keeps two packings of the same split from
being read as one estimator.

Schema 1 wrote one flat record. It is still read and never rewritten in place,
because the version directory holding it is released.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "entries",
    "entry_for",
    "merge",
    "protocol_ids",
    "read",
    "write",
]

SCHEMA_VERSION = 2

# What a per-protocol entry carries. ``pathway`` and ``version`` describe the
# version, not the measurement, so they stay at the top level.
_ENTRY_KEYS = (
    "protocol_id",
    "dataset_sha256",
    "dataset",
    "evaluated",
    "n",
    "folds",
    "test",
    "additional_outputs",
)


def read(version_dir: str | Path) -> dict[str, Any] | None:
    """The raw record, or None when the version has not been evaluated."""
    path = Path(version_dir) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def entries(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Measurements keyed by protocol id, whichever schema wrote them."""
    if not record:
        return {}
    if record.get("schema", 1) >= SCHEMA_VERSION:
        return dict(record["protocols"])
    return {record["protocol_id"]: record}


def protocol_ids(
    record: dict[str, Any] | None, *, declared: str | None = None
) -> list[str]:
    """Protocol ids in a stable order, the manifest's declared one first."""
    ids = sorted(entries(record))
    if declared in ids:
        ids.remove(declared)
        ids.insert(0, declared)
    return ids


def entry_for(
    record: dict[str, Any] | None, protocol_id: str
) -> dict[str, Any] | None:
    return entries(record).get(protocol_id)


def merge(record: dict[str, Any] | None, entry: dict[str, Any]) -> dict[str, Any]:
    """Place ``entry`` under its protocol id, replacing that protocol's last run."""
    measured = {
        pid: {k: held[k] for k in _ENTRY_KEYS if k in held}
        for pid, held in entries(record).items()
    }
    measured[entry["protocol_id"]] = {k: entry[k] for k in _ENTRY_KEYS if k in entry}
    head = record or entry
    return {
        "schema": SCHEMA_VERSION,
        "pathway": entry.get("pathway", head.get("pathway")),
        "version": entry.get("version", head.get("version")),
        "protocols": {pid: measured[pid] for pid in sorted(measured)},
    }


def write(version_dir: str | Path, entry: dict[str, Any]) -> Path:
    """Merge one protocol's result into the version's metrics file."""
    version_dir = Path(version_dir)
    path = version_dir / "metrics.json"
    merged = merge(read(version_dir), entry)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path
