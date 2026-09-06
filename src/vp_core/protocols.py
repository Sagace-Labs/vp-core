"""Evaluation protocols — the identity that makes two metrics comparable.

A protocol fixes everything about how a number was produced *except* the model
and the data: the split, the fold sizes, the seeds, and the metric set. Two
metric values may be compared only when they carry the same protocol id.

Protocols are **append-only**. Editing one in place would silently invalidate
every metric already recorded against it, so a change means a new revision:
``scaffold-shuffle-5seed@1`` becomes ``@2``. ``tests/test_protocols_frozen.py``
pins every definition to a fingerprint so an in-place edit fails CI.

Changing the seed *set* is a protocol change like any other. A mean over seeds
(0, 1, 2, 3, 4) and a mean over (0, 2, 6, 8, 13) are different estimators, and
comparing them is exactly the mistake this module exists to prevent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from vp_core.splits import scaffold_split_indices

__all__ = [
    "PROTOCOLS",
    "Protocol",
    "all_ids",
    "comparable",
    "get",
    "require_comparable",
]


@dataclass(frozen=True)
class Protocol:
    """A frozen, citable evaluation recipe."""

    id: str
    split: str  # "scaffold-shuffle" | "scaffold-sorted"
    val_frac: float
    test_frac: float
    seeds: tuple[int, ...]
    metrics: tuple[str, ...]
    description: str

    def split_indices(
        self, smiles: list[str], seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split ``smiles`` into (train, val, test) index arrays for one seed."""
        if seed not in self.seeds:
            raise ValueError(
                f"seed {seed} is not part of protocol {self.id} (seeds={self.seeds})"
            )
        if self.split not in ("scaffold-shuffle", "scaffold-sorted"):
            raise ValueError(f"unknown split strategy: {self.split!r}")
        return scaffold_split_indices(
            smiles,
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            seed=seed,
            shuffle=self.split == "scaffold-shuffle",
        )

    def fingerprint(self) -> str:
        """Stable hash of the definition. Used to detect an in-place edit."""
        payload = "|".join(
            [
                self.id,
                self.split,
                f"{self.val_frac:.6f}",
                f"{self.test_frac:.6f}",
                ",".join(str(s) for s in self.seeds),
                ",".join(self.metrics),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


_DEFINITIONS: tuple[Protocol, ...] = (
    Protocol(
        id="scaffold-shuffle-5seed@1",
        split="scaffold-shuffle",
        val_frac=0.10,
        test_frac=0.15,
        seeds=(0, 1, 2, 3, 4),
        metrics=("auc_roc", "auprc", "mcc", "brier"),
        description=(
            "Bemis-Murcko scaffold split with scaffold groups permuted by seed, "
            "so distinct seeds give genuinely distinct test sets. Five seeds; "
            "report mean and standard deviation over the held-out test folds."
        ),
    ),
    Protocol(
        id="smoke@1",
        split="scaffold-shuffle",
        val_frac=0.10,
        test_frac=0.15,
        seeds=(0,),
        metrics=("auc_roc",),
        description=(
            "Single-seed smoke protocol for tests and CI. Never cite a smoke "
            "number: it carries no variance estimate."
        ),
    ),
)

PROTOCOLS: dict[str, Protocol] = {p.id: p for p in _DEFINITIONS}


def get(protocol_id: str) -> Protocol:
    """Look up a protocol, raising with the valid ids on a miss."""
    try:
        return PROTOCOLS[protocol_id]
    except KeyError:
        raise KeyError(
            f"unknown protocol {protocol_id!r}; known: {sorted(PROTOCOLS)}"
        ) from None


def all_ids() -> list[str]:
    return sorted(PROTOCOLS)


def comparable(a: str, b: str) -> bool:
    """True when two protocol ids describe the same estimator."""
    return a == b


def require_comparable(a: str, b: str) -> None:
    """Raise unless two metrics may be compared. Call this before any delta."""
    if not comparable(a, b):
        raise ValueError(
            f"metrics from {a!r} and {b!r} are not comparable — they were produced "
            "under different protocols. Re-evaluate one under the other's protocol "
            "instead of comparing them."
        )
