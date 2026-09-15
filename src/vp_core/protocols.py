"""Evaluation protocols.

A protocol fixes how a number was produced apart from the model and the data:
the split, the fold sizes, the seeds and the metric set. Two metric values may
be compared only when they carry the same protocol id.

Protocols are append-only. A change means a new revision —
``scaffold-shuffle-5seed@1`` becomes ``@2`` — and ``tests/test_protocols.py``
pins every definition to a fingerprint.

Changing the seed *set* is a protocol change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from vp_core.splits import scaffold_balanced_indices, scaffold_split_indices

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
    split: str  # "scaffold-shuffle" | "scaffold-sorted" | "scaffold-balanced"
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
        if self.split == "scaffold-balanced":
            return scaffold_balanced_indices(
                smiles, val_frac=self.val_frac, test_frac=self.test_frac, seed=seed
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
        """Stable hash of the definition, used to detect an in-place edit."""
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
            "so distinct seeds give distinct test sets. Five seeds; report mean "
            "and standard deviation over the held-out test folds."
        ),
    ),
    Protocol(
        id="scaffold-balanced-5seed@1",
        split="scaffold-balanced",
        val_frac=0.10,
        test_frac=0.15,
        seeds=(0, 1, 2, 3, 4),
        metrics=("auc_roc", "auprc", "mcc", "brier"),
        description=(
            "Bemis-Murcko scaffold split with scaffold groups permuted by seed "
            "and each group placed in the fold it overfills least, so a group "
            "larger than a fold's capacity settles in train instead of starving "
            "that fold. Same fold fractions, seeds and metrics as "
            "scaffold-shuffle-5seed@1; only the packing differs. Five seeds; "
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
            "Single-seed smoke protocol for tests and CI. Carries no variance "
            "estimate."
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
    """Raise unless two metrics may be compared. Call before any delta."""
    if not comparable(a, b):
        raise ValueError(
            f"metrics from {a!r} and {b!r} are not comparable: they were produced "
            "under different protocols. Re-evaluate one under the other's protocol."
        )
