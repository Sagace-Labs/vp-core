"""Shared infrastructure for virtual-pathway packages.

Every pathway pins this package's major version: the split function, the
featurisers and the metric definitions live here, so a change to any of them
changes what a recorded number means. The core version is written into every
version manifest's ``[protocol]`` table.

Nothing pathway-specific belongs here.
"""

from __future__ import annotations

from vp_core import (
    card,
    dataset,
    docs,
    fingerprints,
    hashing,
    manifest,
    metrics,
    protocols,
    splits,
    standardise,
    xgb,
)
from vp_core.registry import Version, VersionedPathway

__version__ = "1.2.0"

__all__ = [
    "Version",
    "VersionedPathway",
    "__version__",
    "card",
    "dataset",
    "docs",
    "fingerprints",
    "hashing",
    "manifest",
    "metrics",
    "protocols",
    "splits",
    "standardise",
    "xgb",
]
