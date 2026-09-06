"""Shared infrastructure for virtual-pathway packages.

Pathways pin this package's major version. Split functions, featurisers and
metric definitions are defined here. A change to them changes what a recorded
number means. The core version is written into every version manifest's
``[protocol]`` table.
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

__version__ = "1.2.1"

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
