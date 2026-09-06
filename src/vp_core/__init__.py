"""Shared infrastructure for virtual-pathway packages.

Every pathway depends on this and pins its major version, because the split
function, the featurisers and the metric definitions all live here — a change
to any of them changes what a recorded number means. That is why the core
version is part of every version manifest's ``[protocol]`` table.

Nothing pathway-specific and nothing proprietary belongs here.
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

__version__ = "1.0.0"

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
