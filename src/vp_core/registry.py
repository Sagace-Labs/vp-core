"""Version discovery and loading.

List the versions, load one, predict with it.

Two rules are enforced at load time: predictions come back as a DataFrame whose
columns are the manifest's declared outputs, and a version directory is usable
only if its manifest validates.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vp_core import manifest as manifest_mod

__all__ = ["Version", "VersionedPathway", "parse_version"]

_VERSION_RE = re.compile(r"^v(\d+)(?:\.(\d+))?(?:\.(\d+))?$")


def parse_version(name: str) -> tuple[int, int, int]:
    """``"v2"`` → ``(2, 0, 0)``. Raises on anything that is not a version dir."""
    match = _VERSION_RE.match(name)
    if not match:
        raise ValueError(f"{name!r} is not a version directory name (expected v1, v1.2, v1.2.3)")
    major, minor, patch = match.groups()
    return int(major), int(minor or 0), int(patch or 0)


@dataclass(frozen=True)
class Version:
    """One immutable released version."""

    pathway: str
    name: str
    directory: Path
    manifest: dict[str, Any]

    @property
    def signature_version(self) -> int:
        return int(self.manifest["signature"]["version"])

    @property
    def protocol_id(self) -> str:
        return str(self.manifest["protocol"]["id"])

    @property
    def output_names(self) -> list[str]:
        return manifest_mod.output_names(self.manifest)

    @property
    def features(self) -> str | None:
        model = self.manifest.get("model")
        return str(model["features"]) if model else None

    def metrics(self) -> dict[str, Any] | None:
        path = self.directory / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def problems(self) -> list[str]:
        return manifest_mod.validate(self.manifest, version_dir=self.directory)

    def load_model(self) -> Any:
        import joblib

        model = self.manifest.get("model")
        if not model:
            raise ValueError(
                f"{self.pathway} {self.name} declares no model — it is training-free"
            )
        return joblib.load(self.directory / str(model["weights"]))

    def frame(self, values: np.ndarray) -> pd.DataFrame:
        """Wrap a raw ``(n, k)`` array in the declared output columns."""
        names = self.output_names
        values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
        if values.shape[1] != len(names):
            raise ValueError(
                f"{self.pathway} {self.name} produced {values.shape[1]} columns but its "
                f"signature declares {len(names)}: {names}"
            )
        return pd.DataFrame(values, columns=pd.Index(names))


class VersionedPathway:
    """The loader behind a pathway's ``predict`` / ``load`` / ``versions``."""

    def __init__(
        self,
        name: str,
        versions_dir: str | Path,
        *,
        predict_fn: Callable[[Any, list[str], Version], np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self.versions_dir = Path(versions_dir)
        self._predict_fn = predict_fn or _default_predict

    def versions(self) -> list[str]:
        """Released version names, oldest first."""
        if not self.versions_dir.is_dir():
            return []
        found = []
        for child in self.versions_dir.iterdir():
            if child.is_dir() and (child / "manifest.toml").exists():
                try:
                    found.append((parse_version(child.name), child.name))
                except ValueError:
                    continue
        return [name for _, name in sorted(found)]

    def current(self) -> str:
        available = self.versions()
        if not available:
            raise FileNotFoundError(f"no released versions under {self.versions_dir}")
        return available[-1]

    def get(self, version: str | None = None) -> Version:
        """Load a version (default: the newest), validating its manifest."""
        return self._get(version or self.current())

    @lru_cache(maxsize=8)  # noqa: B019 - bounded, keyed on a version name
    def _get(self, version: str) -> Version:
        directory = self.versions_dir / version
        path = directory / "manifest.toml"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name} has no version {version!r}; available: {self.versions()}"
            )
        loaded = manifest_mod.read(path)
        problems = manifest_mod.validate(loaded, version_dir=directory)
        if problems:
            raise ValueError(
                f"{self.name} {version} has an invalid manifest: {'; '.join(problems)}"
            )
        return Version(self.name, version, directory, loaded)

    def predict(self, smiles: list[str], *, version: str | None = None) -> pd.DataFrame:
        """Predict for ``smiles``, returning the declared output columns."""
        if isinstance(smiles, str):
            raise TypeError("pass a list of SMILES, not a single string")
        resolved = self.get(version)
        values = self._predict_fn(resolved.load_model(), list(smiles), resolved)
        return resolved.frame(values)


def _default_predict(model: Any, smiles: list[str], version: Version) -> np.ndarray:
    """XGBoost on a vp-core fingerprint."""
    from rdkit import Chem, RDLogger

    from vp_core import fingerprints, xgb

    # An unparseable input is a declared NaN.
    RDLogger.DisableLog("rdApp.*")

    features = version.features
    if features is None:
        raise ValueError(f"{version.pathway} {version.name} declares no feature kind")
    X = fingerprints.featurize(smiles, features)
    proba = xgb.predict_proba(model, X).astype(np.float32)
    unparseable = np.array([Chem.MolFromSmiles(s) is None for s in smiles])
    proba[unparseable] = np.nan
    return proba.reshape(-1, 1)
