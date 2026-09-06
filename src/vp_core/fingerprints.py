"""Stateless molecular featurisation.

Every featuriser here is stateless — nothing is fitted on the training fold —
so computing the whole matrix once and indexing it per fold is equivalent to
featurising each fold separately, and no split-order leakage is possible.

``rdkit_desc`` (and therefore ``combo3``) depends on the installed RDKit
version, which is why every version manifest records it under ``[provenance]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FINGERPRINTS", "featurize", "width"]

FINGERPRINTS: tuple[str, ...] = (
    "morgan2_2048",
    "morgan3_2048",
    "maccs",
    "rdkit_desc",
    "combo3",
)


def _morgan(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            X[i] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return X


def _maccs(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    X = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(mol)
        for bit in bv.GetOnBits():
            X[i, bit] = 1.0
    return X


def _rdkit_desc(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    names = [n for n, _ in Descriptors._descList]
    fns = [f for _, f in Descriptors._descList]
    X = np.zeros((len(smiles), len(names)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, fn in enumerate(fns):
            try:
                X[i, j] = float(fn(mol))
            except Exception:
                X[i, j] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def featurize(smiles: list[str], kind: str) -> np.ndarray:
    """Featurise ``smiles``. Unparseable inputs become an all-zero row."""
    if kind == "morgan2_2048":
        return _morgan(smiles, 2, 2048)
    if kind == "morgan3_2048":
        return _morgan(smiles, 3, 2048)
    if kind == "maccs":
        return _maccs(smiles)
    if kind == "rdkit_desc":
        return _rdkit_desc(smiles)
    if kind == "combo3":
        return np.hstack(
            [_morgan(smiles, 3, 2048), _maccs(smiles), _rdkit_desc(smiles)]
        ).astype(np.float32)
    raise ValueError(f"unknown fingerprint {kind!r}; known: {list(FINGERPRINTS)}")


def width(kind: str) -> int:
    """Column count for ``kind``, without featurising anything real."""
    return int(featurize(["C"], kind).shape[1])
