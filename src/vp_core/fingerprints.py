"""Stateless molecular featurisation.

Nothing is fitted on the training fold, so computing the whole matrix once and
indexing it per fold is equivalent to featurising each fold separately.

``rdkit_desc`` and ``physchem_ion``, and therefore the composites built on
them, depend on the installed RDKit version, which every version manifest
records under ``[provenance]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FINGERPRINTS", "featurize", "width"]

FINGERPRINTS: tuple[str, ...] = (
    "morgan2_2048",
    "morgan2_count_2048",
    "morgan3_2048",
    "maccs",
    "rdkit_desc",
    "physchem_ion",
    "combo3",
    "morgan2c_physchem_ion",
)

# Bulk properties, then shape and complexity.
_DESCRIPTORS: tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
    "MolMR", "RingCount", "NumHeteroatoms", "HeavyAtomCount",
    "NumSaturatedRings", "NumAliphaticRings", "LabuteASA", "BertzCT",
)

# Groups charged at physiological pH. Standardisation neutralises a molecule
# before it is featurised, so charge state survives only if counted explicitly.
_ACIDS: tuple[tuple[str, str], ...] = (
    ("carboxylic_acid", "[CX3](=O)[OX2H1,OX1H0-]"),
    ("sulfonic_acid", "[SX4](=O)(=O)[OX2H1,OX1H0-]"),
    ("phosphate", "[PX4](=O)([OX2H1,OX1H0-])[OX2H1,OX1H0-]"),
    ("tetrazole", "c1nnn[nH]1"),
    ("acyl_sulfonamide", "[SX4](=O)(=O)[NX3H1][CX3]=O"),
)
_BASES: tuple[tuple[str, str], ...] = (
    ("aliphatic_amine", "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[SX4])]"),
    ("amidine", "[NX3][CX3]=[NX2]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
)

IONISATION_COLUMNS: tuple[str, ...] = (
    *(name for name, _ in _ACIDS),
    *(name for name, _ in _BASES),
    "n_acidic",
    "n_basic",
    "net_charge",
)


def _morgan(
    smiles: list[str], radius: int, n_bits: int, *, counted: bool = False
) -> np.ndarray:
    """Morgan fingerprint. Counted keeps multiplicity, which encodes size."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if counted:
            X[i] = gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        else:
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


def _physchem_ion(smiles: list[str]) -> np.ndarray:
    """Bulk descriptors followed by :data:`IONISATION_COLUMNS`."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    fns = [dict(Descriptors._descList)[name] for name in _DESCRIPTORS]
    acids = [Chem.MolFromSmarts(sma) for _, sma in _ACIDS]
    bases = [Chem.MolFromSmarts(sma) for _, sma in _BASES]
    width = len(_DESCRIPTORS) + len(IONISATION_COLUMNS)

    X = np.zeros((len(smiles), width), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = []
        for fn in fns:
            try:
                row.append(float(fn(mol)))
            except Exception:
                row.append(0.0)
        acid_counts = [len(mol.GetSubstructMatches(p)) for p in acids]
        base_counts = [len(mol.GetSubstructMatches(p)) for p in bases]
        n_acidic, n_basic = float(sum(acid_counts)), float(sum(base_counts))
        X[i] = [*row, *acid_counts, *base_counts, n_acidic, n_basic, n_basic - n_acidic]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def featurize(smiles: list[str], kind: str) -> np.ndarray:
    """Featurise ``smiles``. Unparseable inputs become an all-zero row."""
    if kind == "morgan2_2048":
        return _morgan(smiles, 2, 2048)
    if kind == "morgan2_count_2048":
        return _morgan(smiles, 2, 2048, counted=True)
    if kind == "morgan3_2048":
        return _morgan(smiles, 3, 2048)
    if kind == "maccs":
        return _maccs(smiles)
    if kind == "rdkit_desc":
        return _rdkit_desc(smiles)
    if kind == "physchem_ion":
        return _physchem_ion(smiles)
    if kind == "combo3":
        return np.hstack(
            [_morgan(smiles, 3, 2048), _maccs(smiles), _rdkit_desc(smiles)]
        ).astype(np.float32)
    if kind == "morgan2c_physchem_ion":
        return np.hstack(
            [_morgan(smiles, 2, 2048, counted=True), _physchem_ion(smiles)]
        ).astype(np.float32)
    raise ValueError(f"unknown fingerprint {kind!r}; known: {list(FINGERPRINTS)}")


def width(kind: str) -> int:
    """Column count for ``kind``, without featurising anything real."""
    return int(featurize(["C"], kind).shape[1])
