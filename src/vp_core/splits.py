"""Bemis-Murcko scaffold splitting.

Every compound sharing a Murcko scaffold goes in the same fold, so no scaffold
is divided across folds. Two orderings, not interchangeable:

``shuffle=True``
    Scaffold groups are permuted by the seed before greedy packing, so distinct
    seeds give distinct test sets and multi-seed spread is a real variance
    estimate. The default, and the only ordering the shipped protocols use.

``shuffle=False``
    Groups are packed largest-first, pinning the biggest scaffolds to train
    (the MoleculeNet convention). Harder, but seeds are correlated, so a
    multi-seed standard deviation understates the spread.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["murcko_scaffold", "scaffold_split_indices", "scaffold_train_val"]


def murcko_scaffold(smiles: str) -> str:
    """Canonical Murcko scaffold SMILES, or ``""`` when it cannot be derived."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)
    except Exception:
        return ""


def scaffold_split_indices(
    smiles: list[str],
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into (train, val, test) index arrays by Murcko scaffold.

    Raises when a fold comes out empty, which happens on small sets when one
    scaffold group overflows past the val capacity into test. Use a protocol
    whose seeds all yield valid folds.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    rng = np.random.default_rng(seed)
    items = list(groups.items())
    if shuffle:
        ordered = [items[k] for k in rng.permutation(len(items))]
    else:
        ordered = sorted(items, key=lambda kv: (-len(kv[1]), rng.random()))

    n = len(smiles)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    n_train = n - n_test - n_val

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, idxs in ordered:
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    for name, fold in (("train", train), ("val", val), ("test", test)):
        if not fold:
            raise ValueError(
                f"scaffold split (seed={seed}, shuffle={shuffle}) produced an empty "
                f"{name} fold on {n} compounds — this seed is unusable for this dataset"
            )

    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


def scaffold_train_val(
    smiles: list[str],
    val_frac: float,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-way scaffold split that covers every compound.

    Used for a deployment fit, where the validation fold exists only to stop
    boosting early and nothing may be discarded. Distinct from
    :func:`scaffold_split_indices`, which holds out a test fold.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    rng = np.random.default_rng(seed)
    items = list(groups.items())
    ordered = (
        [items[k] for k in rng.permutation(len(items))]
        if shuffle
        else sorted(items, key=lambda kv: (-len(kv[1]), rng.random()))
    )

    n_val = round(len(smiles) * val_frac)
    train: list[int] = []
    val: list[int] = []
    for _, idxs in ordered:
        if len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            train.extend(idxs)

    if not val or not train:
        raise ValueError(
            f"two-way scaffold split (seed={seed}, val_frac={val_frac}) left a fold "
            f"empty on {len(smiles)} compounds"
        )
    return np.array(sorted(train)), np.array(sorted(val))
