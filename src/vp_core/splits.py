"""Bemis-Murcko scaffold splitting.

A scaffold split places every compound sharing a Murcko scaffold in the same
fold, which is a harder and more realistic generalisation test than a random
split. No scaffold is ever divided across folds, so there is no structural
leakage between train and test.

Two orderings exist and they are not interchangeable:

``shuffle=True``
    Scaffold groups are permuted by the seed before greedy packing, so distinct
    seeds give genuinely distinct test sets. Multi-seed spread is then a real
    variance estimate. This is the default and the only ordering used by the
    shipped protocols.

``shuffle=False``
    Groups are packed largest-first, pinning the biggest scaffolds to train
    (the MoleculeNet convention). Harder, but seeds are correlated, so a
    multi-seed standard deviation under this ordering understates the true
    spread. Retained only to reproduce historical numbers.
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
    scaffold group overflows past the val capacity into test. That is a real
    failure of the seed for this dataset, not something to paper over — pick a
    protocol whose seeds all yield valid folds and record it.
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
    boosting early and nothing may be discarded. This is deliberately not the
    same call as :func:`scaffold_split_indices` — an evaluation split holds out
    a test fold and a deployment fit does not, and conflating them is how a
    model ends up reported on data it trained on.
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
