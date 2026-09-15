"""Bemis-Murcko scaffold splitting.

Every compound sharing a Murcko scaffold goes in the same fold. Three packers:

:func:`scaffold_split_indices` with ``shuffle=True``
    Scaffold groups are permuted by the seed, then offered to train, val and
    test in that fixed order. A group larger than the val capacity can only
    land in train or test, so val can be starved on a library that holds a few
    dominant groups.

:func:`scaffold_split_indices` with ``shuffle=False``
    Groups are packed largest-first, pinning the biggest scaffolds to train
    (the MoleculeNet convention). Harder, but seeds are correlated, so a
    multi-seed standard deviation understates the spread.

:func:`scaffold_balanced_indices`
    Groups are permuted by the seed, then each goes to the fold whose filled
    share it would leave lowest. An oversized group settles in train, the only
    fold it does not overfill.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = [
    "murcko_scaffold",
    "scaffold_balanced_indices",
    "scaffold_split_indices",
    "scaffold_train_val",
]


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

    Raises when a fold comes out empty.
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

    capacity = {"train": n_train, "val": n_val, "test": n_test}
    for name, fold in (("train", train), ("val", val), ("test", test)):
        if not fold:
            largest = ", ".join(
                str(s)
                for s in sorted((len(g) for g in groups.values()), reverse=True)[:3]
            )
            raise ValueError(
                f"scaffold split (seed={seed}, shuffle={shuffle}) produced an empty "
                f"{name} fold on {n} compounds in {len(groups)} scaffold groups. "
                f"Groups are packed into train first, so a group larger than the "
                f"{name} capacity of {capacity[name]} never lands there, and the "
                f"largest groups here hold {largest}. This seed is unusable for "
                f"this dataset, and a protocol naming it cannot measure it."
            )

    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


_FOLDS: tuple[str, str, str] = ("train", "val", "test")


def scaffold_balanced_indices(
    smiles: list[str],
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into (train, val, test) index arrays by Murcko scaffold.

    Groups are permuted by the seed. Each group then goes to the fold whose filled share it would leave lowest, which settles a group too large for val or test into train.

    A fold still left empty takes the smallest group from the fold holding the most.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    if len(groups) < len(_FOLDS):
        raise ValueError(
            f"{len(smiles)} compounds form only {len(groups)} scaffold "
            f"group(s); {len(_FOLDS)} folds cannot each hold one"
        )

    n = len(smiles)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    target = {
        "train": max(1, n - n_test - n_val),
        "val": max(1, n_val),
        "test": max(1, n_test),
    }

    rng = np.random.default_rng(seed)
    items = list(groups.items())
    ordered = [items[k] for k in rng.permutation(len(items))]

    held: dict[str, list[list[int]]] = {name: [] for name in _FOLDS}
    size = dict.fromkeys(_FOLDS, 0)
    for _, idxs in ordered:
        chosen, best = _FOLDS[0], None
        for name in _FOLDS:
            # Filled share the group would leave behind, so a group too large
            # for a small fold lands in the one it distorts least.
            cost = ((size[name] + len(idxs)) / target[name], -target[name])
            if best is None or cost < best:
                chosen, best = name, cost
        held[chosen].append(idxs)
        size[chosen] += len(idxs)

    for name in _FOLDS:
        if held[name]:
            continue
        donor = max(_FOLDS, key=lambda f: len(held[f]))
        counts = [len(group) for group in held[donor]]
        moved = held[donor].pop(counts.index(min(counts)))
        held[name].append(moved)
        size[donor] -= len(moved)
        size[name] += len(moved)

    folds = [
        np.array(sorted(i for group in held[name] for i in group)) for name in _FOLDS
    ]
    return folds[0], folds[1], folds[2]


def scaffold_train_val(
    smiles: list[str],
    val_frac: float,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-way scaffold split that covers every compound.

    Used for a deployment fit, where the validation fold exists only to stop
    boosting early and nothing may be discarded.
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
