"""Splits partition without leaking; metrics survive degenerate folds."""

from __future__ import annotations

import numpy as np
import pytest

from vp_core.metrics import aggregate, binary_metrics
from vp_core.splits import murcko_scaffold, scaffold_split_indices, scaffold_train_val

# Twelve distinct ring systems, two substituted variants each. All cyclic.
_RINGS = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccsc1", "c1ccoc1", "C1CCCCC1",
    "C1CCNCC1", "C1CCOCC1", "c1cc2ccccc2[nH]1", "c1ccc2c(c1)OCO2",
    "c1ccc(-c2ccccc2)cc1", "C1CCCC1",
]
SMILES = [tmpl for ring in _RINGS for tmpl in (ring, ring.replace("c1cc", "Cc1cc", 1)
                                               if ring.startswith("c1cc")
                                               else ring.replace("C1CC", "CC1CC", 1))]


def test_the_fixture_has_enough_distinct_scaffolds():
    """Guards the tests below: greedy packing needs many small groups."""
    scaffolds = {murcko_scaffold(s) for s in SMILES}
    assert "" not in scaffolds, "an acyclic molecule crept into the fixture"
    assert len(scaffolds) >= 10


def test_scaffold_split_partitions_without_overlap():
    train, val, test = scaffold_split_indices(SMILES, 0.25, 0.25, seed=0)
    combined = np.concatenate([train, val, test])
    assert sorted(combined.tolist()) == list(range(len(SMILES)))
    assert not (set(train) & set(val) & set(test))


def test_no_scaffold_spans_two_folds():
    train, val, test = scaffold_split_indices(SMILES, 0.25, 0.25, seed=1)
    folds = {"train": train, "val": val, "test": test}
    seen: dict[str, str] = {}
    for name, idx in folds.items():
        for i in idx:
            scaffold = murcko_scaffold(SMILES[i])
            assert seen.setdefault(scaffold, name) == name, (
                f"scaffold {scaffold!r} appears in both {seen[scaffold]} and {name}"
            )


def test_shuffle_gives_seed_distinct_test_sets():
    a = set(scaffold_split_indices(SMILES, 0.25, 0.25, seed=0)[2].tolist())
    b = set(scaffold_split_indices(SMILES, 0.25, 0.25, seed=3)[2].tolist())
    assert a != b, "shuffled scaffold splits must differ between seeds"


def test_train_val_split_covers_everything():
    train, val = scaffold_train_val(SMILES, 0.25, seed=0)
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(len(SMILES)))


def test_empty_fold_raises_rather_than_silently_shrinking():
    with pytest.raises(ValueError, match="empty"):
        scaffold_split_indices(SMILES[:3], 0.2, 0.2, seed=0)


def test_metrics_on_a_perfect_ranking():
    scored = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert scored["auc_roc"] == pytest.approx(1.0)
    assert scored["n"] == 4 and scored["pos_rate"] == pytest.approx(0.5)


def test_single_class_fold_yields_nan_not_an_exception():
    scored = binary_metrics([1, 1, 1], [0.4, 0.6, 0.8])
    assert np.isnan(scored["auc_roc"]) and np.isnan(scored["auprc"])
    assert np.isfinite(scored["brier"])


def test_aggregate_keeps_every_seed_and_ignores_nan_in_the_mean():
    per_seed = [{"auc_roc": 0.8}, {"auc_roc": 0.6}, {"auc_roc": float("nan")}]
    agg = aggregate(per_seed, ("auc_roc",))["auc_roc"]
    assert agg["mean"] == pytest.approx(0.7)
    assert agg["n_finite"] == 2
    assert len(agg["per_seed"]) == 3
