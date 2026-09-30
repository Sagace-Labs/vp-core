"""Site-table hashes bind atom indices and provenance to the release."""

from __future__ import annotations

import pandas as pd
import pytest

from vp_core.dataset import read_site_table, site_dataset_hash


def test_hash_covers_atom_index_and_source():
    table = pd.DataFrame([
        ("source-a", "one", "CCO", "[1]"),
        ("source-b", "two", "CCN", "[2]"),
    ], columns=["source", "identifier", "smiles", "sites"])
    original = site_dataset_hash(table)
    assert site_dataset_hash(table.iloc[::-1].reset_index(drop=True)) != original
    changed_site = table.copy()
    changed_site.loc[0, "sites"] = "[0]"
    assert site_dataset_hash(changed_site) != original
    changed_source = table.copy()
    changed_source.loc[0, "source"] = "source-c"
    assert site_dataset_hash(changed_source) != original


def test_atom_indices_are_validated(tmp_path):
    path = tmp_path / "sites.parquet"
    pd.DataFrame([("source-a", "one", "CCO", "[3]")],
                 columns=["source", "identifier", "smiles", "sites"]).to_parquet(path)
    with pytest.raises(ValueError, match="invalid atom index"):
        read_site_table(path)
