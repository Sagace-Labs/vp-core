"""The dataset contract, the hash, and manifest validation."""

from __future__ import annotations

import pandas as pd
import pytest

from vp_core import dataset, manifest
from vp_core.docs import check_pathway_readme

GOOD = pd.DataFrame(
    {
        "inchikey": ["AAA", "BBB", "CCC", "DDD"],
        "smiles": ["CCO", "CCC", "c1ccccc1", "CCN"],
        "label": [1, 0, 1, 0],
    }
)


def test_valid_table_passes():
    assert dataset.validate_table(GOOD) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.drop(columns=["label"]), "missing required column"),
        (lambda d: pd.concat([d, d.head(1)]), "duplicate inchikey"),
        (lambda d: d.assign(label=[1, 1, 1, 1]), "single class"),
        (lambda d: d.assign(label=[0, 1, 2, 0]), "binary"),
    ],
)
def test_contract_violations_are_reported(mutate, expected):
    problems = dataset.validate_table(mutate(GOOD.copy()))
    assert any(expected in p for p in problems), problems


def test_hash_ignores_row_order_and_extra_columns():
    base = dataset.dataset_hash(GOOD)
    reordered = dataset.dataset_hash(GOOD.iloc[::-1].reset_index(drop=True))
    decorated = dataset.dataset_hash(GOOD.assign(potency_um=[1.0, 2.0, 3.0, 4.0]))
    assert base == reordered == decorated


def test_hash_moves_when_a_label_changes():
    changed = GOOD.assign(label=[0, 0, 1, 0])
    assert dataset.dataset_hash(changed) != dataset.dataset_hash(GOOD)


def test_example_fixture_keeps_both_classes():
    sample = dataset.stratified_example(GOOD, n=2, seed=0)
    assert set(sample["label"]) == {0, 1}


MINIMAL = {
    "schema": 1,
    "pathway": "demo",
    "version": "v1",
    "released": "2026-01-01",
    "reason": "test",
    "signature": {
        "version": 1,
        "inputs": ["smiles"],
        "outputs": [
            {
                "name": "demo_score",
                "dtype": "float32",
                "range": [0.0, 1.0],
                "semantics": "P(demo)",
                "missing": "NaN on unparseable input",
            }
        ],
    },
    "protocol": {"id": "smoke@1", "provider": "vp-core", "core_version": "1.0.0"},
    "provenance": {
        "repo": "Sagace-Labs/virtual-pathways",
        "commit": "abc1234",
        "python": "3.11.11",
        "rdkit": "2026.03.2",
    },
}


def test_minimal_training_free_manifest_is_valid():
    """A structural-alert pathway has neither dataset nor model, on purpose."""
    assert manifest.validate(MINIMAL) == []


def test_unknown_protocol_is_rejected():
    bad = {**MINIMAL, "protocol": {**MINIMAL["protocol"], "id": "invented@9"}}
    assert any("not in the vp-core registry" in p for p in manifest.validate(bad))


def test_dataset_without_model_is_rejected():
    bad = {**MINIMAL, "dataset": {"name": "x"}}
    assert any("both be present or both absent" in p for p in manifest.validate(bad))


def test_non_redistributable_dataset_may_not_declare_a_path():
    bad = {
        **MINIMAL,
        "dataset": {
            "name": "x", "source": "s", "retrieved": "2026-01-01", "licence": "None",
            "redistributable": False, "path": "data/x.parquet", "sha256": "0",
            "n_rows": 1, "base_rate": 0.5, "fetch": "cmd",
        },
        "model": {
            "family": "xgboost", "features": "morgan2_2048", "fit": "full",
            "weights": "w.joblib", "sha256": "0",
        },
    }
    assert any("only the" in p and "hash" in p for p in manifest.validate(bad))


def test_toml_round_trip(tmp_path):
    path = manifest.write(tmp_path / "manifest.toml", MINIMAL)
    assert manifest.read(path) == {k: v for k, v in MINIMAL.items() if v is not None}


def test_readme_lint_catches_the_two_prohibitions():
    body = "# vp-demo\n\n" + "\n".join(
        f"## {s}\n\ntext\n"
        for s in ["Install", "Use", "Current version", "Data", "Retrain", "Licence", "Cite"]
    )
    problems = check_pathway_readme(body, package="vp-demo")
    assert any("CARD.md" in p for p in problems)

    with_table = body.replace("## Data\n\ntext", "## Data\n\n| a | b |\n|---|---|")
    assert any("markdown table" in p for p in check_pathway_readme(with_table))

    too_long = body + "\nfiller" * 200
    assert any("line cap" in p for p in check_pathway_readme(too_long))
