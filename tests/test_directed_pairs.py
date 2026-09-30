"""Directed-pair releases use a table hash without a binary base rate."""

from vp_core import card, manifest


def test_directed_pair_manifest_and_card():
    record = {
        "schema": manifest.SCHEMA_VERSION,
        "pathway": "metabolite",
        "version": "v1",
        "released": "2026-09-30",
        "reason": "Initial release",
        "signature": {
            "version": 1,
            "inputs": ["smiles"],
            "outputs": [
                {
                    "name": "metabolite_score",
                    "dtype": "float32",
                    "range": [0.0, 1.0],
                    "semantics": "ranking score",
                    "missing": "no rows",
                }
            ],
        },
        "dataset": {
            "name": "Directed pairs",
            "source": "frozen source assertions",
            "retrieved": "2026-09-30",
            "licence": "CC-BY-SA-4.0",
            "redistributable": True,
            "path": "data/pairs.csv",
            "format": "directed-pairs@1",
            "unit": "directed pairs",
            "sha256": "0" * 64,
            "n_rows": 10,
            "fetch": "python -m example.data verify",
        },
        "model": {
            "family": "forest",
            "features": "route and structure",
            "fit": "all pairs",
            "weights": "weights.joblib",
            "sha256": "0" * 64,
        },
        "protocol": {
            "id": "metabolite-v1-fixed-similarity@1",
            "provider": "vp-core",
            "core_version": "1.7.0",
        },
        "provenance": {"python": "3.12", "rdkit": "2026.03.6"},
    }
    assert manifest.validate(record) == []
    rendered = card.render_card(record, None)
    assert "10` directed pairs" in rendered
    assert "positive rate" not in rendered
