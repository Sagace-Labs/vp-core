"""Shared fitting and evaluation for fixed-recipe, multi-endpoint binary panels."""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from vp_core import dataset, fingerprints, hashing, manifest, protocols

DEFAULT_PROTOCOL = "scaffold-balanced-5seed@1"


def _rows(table, label: str) -> np.ndarray:
    return np.flatnonzero(table[label].notna().to_numpy())


def _matrices(smiles: list[str], model_mod, outputs: list[str]) -> dict[str, np.ndarray]:
    kinds = {model_mod.features_for(output) for output in outputs}
    return {kind: fingerprints.featurize(smiles, kind) for kind in kinds}


def predict_values(fitted: dict, smiles: list[str], version, model_mod) -> np.ndarray:
    """Score named outputs, preserving invalid-input rows as NaN."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    matrices = {
        kind: fingerprints.featurize(smiles, kind)
        for kind in {version.features_for(name) for name in version.output_names}
    }
    values = np.column_stack([
        model_mod.predict_proba(fitted[name], matrices[version.features_for(name)])
        for name in version.output_names
    ]).astype(np.float32)
    values[np.array([Chem.MolFromSmiles(s) is None for s in smiles])] = np.nan
    return values


def build_version(
    package: str,
    versions_dir: Path,
    contract,
    data_mod,
    model_mod,
    version: str,
    *,
    reason: str,
    protocol_id: str = DEFAULT_PROTOCOL,
    seed: int = 0,
) -> Path:
    """Fit all labeled rows and create one immutable version directory."""
    import joblib
    import rdkit
    import sklearn
    import xgboost

    import vp_core

    protocols.get(protocol_id)
    directory = versions_dir / version
    if directory.exists():
        raise FileExistsError(f"{directory} already exists; release a new version")
    table = data_mod.load()
    outputs = contract.column_names()
    smiles = table.smiles.tolist()
    matrices = _matrices(smiles, model_mod, outputs)
    fitted = {}
    for output in outputs:
        label = data_mod.OUTPUT_LABELS[output]
        rows = _rows(table, label)
        fitted[output] = model_mod.fit(
            matrices[model_mod.features_for(output)][rows],
            table[label].to_numpy()[rows].astype(int),
            output=output,
            seed=seed,
        )
    directory.mkdir(parents=True)
    weights = directory / "weights.joblib"
    joblib.dump(fitted, weights)
    labels = list(data_mod.LABELS)
    record = {
        "schema": manifest.SCHEMA_VERSION,
        "pathway": package,
        "version": version,
        "released": date.today().isoformat(),
        "reason": reason,
        "signature": contract.as_manifest_table(),
        "dataset": {
            **data_mod.MANIFEST_DATA,
            "labels": labels,
            "sha256": dataset.dataset_hash(table, labels=labels),
            "n_rows": len(table),
            "base_rate": round(float(table[labels[0]].mean()), 6),
        },
        "model": {
            "family": model_mod.FAMILY,
            "features": model_mod.FEATURES,
            "fit": "one fixed-recipe estimator per output on all labeled compounds",
            "weights": weights.name,
            "sha256": hashing.sha256_file(weights),
            "features_by_output": [
                {"output": output, "kind": model_mod.features_for(output)}
                for output in outputs
                if model_mod.features_for(output) != model_mod.FEATURES
            ],
        },
        "protocol": {"id": protocol_id, "provider": "vp-core", "core_version": vp_core.__version__},
        "provenance": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    problems = manifest.validate(record, version_dir=directory)
    if problems:
        weights.unlink()
        directory.rmdir()
        raise ValueError("invalid manifest: " + "; ".join(problems))
    manifest.write(directory / "manifest.toml", record)
    return directory


def evaluate_version(
    package: str,
    get_version,
    data_mod,
    model_mod,
    version: str,
    *,
    protocol_id: str | None = None,
    use_example: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Refit per seed, score held-out folds, and generate the release card."""
    from vp_core import card, metrics, metrics_store

    resolved = get_version(version)
    protocol_id = protocol_id or resolved.protocol_id
    protocol = protocols.get(protocol_id)
    if use_example and write:
        raise ValueError("refusing to record fixture metrics as a released result")
    table = data_mod.example() if use_example else data_mod.load()
    labels = list(data_mod.LABELS)
    smiles = table.smiles.tolist()
    matrices = _matrices(smiles, model_mod, resolved.output_names)
    scored = {}
    for output in resolved.output_names:
        label = data_mod.OUTPUT_LABELS[output]
        y = table[label].to_numpy()
        keep = set(_rows(table, label).tolist())
        X = matrices[resolved.features_for(output)]
        per_seed = []
        folds = []
        for seed in protocol.seeds:
            tr, va, te = protocol.split_indices(smiles, seed)
            train = np.array([i for i in tr if i in keep], dtype=int)
            val = np.array([i for i in va if i in keep], dtype=int)
            test = np.array([i for i in te if i in keep], dtype=int)
            if min(len(train), len(val), len(test)) == 0:
                raise ValueError(f"{output} produced an empty fold for seed {seed}")
            fitted = model_mod.fit(X[train], y[train].astype(int), output=output, seed=seed)
            probability = model_mod.predict_proba(fitted, X[test])
            per_seed.append(metrics.binary_metrics(y[test].astype(int), probability))
            folds.append({"seed": seed, "train": len(train), "val": len(val), "test": len(test)})
        scored[output] = {
            "n": {"total": len(keep), **{k: folds[0][k] for k in ("train", "val", "test")}},
            "folds": folds,
            "test": metrics.aggregate(per_seed, protocol.metrics),
        }
    primary = scored[resolved.output_names[0]]
    record = {
        "pathway": package,
        "version": resolved.name,
        "protocol_id": protocol_id,
        "dataset_sha256": dataset.dataset_hash(table, labels=labels),
        "dataset": "example fixture" if use_example else "full",
        "evaluated": date.today().isoformat(),
        "n": primary["n"],
        "folds": primary["folds"],
        "test": primary["test"],
        "additional_outputs": [
            {"name": name, **scored[name]} for name in resolved.output_names[1:]
        ],
    }
    if write:
        metrics_store.write(resolved.directory, record)
        card.write_card(resolved.directory)
    return record
