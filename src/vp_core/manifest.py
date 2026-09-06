"""The version manifest: schema, reader, writer and validator.

One ``manifest.toml`` per released version, written by a pathway's ``train``
command and never by hand. TOML rather than JSON because this is the document a
human reads in a review diff; the numbers live in ``metrics.json`` beside it,
which tooling consumes and which carries per-seed arrays.

A pathway with no trained model omits the ``[dataset]`` and ``[model]``
tables, and the version is then the rule set itself. The validator accepts
that shape.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

__all__ = [
    "DTYPES",
    "SCHEMA_VERSION",
    "dumps",
    "output_names",
    "read",
    "validate",
    "write",
]

SCHEMA_VERSION = 1

DTYPES: tuple[str, ...] = ("float32", "float64", "int8", "int32", "bool")

_TOP_REQUIRED = ("schema", "pathway", "version", "released", "reason")
_DATASET_REQUIRED = (
    "name",
    "source",
    "retrieved",
    "licence",
    "redistributable",
    "sha256",
    "n_rows",
    "base_rate",
    "fetch",
)
_MODEL_REQUIRED = ("family", "features", "fit", "weights", "sha256")
_PROTOCOL_REQUIRED = ("id", "provider", "core_version")
_PROVENANCE_REQUIRED = ("python", "rdkit")

# Key order used when writing, so manifests diff cleanly against each other.
_ORDER = ("schema", "pathway", "version", "released", "supersedes", "reason")
_TABLE_ORDER = ("signature", "dataset", "model", "protocol", "provenance")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 10))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {type(value).__name__} to TOML: {value!r}")


def dumps(data: dict[str, Any]) -> str:
    """Serialise a manifest dict to TOML with a stable key order."""
    lines: list[str] = []

    for key in _ORDER:
        if key in data and data[key] is not None:
            lines.append(f"{key} = {_fmt(data[key])}")
    for key, value in data.items():
        if key in _ORDER or isinstance(value, dict) or key in _TABLE_ORDER:
            continue
        lines.append(f"{key} = {_fmt(value)}")

    for name in _TABLE_ORDER:
        table = data.get(name)
        if not isinstance(table, dict):
            continue
        lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                continue
            lines.append(f"{key} = {_fmt(value)}")
        for key, value in table.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for entry in value:
                    lines.append("")
                    lines.append(f"[[{name}.{key}]]")
                    for k2, v2 in entry.items():
                        lines.append(f"{k2} = {_fmt(v2)}")

    return "\n".join(lines).strip() + "\n"


def write(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data), encoding="utf-8")
    return path


def read(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def output_names(manifest: dict[str, Any]) -> list[str]:
    """Declared output column names, in declaration order."""
    return [o["name"] for o in manifest.get("signature", {}).get("outputs", [])]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(manifest: dict[str, Any], *, version_dir: Path | None = None) -> list[str]:
    """Return a list of problems; empty means the manifest is valid.

    When ``version_dir`` is given, the referenced weights file must exist and
    its recorded hash must match, which catches an edit to a released version.
    """
    from vp_core import protocols
    from vp_core.hashing import sha256_file

    problems: list[str] = []

    for key in _TOP_REQUIRED:
        if key not in manifest:
            problems.append(f"missing top-level key {key!r}")
    if manifest.get("schema") != SCHEMA_VERSION:
        problems.append(
            f"schema is {manifest.get('schema')!r}, this vp-core writes {SCHEMA_VERSION}"
        )
    if version_dir is not None and manifest.get("version") != version_dir.name:
        problems.append(
            f"version {manifest.get('version')!r} does not match directory "
            f"{version_dir.name!r}"
        )

    sig = manifest.get("signature")
    if not isinstance(sig, dict):
        problems.append("missing [signature] table")
    else:
        if not isinstance(sig.get("version"), int):
            problems.append("signature.version must be an integer")
        outputs = sig.get("outputs") or []
        if not outputs:
            problems.append("signature declares no outputs")
        names = [o.get("name") for o in outputs]
        if len(set(names)) != len(names):
            problems.append(f"duplicate output names: {names}")
        for out in outputs:
            for key in ("name", "dtype", "semantics", "missing"):
                if not out.get(key):
                    problems.append(f"output {out.get('name')!r} missing {key!r}")
            if out.get("dtype") not in DTYPES:
                problems.append(
                    f"output {out.get('name')!r} dtype {out.get('dtype')!r} "
                    f"not one of {list(DTYPES)}"
                )

    proto = manifest.get("protocol")
    if not isinstance(proto, dict):
        problems.append("missing [protocol] table")
    else:
        for key in _PROTOCOL_REQUIRED:
            if key not in proto:
                problems.append(f"protocol missing {key!r}")
        if proto.get("id") and proto["id"] not in protocols.PROTOCOLS:
            problems.append(
                f"protocol id {proto['id']!r} is not in the vp-core registry "
                f"{protocols.all_ids()}"
            )

    prov = manifest.get("provenance")
    if not isinstance(prov, dict):
        problems.append("missing [provenance] table")
    else:
        for key in _PROVENANCE_REQUIRED:
            if not prov.get(key):
                problems.append(f"provenance missing {key!r}")

    data = manifest.get("dataset")
    if isinstance(data, dict):
        for key in _DATASET_REQUIRED:
            if key not in data:
                problems.append(f"dataset missing {key!r}")
        if data.get("redistributable") and not data.get("path"):
            problems.append("dataset is redistributable but declares no path")
        if data.get("redistributable") is False and data.get("path"):
            problems.append(
                "dataset is not redistributable but declares a path — only the "
                "hash and fetch command may ship"
            )

    model = manifest.get("model")
    if isinstance(model, dict):
        for key in _MODEL_REQUIRED:
            if key not in model:
                problems.append(f"model missing {key!r}")
        if version_dir is not None and model.get("weights"):
            weights = version_dir / str(model["weights"])
            if not weights.exists():
                problems.append(f"weights file missing: {weights}")
            elif model.get("sha256") and sha256_file(weights) != model["sha256"]:
                problems.append(
                    f"weights hash mismatch for {weights.name} — a released version "
                    "was edited in place"
                )

    if isinstance(data, dict) != isinstance(model, dict):
        problems.append(
            "[dataset] and [model] must both be present or both absent "
            "(a training-free pathway has neither)"
        )

    return problems
