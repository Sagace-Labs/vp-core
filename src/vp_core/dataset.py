"""The dataset table contract and its hash.

Every pathway dataset is reduced to the same three columns: ``inchikey``,
``smiles`` (standardised) and ``label``. Extra columns may travel alongside as
provenance but are excluded from the hash.

The hash covers the **standardised** table, not the raw download, so it is
stable under cosmetic upstream change and moves when the science does.

Serialisation for hashing is CSV with a fixed float format and a fixed row
order, not Parquet — Parquet bytes vary across writer versions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "dataset_hash",
    "read_table",
    "stratified_example",
    "validate_table",
    "write_table",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("inchikey", "smiles", "label")


def validate_table(df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations; empty means the table is valid."""
    errors: list[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"missing required column {col!r}")
    if errors:
        return errors

    n_dup = int(df["inchikey"].duplicated().sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate inchikey rows — one row per compound is required")
    if bool(df["inchikey"].isna().any()) or bool(df["smiles"].isna().any()):
        errors.append("null inchikey or smiles")
    labels = set(pd.unique(df["label"].dropna()))
    if not labels <= {0, 1}:
        errors.append(f"label must be binary 0/1, found {sorted(labels)}")
    if len(labels) < 2:
        errors.append("label has a single class — the table is degenerate")
    return errors


def dataset_hash(df: pd.DataFrame) -> str:
    """SHA-256 over the contract columns only, in a canonical order and format."""
    canon = (
        df.loc[:, list(REQUIRED_COLUMNS)]
        .assign(label=lambda d: d["label"].astype(int))
        .sort_values("inchikey", kind="mergesort")
        .reset_index(drop=True)
    )
    payload = canon.to_csv(index=False, lineterminator="\n", float_format="%.6g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a dataset table and fail loudly if it breaks the contract."""
    df = pd.read_parquet(path)
    errors = validate_table(df)
    if errors:
        raise ValueError(f"{path} violates the dataset contract: {'; '.join(errors)}")
    return df


def write_table(df: pd.DataFrame, path: str | Path) -> str:
    """Validate, write and return the dataset hash."""
    errors = validate_table(df)
    if errors:
        raise ValueError(f"refusing to write invalid table: {'; '.join(errors)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return dataset_hash(df)


def stratified_example(df: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A small class-stratified sample for the committed test fixture.

    The fixture lets a pathway's tests run with no network and no licensed
    data, so both classes must be present.
    """
    frac = min(1.0, n / len(df))
    parts = []
    for _, group in df.groupby("label", sort=True):
        take = min(len(group), max(2, round(len(group) * frac)))
        parts.append(group.sample(take, random_state=seed))
    return (
        pd.concat(parts)
        .sort_values("inchikey", kind="mergesort")
        .reset_index(drop=True)
    )
