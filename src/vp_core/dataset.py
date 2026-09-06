"""The dataset table contract and its hash.

Every pathway dataset carries the two identity columns ``inchikey`` and
``smiles`` (standardised), plus one column per endpoint it labels. A pathway
with a single endpoint uses the default name ``label`` and needs to say nothing
further. Extra columns may travel alongside as provenance but are excluded from
the hash.

A label may be null where the source made no call for that compound. That is a
different statement from a negative, so a table records it as absent rather
than as zero, and a fitting routine drops those rows for that endpoint only.

The hash covers the **standardised** table, not the raw download, so it is
stable under cosmetic upstream change and moves when the science does. It
covers exactly the label columns named, so a version that reads one endpoint of
a two-endpoint table keeps its recorded hash when the other endpoint is added.

Serialisation for hashing is CSV with a fixed float format and a fixed row
order, not Parquet — Parquet bytes vary across writer versions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

__all__ = [
    "DEFAULT_LABELS",
    "IDENTITY_COLUMNS",
    "REQUIRED_COLUMNS",
    "dataset_hash",
    "read_table",
    "stratified_example",
    "validate_table",
    "write_table",
]

IDENTITY_COLUMNS: tuple[str, ...] = ("inchikey", "smiles")
DEFAULT_LABELS: tuple[str, ...] = ("label",)

#: The single-endpoint contract, kept for pathways that declare nothing else.
REQUIRED_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *DEFAULT_LABELS)


def _labels(labels: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(labels) if labels else DEFAULT_LABELS


def validate_table(
    df: pd.DataFrame, *, labels: Sequence[str] | None = None
) -> list[str]:
    """Return a list of contract violations; empty means the table is valid."""
    label_columns = _labels(labels)
    errors: list[str] = []
    for col in (*IDENTITY_COLUMNS, *label_columns):
        if col not in df.columns:
            errors.append(f"missing required column {col!r}")
    if errors:
        return errors

    n_dup = int(df["inchikey"].duplicated().sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate inchikey rows — one row per compound is required")
    if bool(df["inchikey"].isna().any()) or bool(df["smiles"].isna().any()):
        errors.append("null inchikey or smiles")

    for col in label_columns:
        values = set(pd.unique(df[col].dropna()))
        if not values <= {0, 1}:
            errors.append(f"{col} must be binary 0/1, found {sorted(values)}")
        if len(values) < 2:
            errors.append(f"{col} has a single class — the table is degenerate")
    return errors


def dataset_hash(df: pd.DataFrame, *, labels: Sequence[str] | None = None) -> str:
    """SHA-256 over the identity and label columns, in a canonical order and format.

    A null label serialises as an empty field, so adding an endpoint that some
    compounds lack still moves the hash for that endpoint and no other.
    """
    label_columns = _labels(labels)
    canon = df.loc[:, [*IDENTITY_COLUMNS, *label_columns]].copy()
    for col in label_columns:
        # Nullable integers keep 0/1 rendering while leaving absence blank.
        canon[col] = canon[col].astype("Int64")
    canon = canon.sort_values("inchikey", kind="mergesort").reset_index(drop=True)
    payload = canon.to_csv(index=False, lineterminator="\n", float_format="%.6g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_table(
    path: str | Path, *, labels: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a dataset table and fail loudly if it breaks the contract."""
    df = pd.read_parquet(path)
    errors = validate_table(df, labels=labels)
    if errors:
        raise ValueError(f"{path} violates the dataset contract: {'; '.join(errors)}")
    return df


def write_table(
    df: pd.DataFrame, path: str | Path, *, labels: Sequence[str] | None = None
) -> str:
    """Validate, write and return the dataset hash."""
    errors = validate_table(df, labels=labels)
    if errors:
        raise ValueError(f"refusing to write invalid table: {'; '.join(errors)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return dataset_hash(df, labels=labels)


def stratified_example(
    df: pd.DataFrame,
    n: int = 200,
    seed: int = 0,
    *,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """A small sample stratified on the combination of every label column.

    The fixture lets a pathway's tests run with no network and no licensed
    data, so every class of every endpoint must survive the sample — including
    the rarest, which is why groups are taken from the label tuple rather than
    from one column.
    """
    label_columns = _labels(labels)
    frac = min(1.0, n / len(df))
    keys = df[list(label_columns)].astype("Int64").astype(object).fillna("na")
    parts = []
    for _, group in df.groupby([keys[col] for col in label_columns], sort=True):
        take = min(len(group), max(2, round(len(group) * frac)))
        parts.append(group.sample(take, random_state=seed))
    return (
        pd.concat(parts)
        .sort_values("inchikey", kind="mergesort")
        .reset_index(drop=True)
    )
