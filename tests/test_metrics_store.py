"""The metrics file keys one entry per protocol and still reads schema 1."""

from __future__ import annotations

import json

from vp_core import metrics_store


def _entry(protocol_id: str, auc: float) -> dict:
    return {
        "pathway": "demo",
        "version": "v1",
        "protocol_id": protocol_id,
        "dataset_sha256": "0" * 64,
        "dataset": "full",
        "evaluated": "2026-09-19",
        "n": {"total": 10, "train": 6, "val": 2, "test": 2},
        "folds": [{"seed": 0, "train": 6, "val": 2, "test": 2}],
        "test": {"auc_roc": {"mean": auc, "std": 0.0, "per_seed": [auc]}},
    }


def test_a_schema_1_record_reads_as_one_entry():
    """A released version's file is never rewritten, so it must still parse."""
    legacy = _entry("scaffold-shuffle-5seed@1", 0.7)
    assert set(metrics_store.entries(legacy)) == {"scaffold-shuffle-5seed@1"}


def test_an_absent_record_has_no_entries():
    assert metrics_store.entries(None) == {}


def test_merging_keeps_both_protocols():
    record = metrics_store.merge(None, _entry("scaffold-shuffle-5seed@1", 0.7))
    record = metrics_store.merge(record, _entry("scaffold-balanced-5seed@1", 0.8))
    assert record["schema"] == metrics_store.SCHEMA_VERSION
    assert set(record["protocols"]) == {
        "scaffold-shuffle-5seed@1",
        "scaffold-balanced-5seed@1",
    }


def test_re_running_one_protocol_replaces_rather_than_appends():
    record = metrics_store.merge(None, _entry("scaffold-shuffle-5seed@1", 0.7))
    record = metrics_store.merge(record, _entry("scaffold-shuffle-5seed@1", 0.9))
    entries = metrics_store.entries(record)
    assert len(entries) == 1
    assert entries["scaffold-shuffle-5seed@1"]["test"]["auc_roc"]["mean"] == 0.9


def test_merging_onto_a_schema_1_record_carries_the_old_measurement_over():
    legacy = _entry("scaffold-shuffle-5seed@1", 0.7)
    record = metrics_store.merge(legacy, _entry("scaffold-balanced-5seed@1", 0.8))
    assert set(record["protocols"]) == {
        "scaffold-shuffle-5seed@1",
        "scaffold-balanced-5seed@1",
    }
    # The version describes itself once, not once per measurement.
    assert "pathway" not in record["protocols"]["scaffold-shuffle-5seed@1"]


def test_the_declared_protocol_orders_first():
    record = metrics_store.merge(None, _entry("scaffold-shuffle-5seed@1", 0.7))
    record = metrics_store.merge(record, _entry("scaffold-balanced-5seed@1", 0.8))
    ids = metrics_store.protocol_ids(record, declared="scaffold-shuffle-5seed@1")
    assert ids[0] == "scaffold-shuffle-5seed@1"


def test_write_round_trips_through_the_version_directory(tmp_path):
    metrics_store.write(tmp_path, _entry("scaffold-shuffle-5seed@1", 0.7))
    metrics_store.write(tmp_path, _entry("scaffold-balanced-5seed@1", 0.8))
    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(written["protocols"]) == {
        "scaffold-shuffle-5seed@1",
        "scaffold-balanced-5seed@1",
    }
    assert metrics_store.read(tmp_path) == written
    assert metrics_store.entry_for(written, "scaffold-balanced-5seed@1") is not None
