"""Protocols are append-only and their definitions are pinned.

Editing a protocol in place changes what every metric already recorded against
it means. The fingerprint file makes that edit fail here; a genuine change is a
new revision (``@2``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vp_core import protocols
from vp_core.metrics import METRICS

FINGERPRINTS = Path(__file__).parent / "protocol_fingerprints.json"


def test_definitions_are_unchanged():
    recorded = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))
    current = {pid: p.fingerprint() for pid, p in protocols.PROTOCOLS.items()}

    removed = sorted(set(recorded) - set(current))
    assert not removed, (
        f"protocols {removed} were removed — protocols are append-only, and "
        "metrics recorded against them still exist"
    )
    changed = [pid for pid in recorded if current[pid] != recorded[pid]]
    assert not changed, (
        f"protocols {changed} were edited in place. Add a new revision instead "
        "(e.g. name@2) so previously recorded metrics keep their meaning."
    )


def test_new_protocols_are_recorded():
    recorded = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))
    added = sorted(set(protocols.PROTOCOLS) - set(recorded))
    assert not added, (
        f"new protocols {added} are not in protocol_fingerprints.json — add them "
        "deliberately"
    )


@pytest.mark.parametrize("protocol_id", protocols.all_ids())
def test_metric_names_are_defined(protocol_id):
    protocol = protocols.get(protocol_id)
    unknown = set(protocol.metrics) - set(METRICS)
    assert not unknown, f"{protocol_id} names undefined metrics {sorted(unknown)}"
    assert protocol.seeds, f"{protocol_id} has no seeds"
    assert 0 < protocol.val_frac < 1 and 0 < protocol.test_frac < 1


def test_unknown_protocol_raises_with_the_options():
    with pytest.raises(KeyError, match="scaffold-shuffle-5seed@1"):
        protocols.get("does-not-exist")


def test_cross_protocol_comparison_is_refused():
    protocols.require_comparable("smoke@1", "smoke@1")
    with pytest.raises(ValueError, match="not comparable"):
        protocols.require_comparable("smoke@1", "scaffold-shuffle-5seed@1")
