"""Generate a version's human-readable card from its machine record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["render_card", "write_card"]


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"


def _metric_rows(test: dict[str, Any]) -> list[str]:
    rows = ["| metric | mean | std | per seed |", "|---|---|---|---|"]
    for name, agg in test.items():
        per_seed = ", ".join(_fmt(v) for v in agg["per_seed"])
        rows.append(
            f"| {name} | {_fmt(agg['mean'])} | {_fmt(agg['std'])} | {per_seed} |"
        )
    return rows


def _performance_section(
    entry: dict[str, Any],
    manifest: dict[str, Any],
    protocol_id: str,
    *,
    multi: bool,
) -> list[str]:
    """One protocol's section, from its heading to its last metric row.

    The heading names the protocol only when the version carries more than one,
    so a single-protocol card renders exactly as it did before protocols could
    be stacked — released cards must stay reproducible.
    """
    from vp_core import manifest as manifest_mod
    from vp_core import protocols

    proto = protocols.PROTOCOLS.get(protocol_id)
    primary = manifest_mod.output_names(manifest)
    heading = f"### `{primary[0]}`" if entry.get("additional_outputs") else None
    lines = [
        f"## Performance — `{protocol_id}`" if multi else "## Performance",
        "",
        f"Protocol `{protocol_id}` — "
        f"{proto.description if proto else 'unknown protocol'}",
        "",
        f"Evaluated {entry.get('evaluated')} on "
        f"n_train={entry['n']['train']}, n_val={entry['n']['val']}, "
        f"n_test={entry['n']['test']}.",
        "",
        *([heading, ""] if heading else []),
        *_metric_rows(entry["test"]),
    ]
    for extra in entry.get("additional_outputs", []):
        lines += [
            "",
            f"### `{extra['name']}`",
            "",
            f"Measured on n_train={extra['n']['train']}, "
            f"n_val={extra['n']['val']}, n_test={extra['n']['test']}.",
            "",
            *_metric_rows(extra["test"]),
        ]
    return lines


def render_card(manifest: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    from vp_core import metrics_store

    pathway = manifest["pathway"]
    version = manifest["version"]
    sig = manifest.get("signature", {})
    lines: list[str] = [
        f"# {pathway} {version}",
        "",
        "_Generated from `manifest.toml` and `metrics.json`. Do not edit._",
        "",
        f"Released {manifest['released']} · signature {sig.get('version')}"
        + (f" · supersedes {manifest['supersedes']}" if manifest.get("supersedes") else ""),
        "",
        f"**Why this version.** {manifest['reason']}",
        "",
        "## Outputs",
        "",
        "| column | dtype | range | meaning |",
        "|---|---|---|---|",
    ]
    for out in sig.get("outputs", []):
        rng = out.get("range")
        rng_s = f"{rng[0]}–{rng[1]}" if rng else "—"
        lines.append(
            f"| `{out['name']}` | {out['dtype']} | {rng_s} | {out['semantics']} |"
        )
    lines += [
        "",
        f"Missing values: {sig.get('outputs', [{}])[0].get('missing', 'n/a')}",
        "",
    ]

    declared = manifest.get("protocol", {}).get("id")
    measured = metrics_store.entries(metrics)
    if measured:
        ids = metrics_store.protocol_ids(metrics, declared=declared)
        for position, protocol_id in enumerate(ids):
            if position:
                lines.append("")
            lines += _performance_section(
                measured[protocol_id], manifest, protocol_id, multi=len(ids) > 1
            )
        lines += [
            "",
            "> Comparable only with metrics carrying the same protocol id.",
            "",
        ]

    data = manifest.get("dataset")
    if data:
        lines += [
            "## Data",
            "",
            f"{data['name']} — {data['source']}. Retrieved {data['retrieved']}, "
            f"licensed {data['licence']}"
            + (", redistributed here." if data.get("redistributable") else ", not redistributable."),
            "",
            f"`{data['n_rows']}` compounds, positive rate `{data['base_rate']:.3f}`, "
            f"table SHA-256 `{data['sha256'][:16]}…`",
            "",
            f"Regenerate and check for upstream drift with `{data['fetch']}`.",
            "",
        ]

    model = manifest.get("model")
    if model:
        lines += [
            "## Model",
            "",
            f"{model['family']} on `{model['features']}` features. "
            f"Shipped weights: {model['fit']}",
            "",
            f"`{model['weights']}` SHA-256 `{model['sha256'][:16]}…`",
            "",
        ]

    prov = manifest.get("provenance", {})
    env = ", ".join(f"{k} {v}" for k, v in prov.items())
    lines += [
        "## Provenance",
        "",
        f"Environment: {env}.",
        "",
        "Reproducibility is to this dataset hash and this environment, not "
        "bit-exact: the sources are live endpoints and RDKit descriptor values "
        "move between releases.",
    ]
    return "\n".join(lines) + "\n"


def write_card(version_dir: str | Path) -> Path:
    """Render ``CARD.md`` from the manifest and metrics in ``version_dir``."""
    from vp_core import manifest as manifest_mod
    from vp_core import metrics_store

    version_dir = Path(version_dir)
    manifest = manifest_mod.read(version_dir / "manifest.toml")
    metrics = metrics_store.read(version_dir)
    out = version_dir / "CARD.md"
    out.write_text(render_card(manifest, metrics), encoding="utf-8")
    return out
