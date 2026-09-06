"""Generate a version's human-readable card from its machine record.

``CARD.md`` is derived, never authored. It restates the manifest and metrics so
a reader can see what a version is without parsing TOML, and because it is
generated it cannot disagree with the files it describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["render_card", "write_card"]


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"


def render_card(manifest: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    from vp_core import protocols

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

    proto_id = manifest.get("protocol", {}).get("id")
    if metrics:
        proto = protocols.PROTOCOLS.get(proto_id)
        lines += [
            "## Performance",
            "",
            f"Protocol `{proto_id}` — {proto.description if proto else 'unknown protocol'}",
            "",
            f"Evaluated {metrics.get('evaluated')} on "
            f"n_train={metrics['n']['train']}, n_val={metrics['n']['val']}, "
            f"n_test={metrics['n']['test']}.",
            "",
            "| metric | mean | std | per seed |",
            "|---|---|---|---|",
        ]
        for name, agg in metrics["test"].items():
            per_seed = ", ".join(_fmt(v) for v in agg["per_seed"])
            lines.append(
                f"| {name} | {_fmt(agg['mean'])} | {_fmt(agg['std'])} | {per_seed} |"
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
    env = ", ".join(
        f"{k} {v}" for k, v in prov.items() if k not in ("repo", "commit", "round")
    )
    lines += [
        "## Provenance",
        "",
        f"Built in `{prov.get('repo')}` at commit `{prov.get('commit')}`"
        + (f", round `{prov['round']}`" if prov.get("round") else "")
        + f". Environment: {env}.",
        "",
        "Reproducibility is to this dataset hash and this environment, not "
        "bit-exact: the sources are live endpoints and RDKit descriptor values "
        "move between releases.",
    ]
    return "\n".join(lines) + "\n"


def write_card(version_dir: str | Path) -> Path:
    """Render ``CARD.md`` from the manifest and metrics in ``version_dir``."""
    from vp_core import manifest as manifest_mod

    version_dir = Path(version_dir)
    manifest = manifest_mod.read(version_dir / "manifest.toml")
    metrics_path = version_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    out = version_dir / "CARD.md"
    out.write_text(render_card(manifest, metrics), encoding="utf-8")
    return out
