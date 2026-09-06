"""Documentation lints.

A pathway README has a fixed shape so that a reader landing on any pathway repo
finds the same seven answers in the same order, and so that the document cannot
quietly rot. The two prohibitions matter most: a README may carry no metric
table and must stay short. Inlined metric tables are exactly what went stale in
the predecessor repo, so the README states one headline and links to the version
card, which is generated.

These checks are importable rather than living in a test file, so a published
pathway repo lints its own README with no access to the monorepo.
"""

from __future__ import annotations

import re

__all__ = ["MAX_README_LINES", "REQUIRED_SECTIONS", "check_pathway_readme"]

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Install",
    "Use",
    "Current version",
    "Data",
    "Retrain",
    "Licence",
    "Cite",
)

MAX_README_LINES = 80


def check_pathway_readme(text: str, *, package: str | None = None) -> list[str]:
    """Return a list of lint failures; empty means the README conforms."""
    problems: list[str] = []
    lines = text.splitlines()

    if len(lines) > MAX_README_LINES:
        problems.append(
            f"{len(lines)} lines exceeds the {MAX_README_LINES}-line cap — move detail "
            "into the version card or CHANGELOG"
        )

    h1 = next((ln for ln in lines if ln.startswith("# ")), None)
    if h1 is None:
        problems.append("no H1 title")
    elif package and h1[2:].strip() != package:
        problems.append(f"H1 is {h1[2:].strip()!r}, expected the package name {package!r}")

    found = [ln[3:].strip() for ln in lines if ln.startswith("## ")]
    missing = [s for s in REQUIRED_SECTIONS if s not in found]
    if missing:
        problems.append(f"missing required sections: {missing}")
    else:
        order = [found.index(s) for s in REQUIRED_SECTIONS]
        if order != sorted(order):
            problems.append(
                f"sections out of order; expected {list(REQUIRED_SECTIONS)}, found {found}"
            )

    if any(ln.lstrip().startswith("|") for ln in lines):
        problems.append(
            "contains a markdown table — READMEs carry no metric tables; link the "
            "generated version card instead"
        )

    body = _section_body(lines, "Current version")
    if body is not None and not re.search(r"versions/[^/\s]+/CARD\.md", body):
        problems.append(
            "the 'Current version' section does not link a versions/<v>/CARD.md record"
        )

    return problems


def _section_body(lines: list[str], heading: str) -> str | None:
    try:
        start = lines.index(f"## {heading}")
    except ValueError:
        return None
    rest = lines[start + 1 :]
    end = next((i for i, ln in enumerate(rest) if ln.startswith("## ")), len(rest))
    return "\n".join(rest[:end])
