# vp-core

Shared infrastructure for virtual-pathway packages: scaffold splits, stateless
featurisers, binary metrics, the evaluation-protocol registry, the version
manifest schema and the documentation lints.

Pathways pin this package's major version. Split functions, featurisers and
metric definitions are defined here. A change to them changes what a recorded
number means. The core version is written into every version manifest's
`[protocol]` table.

## Content

`protocols` defines the evaluation recipes and refuses cross-protocol
comparisons. `splits` provides Bemis-Murcko scaffold splitting in its
evaluation (three-way) and deployment (two-way) forms. `dataset` fixes the
three-column table contract and its canonical hash. `manifest` reads, writes
and validates the version record. `registry` turns a versions directory into a
loadable, self-describing predictor. `docs` holds the README lint.

## Install

    pip install vp-core

## Licence

Apache-2.0. See `LICENSE`.
