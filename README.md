# vp-core

Shared infrastructure for virtual-pathway packages: scaffold splits, stateless
featurisers, binary metrics, the evaluation-protocol registry, the version
manifest schema and the documentation lints.

Every pathway pins this package's major version. The split function, the
featurisers and the metric definitions live here, so a change to any of them
changes what a recorded number means, and the core version is written into every
version manifest's `[protocol]` table.

## What is in it

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
