# vp-core

Shared infrastructure for virtual-pathway packages: scaffold splits, stateless
featurisers, binary metrics, the evaluation-protocol registry, the version
manifest schema and the documentation lints.

Every pathway package depends on this one and pins its major version, because
the split function, the featurisers and the metric definitions all live here —
a change to any of them changes what a recorded number means. That is why the
core version is written into every version manifest's `[protocol]` table.

Nothing pathway-specific and nothing proprietary belongs here.

## What is in it

`protocols` defines the evaluation recipes and refuses cross-protocol
comparisons. `splits` provides Bemis-Murcko scaffold splitting in its
evaluation (three-way) and deployment (two-way) forms. `dataset` fixes the
three-column table contract and its canonical hash. `manifest` reads, writes
and validates the version record. `registry` turns a versions directory into a
loadable, self-describing predictor. `docs` holds the README lint so a
published pathway repository can check itself.

## Install

    pip install vp-core

## Licence

Apache-2.0. See `LICENSE`.
