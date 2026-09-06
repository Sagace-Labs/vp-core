"""The SMILES standardiser.

Datasets and hashes are keyed on an InChIKey produced here.

Normalise, take the largest fragment, neutralise, then emit the canonical
isomeric SMILES and its InChIKey.
"""

from __future__ import annotations

__all__ = ["standardise", "standardise_many"]


def _tools():
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    return (
        rdMolStandardize.Normalizer(),
        rdMolStandardize.LargestFragmentChooser(),
        rdMolStandardize.Uncharger(),
    )


def standardise(smiles: str) -> tuple[str | None, str | None]:
    """Return ``(canonical_smiles, inchikey)``, or ``(None, None)`` on failure."""
    from rdkit import Chem

    norm, largest, uncharger = _tools()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None, None
        mol = uncharger.uncharge(largest.choose(norm.normalize(mol)))
        return (
            Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
            Chem.MolToInchiKey(mol),
        )
    except Exception:
        return None, None


def standardise_many(
    smiles: list[str],
) -> tuple[list[str | None], list[str | None]]:
    """Vectorised :func:`standardise`; results align positionally with the input."""
    from rdkit import Chem

    norm, largest, uncharger = _tools()
    out_smi: list[str | None] = []
    out_key: list[str | None] = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumHeavyAtoms() == 0:
                out_smi.append(None)
                out_key.append(None)
                continue
            mol = uncharger.uncharge(largest.choose(norm.normalize(mol)))
            out_smi.append(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
            out_key.append(Chem.MolToInchiKey(mol))
        except Exception:
            out_smi.append(None)
            out_key.append(None)
    return out_smi, out_key
