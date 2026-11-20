"""Two small stand-ins for the pretraining corpora, for the GPU check scripts on a
machine without the real data.

Writes ``<output>/pcqm4mv2`` (small molecules joined from fragments, with the
``diverse_100k`` selection) and ``<output>/geom_drugs`` (known drugs of 60-120 atoms
with hydrogens, with the ``top1`` selection), one seeded ETKDG + MMFF conformer per
molecule, through the dataset writer of the test fixtures. The directory names and
selection names are the ones ``scripts/gpu_checks/*.sh`` and the drug_like_small
training config expect. Timings on these stand-ins are not comparable with the real
corpora: they have far fewer distinct molecules, and the benchmark draws batches
from them with replacement.

    uv run python scripts/gpu_checks/make_test_datasets.py --output datasets/gpu_checks
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from rdkit import Chem

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests/support"))

from remedi_data.selection import (  # noqa: E402
    RandomMolecules,
    SelectionConfig,
    select_structures,
)
from testing import write_small_dataset  # noqa: E402

#: Joined as first + linker + second into PCQM4Mv2-sized molecules.
LINKERS = ["", "C", "CC", "O", "N", "C(=O)", "OC", "NC(=O)", "c1ccc(cc1)", "C1CC(C1)"]
FIRST_FRAGMENTS = [
    "C",
    "CC",
    "CCC",
    "CC(C)",
    "OC",
    "NC",
    "FC",
    "ClC",
    "c1ccccc1",
    "c1ccncc1",
    "C1CC1",
    "C1CCOC1",
    "O=C(O)",
    "N#C",
    "CS",
]
SECOND_FRAGMENTS = [
    "CO",
    "CN",
    "C(=O)N",
    "C(=O)O",
    "c1ccccc1",
    "c1ccc(F)cc1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCNCC1",
    "C1CCCCC1",
    "OCC",
    "NC(=O)C",
    "S(=O)(=O)N",
    "CCl",
]

#: Known drugs, written without stereo marks; those outside 60-120 atoms with
#: hydrogens, or that RDKit or the embedding rejects, are skipped.
DRUG_SMILES = {
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "rosuvastatin": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1C=CC(O)CC(O)CC(=O)O",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
    "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
    "loperamide": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
    "fexofenadine": "CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
    "telmisartan": "CCCc1nc2c(C)cc(-c3nc4ccccc4n3C)cc2n1Cc1ccc(-c2ccccc2C(=O)O)cc1",
    "valsartan": "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(C)C)C(=O)O",
    "lisinopril": "NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
    "raloxifene": "Oc1ccc(-c2sc3cc(O)ccc3c2C(=O)c2ccc(OCCN3CCCCC3)cc2)cc1",
    "montelukast": "CC(C)(O)c1ccccc1CCC(SCC1(CC(=O)O)CC1)c1cccc(C=Cc2ccc3ccc(Cl)cc3n2)c1",
    "ritonavir": "CC(C)c1nc(CN(C)C(=O)NC(C(C)C)C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)OCc2cncs2)cs1",
    "saquinavir": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(Cc1ccccc1)NC(=O)C(CC(N)=O)NC(=O)c1ccc2ccccc2n1",
    "diltiazem": "COc1ccc(C2Sc3ccccc3N(CCN(C)C)C(=O)C2OC(C)=O)cc1",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
    "pioglitazone": "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1",
    "olmesartan": "CCCc1nc(C(C)(C)O)c(C(=O)O)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
    "ezetimibe": "OC(CCC1C(=O)N(c2ccc(F)cc2)C1c1ccc(O)cc1)c1ccc(F)cc1",
}

SMALL_HEAVY_ATOMS = (3, 20)
LARGE_ATOMS_WITH_HYDROGENS = (60, 120)


def atoms_with_hydrogens(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return -1
    return Chem.AddHs(molecule).GetNumAtoms()


def small_molecules(count: int) -> list[str]:
    seen: dict[str, None] = {}
    for linker, first, second in itertools.product(
        LINKERS, FIRST_FRAGMENTS, SECOND_FRAGMENTS
    ):
        molecule = Chem.MolFromSmiles(first + linker + second)
        if molecule is None:
            continue
        heavy = molecule.GetNumHeavyAtoms()
        if SMALL_HEAVY_ATOMS[0] <= heavy <= SMALL_HEAVY_ATOMS[1]:
            seen.setdefault(Chem.MolToSmiles(molecule), None)
        if len(seen) == count:
            break
    return list(seen)


def large_molecules() -> list[str]:
    smallest, largest = LARGE_ATOMS_WITH_HYDROGENS
    kept = []
    for name, smiles in DRUG_SMILES.items():
        atoms = atoms_with_hydrogens(smiles)
        if smallest <= atoms <= largest:
            kept.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        else:
            print(f"skipping {name}: {atoms} atoms with hydrogens", file=sys.stderr)
    return kept


def write_corpus(output: Path, dataset_id: str, smiles: list[str], selection: str):
    directory = write_small_dataset(
        output,
        dataset_id=dataset_id,
        labels=[],
        metrics=None,
        targets=None,
        smiles=smiles,
    )
    select_structures(
        directory,
        SelectionConfig(name=selection, steps=[RandomMolecules(count=len(smiles))]),
        overwrite=True,
    )
    print(f"{directory}: {len(smiles)} molecules, selection {selection}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small_molecules", type=int, default=2000)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    small = small_molecules(arguments.small_molecules)
    large = large_molecules()
    if len(large) < 5:
        raise RuntimeError(f"only {len(large)} drugs have 60-120 atoms")
    write_corpus(output, "pcqm4mv2", small, "diverse_100k")
    write_corpus(output, "geom_drugs", large, "top1")


if __name__ == "__main__":
    main()
