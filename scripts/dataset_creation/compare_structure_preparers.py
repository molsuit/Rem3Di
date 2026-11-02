"""Compare a dataset from ``remedi_data.structures`` with the old generators on the same raw files.

The check of §1.4 step 2: the old generators (``remedi.data_handling.dataset_creation``)
run on the first ``--records`` records of a source, with the settings of the old
build scripts, and are matched to the new dataset by source record. Reported:
structure counts, records kept by only one side (with the new side's drop
reason), SMILES / charge / multiplicity disagreements, and the largest position
difference after matching atoms (the new path reorders atoms to the SMILES, so
frames are compared as sorted ``(atomic number, x, y, z)`` rows).

Deleted with the old path (§1.4 step 5).

Usage::

    uv run --no-sync python scripts/dataset_creation/compare_structure_preparers.py \
        --source pcqm4mv2 --config configs/dataset_creation/prepare_structures.yaml \
        --records 100000 --output comparison_pcqm4mv2.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml
import yaml
from rdkit import Chem

from remedi_data.build import read_packed_structures
from remedi_data.bundle import read_bundle
from remedi_data.chemistry.elements import ElementSet
from remedi_data.chemistry.geometry import GeometryLimits, geometry_violations
from remedi_data.chemistry.smiles_filter import SmilesFilterConfig
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.structures import (
    GeomDrugsSource,
    Pcqm4mv2Source,
    StructurePreparationConfig,
    StructureRecord,
    TmqmSource,
    prepare_record,
)


@dataclass
class OldStructure:
    key: str
    isomeric_smiles: str | None
    total_charge: float
    multiplicity: float
    atomic_numbers: np.ndarray
    positions: np.ndarray


def sorted_frame(atomic_numbers: np.ndarray, positions: np.ndarray) -> np.ndarray:
    rows = np.column_stack([atomic_numbers.astype(np.float64), positions])
    return rows[np.lexsort(rows.T[::-1])]


def old_pcqm(source: Pcqm4mv2Source, records: int) -> list[OldStructure]:
    """``load_pcqm100k_juwels.py``: filter with neutralise + dedupe, mace_off gate."""
    from remedi.data_handling.dataset_creation.generators.sdf_generator import (
        SDFMoleculeGenerator,
    )

    generator = SDFMoleculeGenerator(
        sdf_file=source.sdf,
        loading_batch_size=10000,
        filter_config=SmilesFilterConfig(
            max_atoms=100,
            element_set=ElementSet.mace_off,
            strip_salts=True,
            neutralize=True,
            dedupe=True,
        ),
    )
    limits = GeometryLimits(elements=ElementSet.mace_off)
    found: list[OldStructure] = []
    for batch in generator:
        for atoms, smiles, identifier, charge, multiplicity in zip(
            batch.molecules,
            batch.smiles,
            batch.structure_ids,
            batch.total_charge,
            batch.multiplicity,
            strict=True,
        ):
            if identifier.structure_id >= records:
                return found
            if geometry_violations(atoms, limits):
                continue
            found.append(
                OldStructure(
                    key=str(identifier.structure_id),
                    isomeric_smiles=smiles.isomeric_smiles,
                    total_charge=charge,
                    multiplicity=multiplicity,
                    atomic_numbers=atoms.get_atomic_numbers(),
                    positions=atoms.get_positions(),
                )
            )
    return found


def old_tmqm(
    source: TmqmSource, records: int, work_directory: Path
) -> list[OldStructure]:
    """``load_from_xyz_tmqm.py``: ase extxyz reader, charge ``q``, multiplicity 1.

    ase stops silently at the first blank line, and the current release
    separates entries by blank lines, so the old reader gets copies without
    them (on the release files themselves it reads one entry per file).
    """
    import gzip

    from remedi.data_handling.dataset_creation.generators.xyz_generator import (
        XYZMoleculeGenerator,
    )

    stripped_files = []
    for name in ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz"):
        stripped = work_directory / "old_reader" / name.removesuffix(".gz")
        stripped.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(source.directory / name, "rt") as raw, stripped.open("w") as out:
            out.writelines(line for line in raw if line.strip())
        stripped_files.append(stripped)
    generator = XYZMoleculeGenerator(
        xyz_file=stripped_files,
        loading_batch_size=10000,
        charge_key="q",
        spin_key=None,
    )
    limits = GeometryLimits(
        max_atoms=200, reject_zero_hydrogen=True, min_hydrogen_heavy_ratio=0.1
    )
    found: list[OldStructure] = []
    for batch in generator:
        for atoms, identifier, charge, multiplicity in zip(
            batch.molecules,
            batch.structure_ids,
            batch.total_charge,
            batch.multiplicity,
            strict=True,
        ):
            if identifier.structure_id >= records:
                return found
            if geometry_violations(atoms, limits):
                continue
            found.append(
                OldStructure(
                    key=str(atoms.info.get("CSD_code", identifier.structure_id)),
                    isomeric_smiles=None,
                    total_charge=charge,
                    multiplicity=multiplicity,
                    atomic_numbers=atoms.get_atomic_numbers(),
                    positions=atoms.get_positions(),
                )
            )
    return found


def old_geom(source: GeomDrugsSource, records: int) -> list[OldStructure]:
    """``load_geom_drugs.py``: conformers with Boltzmann weight > 0.05, the old filter.

    The old worker is called per molecule in summary order (the old build
    shuffled molecules; the order does not change which conformers it keeps).
    Its conformers are keyed by rank in decreasing weight, as the new ones are.
    """
    from remedi.data_handling.dataset_creation.generators.geom_generator import (
        GeomGenerator,
    )

    with (source.directory / "summary_drugs.json").open() as handle:
        summary = json.load(handle)
    old_filter = SmilesFilterConfig(
        max_atoms=100,
        element_set=ElementSet.mace_off,
        strip_salts=True,
        neutralize=True,
        dedupe=True,
    )
    found: list[OldStructure] = []
    for position, (geom_smiles, entry) in enumerate(summary.items()):
        if position >= records:
            break
        path = source.directory / str(entry.get("pickle_path"))
        if not path.exists():
            continue
        results = GeomGenerator._process_one_file(
            (position, path, 0.05, 128, old_filter)
        )
        weights = _geom_weights(path)
        ranked = sorted(
            results, key=lambda result: -weights[_kept_conformer_index(path, result[1])]
        )
        for rank, (_, _, smiles, numbers, positions) in enumerate(ranked):
            found.append(
                OldStructure(
                    key=f"{geom_smiles}#{rank}",
                    isomeric_smiles=smiles,
                    total_charge=0.0,
                    multiplicity=1.0,
                    atomic_numbers=np.asarray(numbers),
                    positions=np.asarray(positions),
                )
            )
    return found


def _geom_weights(path: Path) -> list[float]:
    import pickle

    with path.open("rb") as handle:
        return [
            conformer["boltzmannweight"]
            for conformer in pickle.load(handle)["conformers"]
        ]


def _kept_conformer_index(path: Path, kept_position: int) -> int:
    """The old worker numbered only the conformers above 0.05; map back to the pickle."""
    weights = _geom_weights(path)
    kept = [index for index, weight in enumerate(weights) if weight > 0.05]
    return kept[kept_position]


def new_structures(
    dataset_path: Path, keys: set[str] | None
) -> dict[str, OldStructure]:
    bundle = read_bundle(dataset_path)
    table = bundle.table
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)
    try:
        packed = read_packed_structures(dataset)
    finally:
        dataset.close()
    found: dict[str, OldStructure] = {}
    smiles = table["isomeric_smiles"] if "isomeric_smiles" in table else None
    for row, key in enumerate(table["source_key"].astype(str)):
        if keys is not None and key not in keys:
            continue
        numbers, positions = packed.frame(row)
        found[key] = OldStructure(
            key=key,
            isomeric_smiles=None if smiles is None else str(smiles.iloc[row]),
            total_charge=float(table["total_charge"].iloc[row]),
            multiplicity=float(table["multiplicity"].iloc[row]),
            atomic_numbers=np.asarray(numbers),
            positions=np.asarray(positions, dtype=np.float64),
        )
    return found


def new_drop_reasons(
    source, keys: list[str], work_directory: Path
) -> collections.Counter:
    """Why the new path dropped records the old path kept (re-run on those records)."""
    reasons: collections.Counter = collections.Counter()
    wanted = set(keys)
    limits = source.geometry_limits
    allowed = limits.allowed_atomic_numbers()
    if isinstance(source, GeomDrugsSource):
        stop = len(pd.read_parquet(work_directory / "geom_molecules.parquet"))
    else:
        stop = (
            source.record_count(work_directory)
            if not (work_directory / "plan.yaml").exists()
            else yaml.safe_load((work_directory / "plan.yaml").read_text())[
                "record_count"
            ]
        )
    for record in source.read_records(0, stop, work_directory):
        if record.source_key not in wanted:
            continue
        wanted.discard(record.source_key)
        if not isinstance(record, StructureRecord):
            reasons[record.reason] += 1
            continue
        outcome = prepare_record(
            record,
            smiles_filter=source.smiles_filter,
            limits=limits,
            allowed_atomic_numbers=allowed,
        )
        reasons[outcome if isinstance(outcome, str) else "kept_by_new"] += 1
        if not wanted:
            break
    reasons["not_produced_by_new_source"] += len(wanted)
    return reasons


def load_both(
    source, records: int, dataset_path: Path, work_directory: Path
) -> tuple[list[OldStructure], dict[str, OldStructure]]:
    """The old generator's structures and the new dataset's, over the same records."""
    if isinstance(source, Pcqm4mv2Source):
        old = old_pcqm(source, records)
        new = {
            key: value
            for key, value in new_structures(dataset_path, None).items()
            if int(key) < records
        }
    elif isinstance(source, TmqmSource):
        old = old_tmqm(source, records, work_directory)
        new = dict(list(new_structures(dataset_path, None).items())[:records])
    else:
        old = old_geom(source, records)
        index = pd.read_parquet(work_directory / "geom_molecules.parquet")
        first = set(index["geom_smiles"].iloc[:records])
        new = {
            key: value
            for key, value in new_structures(dataset_path, None).items()
            if key.rsplit("#", 1)[0] in first
        }
    return old, new


def pair_differences(
    old: OldStructure, new: OldStructure
) -> tuple[dict[str, list], float | None]:
    """Field -> ``[old, new]`` for each disagreement, and the largest position
    difference (``None`` when the atoms themselves differ)."""
    differences: dict[str, list] = {}
    if old.isomeric_smiles is not None and new.isomeric_smiles is not None:
        if Chem.CanonSmiles(old.isomeric_smiles, useChiral=False) != Chem.CanonSmiles(
            new.isomeric_smiles, useChiral=False
        ):
            differences["nonisomeric_smiles"] = [
                old.isomeric_smiles,
                new.isomeric_smiles,
            ]
        elif Chem.CanonSmiles(old.isomeric_smiles) != new.isomeric_smiles:
            differences["stereo_only"] = [old.isomeric_smiles, new.isomeric_smiles]
    for field in ("total_charge", "multiplicity"):
        if getattr(old, field) != getattr(new, field):
            differences[field] = [getattr(old, field), getattr(new, field)]
    old_rows = sorted_frame(
        old.atomic_numbers, old.positions.astype(np.float32).astype(np.float64)
    )
    new_rows = sorted_frame(new.atomic_numbers, new.positions)
    if old_rows.shape != new_rows.shape or not np.array_equal(
        old_rows[:, 0], new_rows[:, 0]
    ):
        differences["atoms"] = [len(old_rows), len(new_rows)]
        return differences, None
    return differences, float(np.abs(old_rows[:, 1:] - new_rows[:, 1:]).max())


def compare(source, records: int, dataset_path: Path, work_directory: Path) -> dict:
    old, new = load_both(source, records, dataset_path, work_directory)
    old_by_key = {structure.key: structure for structure in old}
    only_old = sorted(set(old_by_key) - set(new))
    only_new = sorted(set(new) - set(old_by_key))
    common = sorted(set(old_by_key) & set(new))
    counts: collections.Counter = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    largest_position_difference = 0.0
    for key in common:
        differences, position_difference = pair_differences(old_by_key[key], new[key])
        for field, values in differences.items():
            counts[field] += 1
            if len(examples[field]) < 5:
                examples[field].append([key, *values])
        if position_difference is not None:
            largest_position_difference = max(
                largest_position_difference, position_difference
            )
    return {
        "source": source.kind,
        "records_compared": records,
        "old_structures": len(old_by_key),
        "new_structures": len(new),
        "common": len(common),
        "only_old": len(only_old),
        "only_new": len(only_new),
        "only_old_new_drop_reasons": (
            dict(new_drop_reasons(source, only_old, work_directory)) if only_old else {}
        ),
        "only_old_examples": only_old[:5],
        "only_new_examples": only_new[:5],
        "disagreements_among_common": dict(counts),
        "largest_position_difference_angstrom": largest_position_difference,
        "examples": dict(examples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = pydantic_yaml.parse_yaml_file_as(
        StructurePreparationConfig, arguments.config
    )
    source = next(
        source for source in config.sources if source.dataset_id == arguments.source
    )
    report = compare(
        source,
        arguments.records,
        config.output_root / source.dataset_id,
        config.work_root / source.dataset_id,
    )
    arguments.output.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    )
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
