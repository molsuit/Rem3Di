"""Direct unit tests for FilterMoleculeStage and FilterAtomsStage.

Generator-level tests (TDC, MoleculeNet, etc.) cover the integrated path; this
file pokes the stages directly so the reindexing of parallel arrays
(regression_data, charges, multiplicities, structure_ids) is verified once.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from remedi.configuration.dataset_config import (
    FilterAtomsStageConfig,
    FilterMoleculeStageConfig,
)
from remedi.data_handling.dataset.tasks import ElementSet, Split
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    FilterAtomsStage,
    FilterMoleculeStage,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID


def _smiles_batch(
    raw: list[str | None],
    *,
    with_targets: bool = False,
    split_codes: list[int] | None = None,
) -> InputBatch:
    n = len(raw)
    structure_ids = [
        StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i) for i in range(n)
    ]
    rd = None
    if with_targets:
        targets = np.arange(n, dtype=np.float64).reshape(-1, 1)
        masks = np.ones_like(targets, dtype=np.uint8)
        kwargs: dict = {"targets_system": targets, "mask_system": masks}
        if split_codes is not None:
            kwargs["split"] = np.asarray(split_codes, dtype=np.uint8)
        rd = RegressionData(**kwargs)
    return InputBatch(
        smiles=None,
        molecules=None,
        raw_smiles=raw,
        structure_ids=structure_ids,
        regression_data=rd,
    )


def test_filter_mol_stage_drops_invalid_and_canonicalizes():
    stage = FilterMoleculeStage(config=FilterMoleculeStageConfig(max_atoms=100))
    batch = _smiles_batch(
        ["CCO", "not_a_smiles", "c1ccccc1", "C"],  # methane has <3 atoms
        with_targets=True,
    )
    out, _ = stage(batch, None)

    isos = [s.isomeric_smiles for s in out.smiles]
    assert isos == ["CCO", "c1ccccc1"]
    np.testing.assert_array_equal(
        out.regression_data.targets_system, np.array([[0.0], [2.0]])
    )
    assert out.raw_smiles is None
    assert len(out.structure_ids) == 2
    assert stage.load_stats.n_raw_rows == 4
    assert stage.load_stats.n_invalid_smiles == 1
    assert stage.load_stats.n_filtered_out == 1
    assert stage.load_stats.n_kept == 2


def test_filter_mol_stage_dedupe_across_batches_first_wins():
    """Stateful ``_seen`` set preserves TDC's first-split-wins priority."""
    stage = FilterMoleculeStage(config=FilterMoleculeStageConfig(max_atoms=100))

    b1 = _smiles_batch(
        ["CCO", "c1ccccc1"],
        with_targets=True,
        split_codes=[Split.train.value, Split.train.value],
    )
    b2 = _smiles_batch(
        ["CCO", "CCN"],  # CCO duplicate
        with_targets=True,
        split_codes=[Split.test.value, Split.valid.value],
    )

    out1, _ = stage(b1, None)
    out2, _ = stage(b2, None)

    assert [s.isomeric_smiles for s in out1.smiles] == ["CCO", "c1ccccc1"]
    # CCO dropped on the second batch as a duplicate; CCN survives.
    assert [s.isomeric_smiles for s in out2.smiles] == ["CCN"]
    assert int(out2.regression_data.split[0]) == Split.valid.value
    assert stage.load_stats.n_duplicates == 1


def test_filter_mol_stage_reindexes_charge_and_mult():
    stage = FilterMoleculeStage(config=FilterMoleculeStageConfig(max_atoms=100))
    batch = InputBatch(
        smiles=None,
        molecules=None,
        raw_smiles=["CCO", "not_a_smiles", "c1ccccc1"],
        structure_ids=[
            StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i)
            for i in range(3)
        ],
        total_charge=[1.0, 2.0, 3.0],
        multiplicity=[1.0, 2.0, 3.0],
    )
    out, _ = stage(batch, None)
    assert out.total_charge == [1.0, 3.0]
    assert out.multiplicity == [1.0, 3.0]


def test_filter_mol_stage_dedupe_off_keeps_duplicates():
    stage = FilterMoleculeStage(
        config=FilterMoleculeStageConfig(max_atoms=100, dedupe=False)
    )
    batch = _smiles_batch(["CCO", "CCO"])
    out, _ = stage(batch, None)
    assert [s.isomeric_smiles for s in out.smiles] == ["CCO", "CCO"]


def test_filter_mol_stage_requires_raw_smiles():
    stage = FilterMoleculeStage(config=FilterMoleculeStageConfig())
    batch = InputBatch(
        smiles=None,
        molecules=None,
        raw_smiles=None,
        structure_ids=[],
    )
    with pytest.raises(ValueError, match="raw_smiles"):
        stage(batch, None)


def _atoms_batch(
    sized: list[int],
    *,
    elements: list[str] | None = None,
    charges: list[float] | None = None,
    mults: list[float] | None = None,
) -> InputBatch:
    """Build a batch of ASE Atoms with given (heavy) atom counts.

    Each entry becomes a single carbon chain padded with H, unless
    ``elements[i]`` is set, in which case it's a homo-element block of
    length ``sized[i]``.
    """
    mols: list[Atoms] = []
    for i, n in enumerate(sized):
        sym = (elements[i] if elements is not None else "C") * 1
        formula = sym * n
        pos = np.zeros((n, 3), dtype=float)
        mols.append(Atoms(formula, positions=pos))
    structure_ids = [
        StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i)
        for i in range(len(sized))
    ]
    return InputBatch(
        smiles=None,
        molecules=mols,
        structure_ids=structure_ids,
        total_charge=charges,
        multiplicity=mults,
    )


def test_filter_atoms_stage_max_atoms_gate():
    stage = FilterAtomsStage(config=FilterAtomsStageConfig(max_atoms=4))
    # Heavy-only check: keep <=4, drop the 6-atom carbon chain.
    batch = _atoms_batch([3, 4, 6], charges=[0.0, 1.0, -1.0], mults=[1.0, 2.0, 3.0])
    out, _ = stage(batch, None)
    assert len(out.molecules) == 2
    assert out.total_charge == [0.0, 1.0]
    assert out.multiplicity == [1.0, 2.0]
    assert stage.load_stats.n_filtered_out == 1
    assert stage.load_stats.n_kept == 2


def test_filter_atoms_stage_element_gate_rejects_si_under_mace_off():
    stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
    )
    # One pure-carbon block (kept), one pure-silicon block (dropped under MACE-OFF).
    batch = _atoms_batch([3, 3], elements=["C", "Si"])
    out, _ = stage(batch, None)
    assert len(out.molecules) == 1
    assert stage.load_stats.n_filtered_out == 1


def test_filter_atoms_stage_h_ratio_and_zero_h():
    stage = FilterAtomsStage(
        config=FilterAtomsStageConfig(reject_zero_h=True, min_h_heavy_ratio=0.5)
    )
    # 4 carbons + 1 H -> ratio 0.25 (rejected); CH4 (1 C + 4 H) ratio 4.0 (kept);
    # pure 3 C (zero H) -> rejected.
    a_low = Atoms("CCCCH", positions=np.zeros((5, 3)))
    a_ok = Atoms("CH4", positions=np.zeros((5, 3)))
    a_zero_h = Atoms("CCC", positions=np.zeros((3, 3)))
    batch = InputBatch(
        smiles=None,
        molecules=[a_low, a_ok, a_zero_h],
        structure_ids=[
            StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i)
            for i in range(3)
        ],
    )
    out, _ = stage(batch, None)
    assert len(out.molecules) == 1
    assert stage.load_stats.n_filtered_out == 2


def test_filter_atoms_stage_reindexes_regression_data():
    stage = FilterAtomsStage(config=FilterAtomsStageConfig(max_atoms=5))
    mols = [Atoms("C" * n, positions=np.zeros((n, 3))) for n in (3, 6, 4)]
    targets = np.array([[10.0], [20.0], [30.0]])
    masks = np.ones_like(targets, dtype=np.uint8)
    batch = InputBatch(
        smiles=None,
        molecules=mols,
        structure_ids=[
            StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i)
            for i in range(3)
        ],
        regression_data=RegressionData(targets_system=targets, mask_system=masks),
    )
    out, _ = stage(batch, None)
    np.testing.assert_array_equal(
        out.regression_data.targets_system, np.array([[10.0], [30.0]])
    )
