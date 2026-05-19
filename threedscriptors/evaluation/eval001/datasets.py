from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem

TaskType = Literal["regression", "binary", "multilabel"]
SplitVariant = Literal["tdc_default", "random", "scaffold"]


@dataclass(frozen=True)
class TdcTaskSpec:
    name: str
    short_name: str
    task_type: TaskType
    mumo_metric: str
    official_metric_hint: str


@dataclass(frozen=True)
class MoleculeNetTaskSpec:
    dataset: str
    task: str
    path: Path
    smiles_column: str
    target_columns: tuple[str, ...]
    task_type: TaskType
    metric: str
    split_variant: SplitVariant


TDC_TASKS: tuple[TdcTaskSpec, ...] = (
    TdcTaskSpec("BBB_Martins", "BBB", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("HIA_Hou", "HIA", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("Pgp_Broccatelli", "Pgp", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("Bioavailability_Ma", "Bioavailability", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("DILI", "DILI", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("hERG", "hERG", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("AMES", "AMES", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("CYP2C9_Veith", "CYP2C9-I", "binary", "AUPRC", "tdc_official"),
    TdcTaskSpec("CYP2D6_Veith", "CYP2D6-I", "binary", "AUPRC", "tdc_official"),
    TdcTaskSpec("CYP3A4_Veith", "CYP3A4-I", "binary", "AUPRC", "tdc_official"),
    TdcTaskSpec("CYP2C9_Substrate_CarbonMangels", "CYP2C9-S", "binary", "AUPRC", "tdc_official"),
    TdcTaskSpec("CYP2D6_Substrate_CarbonMangels", "CYP2D6-S", "binary", "AUPRC", "tdc_official"),
    TdcTaskSpec("CYP3A4_Substrate_CarbonMangels", "CYP3A4-S", "binary", "AUROC", "tdc_official"),
    TdcTaskSpec("LD50_Zhu", "LD50", "regression", "MAE", "MAE"),
    TdcTaskSpec("Caco2_Wang", "Caco-2", "regression", "MAE", "MAE"),
    TdcTaskSpec("PPBR_AZ", "PPBR", "regression", "MAE", "MAE"),
    TdcTaskSpec("Lipophilicity_AstraZeneca", "Lipophilicity", "regression", "MAE", "MAE"),
    # Added 2026-05-04 for EXP-055: present in EXP-049/050/051b's 23-task panel
    # but missing from the original EVAL-001 TDC list. PyTDC admet_group exposes
    # this as a regression task (aqueous solubility).
    TdcTaskSpec("Solubility_AqSolDB", "Solubility", "regression", "MAE", "MAE"),
    TdcTaskSpec("VDss_Lombardo", "VDss", "regression", "Spearman", "Spearman"),
    TdcTaskSpec("Half_Life_Obach", "Half-life", "regression", "Spearman", "Spearman"),
    TdcTaskSpec("Clearance_Microsome_AZ", "CL-micro", "regression", "Spearman", "Spearman"),
    TdcTaskSpec("Clearance_Hepatocyte_AZ", "CL-hepa", "regression", "Spearman", "Spearman"),
)

TDC_TASK_BY_NAME = {spec.name: spec for spec in TDC_TASKS}


def moleculenet_specs(raw_root: Path) -> tuple[MoleculeNetTaskSpec, ...]:
    processed = raw_root / "processed"
    return (
        MoleculeNetTaskSpec(
            dataset="BACE-S",
            task="bace_active",
            path=processed / "bace.csv",
            smiles_column="smiles",
            target_columns=("bace_active",),
            task_type="binary",
            metric="AUROC",
            split_variant="scaffold",
        ),
        MoleculeNetTaskSpec(
            dataset="BBBP-S",
            task="bbbp_permeable",
            path=processed / "bbbp.csv",
            smiles_column="smiles",
            target_columns=("bbbp_permeable",),
            task_type="binary",
            metric="AUROC",
            split_variant="scaffold",
        ),
        MoleculeNetTaskSpec(
            dataset="ClinTox",
            task="clintox",
            path=raw_root / "clintox.csv",
            smiles_column="smiles",
            target_columns=("FDA_APPROVED", "CT_TOX"),
            task_type="multilabel",
            metric="macro-AUROC",
            split_variant="random",
        ),
        MoleculeNetTaskSpec(
            dataset="SIDER",
            task="sider",
            path=raw_root / "sider.csv",
            smiles_column="smiles",
            target_columns=(
                "Hepatobiliary disorders",
                "Metabolism and nutrition disorders",
                "Product issues",
                "Eye disorders",
                "Investigations",
                "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders",
                "Social circumstances",
                "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions",
                "Endocrine disorders",
                "Surgical and medical procedures",
                "Vascular disorders",
                "Blood and lymphatic system disorders",
                "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders",
                "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders",
                "Psychiatric disorders",
                "Renal and urinary disorders",
                "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders",
                "Cardiac disorders",
                "Nervous system disorders",
                "Injury, poisoning and procedural complications",
            ),
            task_type="multilabel",
            metric="macro-AUROC",
            split_variant="random",
        ),
        MoleculeNetTaskSpec(
            dataset="Tox21",
            task="tox21",
            path=raw_root / "tox21.csv",
            smiles_column="smiles",
            target_columns=(
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ),
            task_type="multilabel",
            metric="macro-AUROC",
            split_variant="random",
        ),
        MoleculeNetTaskSpec(
            dataset="ESOL",
            task="solubility",
            path=processed / "esol.csv",
            smiles_column="smiles",
            target_columns=("solubility",),
            task_type="regression",
            metric="RMSE",
            split_variant="random",
        ),
        MoleculeNetTaskSpec(
            dataset="Lipophilicity",
            task="lipophilicity",
            path=processed / "lipophilicity.csv",
            smiles_column="smiles",
            target_columns=("lipophilicity",),
            task_type="regression",
            metric="RMSE",
            split_variant="random",
        ),
        MoleculeNetTaskSpec(
            dataset="FreeSolv",
            task="hydration_free_energy",
            path=processed / "free_solv.csv",
            smiles_column="smiles",
            target_columns=("hydration_free_energy",),
            task_type="regression",
            metric="RMSE",
            split_variant="random",
        ),
    )


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True)


def canonical_smiles_polar(smiles: str) -> str | None:
    """Canonical SMILES with the POLAR-flavour standardisation applied.

    Pipeline: SMILES → Mol → salt-strip (SaltRemover) → uncharge
    (rdMolStandardize.Uncharger) → canonical SMILES. Mirrors the standardisation
    `eval001_moleculenet_polar/` zarrs applied at build time, so a raw-CSV
    SMILES routed through this fn will match the SMILES stored in the POLAR
    zarr 1:1 (instead of dropping every salt / charged form).

    Use this when looking up raw-release SMILES against any zarr built with the
    POLAR pipeline (see `threedscriptors/data_handling/dataset_creation/generators/utils.py:standardize_mol`).
    """
    from threedscriptors.data_handling.dataset_creation.generators.utils import (
        standardize_mol,
    )

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = standardize_mol(mol, strip_salts=True, neutralize=True)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True)
    except Exception:
        return None


def load_moleculenet_raw(
    spec: MoleculeNetTaskSpec,
    canonicalizer: Callable[[str], str | None] = canonical_smiles,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Load a MoleculeNet raw release, canonicalising SMILES and dropping
    invalid/duplicate rows.

    `canonicalizer` defaults to plain RDKit canonicalisation (OFF24 lookup).
    Pass `canonical_smiles_polar` to match a POLAR-pipeline zarr (salt-strip +
    uncharge) so charged/salt forms are not silently dropped before the lookup.
    """
    df = pd.read_csv(spec.path)
    total_rows = len(df)
    cols = [spec.smiles_column, *spec.target_columns]
    df = df[cols].copy()
    df[spec.smiles_column] = df[spec.smiles_column].map(canonicalizer)
    df = df[df[spec.smiles_column].notna()].drop_duplicates(subset=[spec.smiles_column])

    smiles = df[spec.smiles_column].astype(str).tolist()
    y = df[list(spec.target_columns)].to_numpy(dtype=float)
    if spec.task_type != "multilabel":
        y = y.reshape(-1)

    counts = {
        "n_source_rows": int(total_rows),
        "n_valid_smiles": len(smiles),
        "n_invalid_or_duplicate": int(total_rows - len(smiles)),
    }
    return smiles, y, counts


def select_tdc_tasks(names: list[str] | None, limit: int | None = None) -> list[TdcTaskSpec]:
    specs = list(TDC_TASKS)
    if names:
        wanted = set(names)
        specs = [s for s in specs if s.name in wanted or s.short_name in wanted]
    if limit is not None:
        specs = specs[:limit]
    return specs


def select_moleculenet_tasks(
    raw_root: Path,
    names: list[str] | None,
    limit: int | None = None,
) -> list[MoleculeNetTaskSpec]:
    specs = list(moleculenet_specs(raw_root))
    if names:
        wanted = set(names)
        specs = [s for s in specs if s.dataset in wanted or s.task in wanted]
    if limit is not None:
        specs = specs[:limit]
    return specs
