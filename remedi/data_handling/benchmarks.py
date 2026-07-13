"""Pydantic registry of the benchmark panel (MoleculeNet + TDC ADMET + Polaris).

This replaces the junior eval001 frozen-dataclass / ``Literal`` specs with the
package idiom: pydantic models, an ``Annotated`` discriminated union on
``source``, and reuse of the canonical :class:`TaskConfig` / :class:`TaskSet`
vocabulary. One registry, consumed by both the dataset-creation generators
(which materialize the split into the zarr) and the descriptor evaluation.

Multi-label MoleculeNet datasets (ClinTox/SIDER/Tox21) are modelled as several
``classification`` :class:`BenchmarkTask` columns with a dataset-level
``macro-AUROC`` metric; the package ``TaskType`` only distinguishes
regression vs classification, and column count carries single- vs multi-label.

Polaris datasets are loaded indirectly: ``polaris-lib`` pins ``zarr<3`` whereas
the rest of this project requires ``zarr>=3.2`` for the MoleculeDataset store,
so the two cannot share a venv. The workaround is a standalone PEP 723 dump
script (``scripts/dataset_download/dump_polaris.py``) that runs in its own
ephemeral env, fetches a polaris dataset, and writes a standardized parquet
(columns ``smiles``, ``split``, plus one column per task) under
``polaris_raw_root``. :class:`PolarisBenchmark` entries here describe that
parquet; the build pipeline reads it via ``PolarisOfflineGenerator`` without
ever importing ``polaris``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)


class EvalMetric(StrEnum):
    rmse = "RMSE"
    mae = "MAE"
    r2 = "R2"
    spearman = "Spearman"
    auroc = "AUROC"
    auprc = "AUPRC"
    macro_auroc = "macro-AUROC"
    # Single-label multi-class metrics (TaskType.multiclass). The probe emits a
    # (N, n_classes) softmax; balanced-accuracy / macro-F1 argmax it, macro-OvR
    # AUROC consumes the probabilities. Balanced metrics are the headline for
    # class-imbalanced multiclass tasks (e.g. chiral_cat).
    balanced_accuracy = "balanced-accuracy"
    macro_f1 = "macro-F1"
    macro_auroc_ovr = "macro-AUROC-OvR"
    # Pairwise enantiomer ranking accuracy (chiral_docking). The probe regresses
    # the per-conformer docking ``top_score``; conformer predictions are pooled
    # per stereoisomer, the two enantiomers of each constitution are paired, and
    # the metric is the fraction of pairs whose predicted better-docker matches
    # the true one (ties scored 0.5). Needs the molecule/stereoisomer grouping,
    # so it is evaluated by a dedicated path, not via ``metric_for``.
    pair_ranking_accuracy = "pair-ranking-accuracy"


class SplitVariant(StrEnum):
    # Deterministic DeepChem Bemis-Murcko scaffold split (eval001 splits.py).
    scaffold = "scaffold"
    # PyTDC admet_group official split (fixed scaffold test; seeded train/valid).
    tdc_default = "tdc_default"
    # Polaris ``Set`` column (Train / Valid / Test labels per row), materialized
    # by the standalone dump script into the parquet's ``split`` uint8 codes.
    polaris_set = "polaris_set"
    random = "random"
    # Group-aware, class-stratified split (splits.stratified_group_split):
    # groups (non-isomeric SMILES) are kept whole, class proportions preserved.
    stratified = "stratified"
    # Literature split supplied verbatim by the source (e.g. the Chiro docking
    # dataset ships separate train/valid/test pickle files); the generator writes
    # the per-row code straight into the zarr ``split`` column.
    predefined = "predefined"


class BenchmarkTask(BaseModel):
    """One scored target column within a benchmark dataset."""

    # Canonical task name, stored in the dataset TaskSet / zarr.
    name: str
    # Column name in the raw source table (defaults to ``name``).
    source_column: str | None = None
    task_type: TaskType

    @property
    def column(self) -> str:
        return self.source_column if self.source_column is not None else self.name

    def to_task_config(self) -> TaskConfig:
        return TaskConfig(
            name=self.name, task_type=self.task_type, scope=TaskScope.system
        )


class _BenchmarkBase(BaseModel):
    # Stable id; also the per-dataset zarr subdirectory name.
    dataset_id: str
    tasks: list[BenchmarkTask]
    # Official/panel headline metric for the dataset.
    metric: EvalMetric
    split_variant: SplitVariant

    def task_set(self) -> TaskSet:
        return TaskSet.from_list([t.to_task_config() for t in self.tasks])


class MoleculeNetBenchmark(_BenchmarkBase):
    source: Literal["moleculenet"] = "moleculenet"
    # Raw release CSV under the MoleculeNet raw root.
    csv_name: str
    smiles_column: str = "smiles"
    split_variant: SplitVariant = SplitVariant.scaffold


class TdcBenchmark(_BenchmarkBase):
    source: Literal["tdc"] = "tdc"
    # PyTDC admet_group benchmark name.
    tdc_name: str
    split_variant: SplitVariant = SplitVariant.tdc_default


class PolarisBenchmark(_BenchmarkBase):
    source: Literal["polaris"] = "polaris"
    # Polaris-hub slug. Consumed by ``scripts/dataset_download/dump_polaris.py``
    # to fetch the dataset; not used at build time (the build pipeline reads
    # the standardized parquet only).
    polaris_slug: str
    # Filename within ``polaris_raw_root`` of the standardized parquet dumped
    # by the polaris fetch script (columns: ``smiles``, ``split``, one per
    # task). Defaults to ``<dataset_id>.parquet``.
    parquet_name: str | None = None
    split_variant: SplitVariant = SplitVariant.polaris_set

    @property
    def parquet_filename(self) -> str:
        return self.parquet_name or f"{self.dataset_id}.parquet"


class LocalXyzBenchmark(_BenchmarkBase):
    """A benchmark whose 3D structures + labels come from a local extended-XYZ
    file (one frame per structure, labels on the comment line). Unlike the
    SMILES sources, geometries are supplied directly — the build pipeline skips
    conformer generation and ingests the coordinates verbatim (the QM9 path)."""

    source: Literal["local"] = "local"
    # Filename of the extended-XYZ dump within the build config's
    # ``local_raw_root``.
    xyz_filename: str
    # ``atoms.info`` key carrying the integer class label per frame.
    label_key: str = "label"
    # ``atoms.info`` key carrying the reference SMILES per frame.
    smiles_key: str = "smiles"
    split_variant: SplitVariant = SplitVariant.stratified


class ChiroDockingBenchmark(_BenchmarkBase):
    """Chiro small-enantiomer docking-ranking benchmark (3D supplied directly).

    Built from three pandas ``.pkl`` DataFrames (one per literature split) of
    RDKit Mols carrying a single 3D conformer plus a molecule-level docking
    ``top_score``. Like :class:`LocalXyzBenchmark` the geometries are ingested
    verbatim (no conformer generation), but the task is **pairwise**: the two
    enantiomers of each constitution are ranked by predicted ``top_score``, so
    the headline metric is :attr:`EvalMetric.pair_ranking_accuracy` and the eval
    runner dispatches to the dedicated pairwise evaluator. The single regression
    column holds the per-conformer ``top_score`` the probe is trained on.
    """

    source: Literal["local_chiro"] = "local_chiro"
    # Source pickle filenames within the build config's local raw root.
    train_file: str
    valid_file: str
    test_file: str
    # Raw DataFrame column names.
    id_column: str = "ID"
    nonstereo_column: str = "SMILES_nostereo"
    mol_column: str = "rdkit_mol_cistrans_stereo"
    score_column: str = "top_score"
    # Conformers ingested per stereoisomer (first-N in file order); None keeps all.
    max_conformers_per_stereoisomer: int | None = 2
    metric: EvalMetric = EvalMetric.pair_ranking_accuracy
    split_variant: SplitVariant = SplitVariant.predefined


Benchmark = Annotated[
    MoleculeNetBenchmark
    | TdcBenchmark
    | PolarisBenchmark
    | LocalXyzBenchmark
    | ChiroDockingBenchmark,
    Field(discriminator="source"),
]


def _clf(name: str, source_column: str | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        name=name, source_column=source_column, task_type=TaskType.classification
    )


def _reg(name: str, source_column: str | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        name=name, source_column=source_column, task_type=TaskType.regression
    )


def _mcls(name: str, source_column: str | None = None) -> BenchmarkTask:
    """A single-label multi-class column (TaskType.multiclass)."""
    return BenchmarkTask(
        name=name, source_column=source_column, task_type=TaskType.multiclass
    )


# --- MoleculeNet (10 datasets: junior's 8 + HIV + BACE-pIC50) --------------
# Raw release CSVs are read from the per-machine ``moleculenet_raw_root`` set
# in the build config. The ``csv_name`` field is the filename within that
# directory; ``source_column`` carries the per-task target column name from
# the original MoleculeNet release.

_SIDER_COLUMNS = (
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
)

_TOX21_COLUMNS = (
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
)

MOLECULENET_BENCHMARKS: tuple[MoleculeNetBenchmark, ...] = (
    # Regression
    MoleculeNetBenchmark(
        dataset_id="esol",
        csv_name="esol.csv",
        tasks=[_reg("solubility", "measured log solubility in mols per litre")],
        metric=EvalMetric.rmse,
    ),
    MoleculeNetBenchmark(
        dataset_id="freesolv",
        csv_name="free_solv.csv",
        tasks=[_reg("hydration_free_energy", "expt")],
        metric=EvalMetric.rmse,
    ),
    MoleculeNetBenchmark(
        dataset_id="lipophilicity",
        csv_name="lipophilicity.csv",
        tasks=[_reg("lipophilicity", "exp")],
        metric=EvalMetric.rmse,
    ),
    MoleculeNetBenchmark(
        dataset_id="bace_pic50",
        csv_name="bace.csv",
        smiles_column="mol",
        tasks=[_reg("bace_pIC50", "pIC50")],
        metric=EvalMetric.rmse,
    ),
    # Classification
    MoleculeNetBenchmark(
        dataset_id="bace",
        csv_name="bace.csv",
        smiles_column="mol",
        tasks=[_clf("bace_active", "Class")],
        metric=EvalMetric.auroc,
    ),
    MoleculeNetBenchmark(
        dataset_id="bbbp",
        csv_name="BBBP.csv",
        tasks=[_clf("bbbp_permeable", "p_np")],
        metric=EvalMetric.auroc,
    ),
    MoleculeNetBenchmark(
        dataset_id="hiv",
        csv_name="HIV.csv",
        tasks=[_clf("hiv_active", "HIV_active")],
        metric=EvalMetric.auroc,
    ),
    MoleculeNetBenchmark(
        dataset_id="clintox",
        csv_name="clintox.csv",
        tasks=[_clf("FDA_APPROVED"), _clf("CT_TOX")],
        metric=EvalMetric.macro_auroc,
    ),
    MoleculeNetBenchmark(
        dataset_id="sider",
        csv_name="sider.csv",
        tasks=[_clf(c) for c in _SIDER_COLUMNS],
        metric=EvalMetric.macro_auroc,
    ),
    MoleculeNetBenchmark(
        dataset_id="tox21",
        csv_name="tox21.csv",
        tasks=[_clf(c) for c in _TOX21_COLUMNS],
        metric=EvalMetric.macro_auroc,
    ),
)


# --- TDC ADMET (22 tasks; PyTDC admet_group official splits) ---------------

TDC_BENCHMARKS: tuple[TdcBenchmark, ...] = (
    # BBB_Martins (TDC) was dropped: same molecule list as MoleculeNet `bbbp`,
    # so it was inflating the eval panel without adding chemistry.
    TdcBenchmark(
        dataset_id="HIA_Hou",
        tdc_name="HIA_Hou",
        tasks=[_clf("HIA")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="Pgp_Broccatelli",
        tdc_name="Pgp_Broccatelli",
        tasks=[_clf("Pgp")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="Bioavailability_Ma",
        tdc_name="Bioavailability_Ma",
        tasks=[_clf("Bioavailability")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="DILI",
        tdc_name="DILI",
        tasks=[_clf("DILI")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="hERG",
        tdc_name="hERG",
        tasks=[_clf("hERG")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="AMES",
        tdc_name="AMES",
        tasks=[_clf("AMES")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="CYP2C9_Veith",
        tdc_name="CYP2C9_Veith",
        tasks=[_clf("CYP2C9-I")],
        metric=EvalMetric.auprc,
    ),
    TdcBenchmark(
        dataset_id="CYP2D6_Veith",
        tdc_name="CYP2D6_Veith",
        tasks=[_clf("CYP2D6-I")],
        metric=EvalMetric.auprc,
    ),
    TdcBenchmark(
        dataset_id="CYP3A4_Veith",
        tdc_name="CYP3A4_Veith",
        tasks=[_clf("CYP3A4-I")],
        metric=EvalMetric.auprc,
    ),
    # CYP{2C9,2D6}_Substrate_CarbonMangels were dropped: the three
    # CarbonMangels substrate panels share their molecule list (~99% scaffold
    # overlap, 662-666 mols each) and only differ in which CYP isoform label
    # they carry. Keeping CYP3A4 alone for the chem-space coverage view.
    TdcBenchmark(
        dataset_id="CYP3A4_Substrate_CarbonMangels",
        tdc_name="CYP3A4_Substrate_CarbonMangels",
        tasks=[_clf("CYP3A4-S")],
        metric=EvalMetric.auroc,
    ),
    TdcBenchmark(
        dataset_id="LD50_Zhu",
        tdc_name="LD50_Zhu",
        tasks=[_reg("LD50")],
        metric=EvalMetric.mae,
    ),
    TdcBenchmark(
        dataset_id="Caco2_Wang",
        tdc_name="Caco2_Wang",
        tasks=[_reg("Caco-2")],
        metric=EvalMetric.mae,
    ),
    TdcBenchmark(
        dataset_id="PPBR_AZ",
        tdc_name="PPBR_AZ",
        tasks=[_reg("PPBR")],
        metric=EvalMetric.mae,
    ),
    # Lipophilicity_AstraZeneca (TDC) was dropped: same molecule list as
    # MoleculeNet `lipophilicity`.
    TdcBenchmark(
        dataset_id="Solubility_AqSolDB",
        tdc_name="Solubility_AqSolDB",
        tasks=[_reg("Solubility")],
        metric=EvalMetric.mae,
    ),
    TdcBenchmark(
        dataset_id="VDss_Lombardo",
        tdc_name="VDss_Lombardo",
        tasks=[_reg("VDss")],
        metric=EvalMetric.spearman,
    ),
    TdcBenchmark(
        dataset_id="Half_Life_Obach",
        tdc_name="Half_Life_Obach",
        tasks=[_reg("Half-life")],
        metric=EvalMetric.spearman,
    ),
    TdcBenchmark(
        dataset_id="Clearance_Microsome_AZ",
        tdc_name="Clearance_Microsome_AZ",
        tasks=[_reg("CL-micro")],
        metric=EvalMetric.spearman,
    ),
    TdcBenchmark(
        dataset_id="Clearance_Hepatocyte_AZ",
        tdc_name="Clearance_Hepatocyte_AZ",
        tasks=[_reg("CL-hepa")],
        metric=EvalMetric.spearman,
    ),
)


# --- Polaris (ASAP / Biogen ADME-Fang) -------------------------------------
# Loaded from per-dataset parquet dumps under the build config's
# ``polaris_raw_root``; see module docstring for the rationale and
# ``scripts/dataset_download/dump_polaris.py`` for how to refresh them
# (single invocation, dumps every curated slug). Task column names match
# polaris's source columns one-for-one (the dump script preserves them) and
# double as the canonical TaskSet column names.
#
# Adding a polaris dataset:
#   1. Append the (dataset_id, slug, smiles_column) row to ``_DATASETS`` in
#      ``scripts/dataset_download/dump_polaris.py``.
#   2. ``uv run scripts/dataset_download/dump_polaris.py --out-root <root>``
#      then ``python -c "import pandas as pd;
#      print(pd.read_parquet('<root>/<id>.parquet').columns.tolist())"`` to
#      read off the source columns.
#   3. Add a ``PolarisBenchmark`` entry below with the task columns
#      transcribed verbatim into ``BenchmarkTask.name`` (``_reg`` for
#      regression columns, ``_clf`` for classification).
#
POLARIS_BENCHMARKS: tuple[PolarisBenchmark, ...] = (
    PolarisBenchmark(
        dataset_id="polaris_antiviral_admet",
        polaris_slug="asap-discovery/antiviral-admet-2025-unblinded",
        tasks=[
            _reg("LogD"),
            _reg("HLM"),
            _reg("MLM"),
            _reg("KSOL"),
            _reg("MDR1-MDCKII"),
        ],
        metric=EvalMetric.mae,
    ),
    PolarisBenchmark(
        dataset_id="polaris_antiviral_potency",
        polaris_slug="asap-discovery/antiviral-potency-2025-unblinded",
        tasks=[
            _reg("pIC50 (MERS-CoV Mpro)"),
            _reg("pIC50 (SARS-CoV-2 Mpro)"),
        ],
        metric=EvalMetric.mae,
    ),
    PolarisBenchmark(
        dataset_id="polaris_adme_fang",
        polaris_slug="biogen/adme-fang-v1",
        # Real biogen parquet column names (the polarishub.io UI shows
        # human-readable labels with units; the underlying columns are
        # underscored). Confirmed by inspecting the dump.
        # LOG_HPPB / LOG_RPPB are very sparse (~5% non-null over 3521 rows);
        # the per-cell mask handles missingness but expect small effective
        # train sets for those two tasks.
        # No "Set" column upstream -> PolarisOfflineGenerator applies a
        # scaffold-split fallback at build time.
        tasks=[
            _reg("LOG_HLM_CLint"),
            _reg("LOG_RLM_CLint"),
            _reg("LOG_MDR1-MDCK_ER"),
            _reg("LOG_SOLUBILITY"),
            _reg("LOG_HPPB"),
            _reg("LOG_RPPB"),
        ],
        metric=EvalMetric.mae,
    ),
    PolarisBenchmark(
        dataset_id="polaris_pkis2_subset",
        polaris_slug="polaris/drewry2017-pkis2-subset-v2",
        # PKIS2 kinase %-inhibition subset (Drewry 2017). Five kinase
        # readouts; SMILES column is MOL_smiles.
        tasks=[
            _reg("KIT"),
            _reg("LOK"),
            _reg("RET"),
            _reg("SLK"),
            _reg("EGFR"),
        ],
        metric=EvalMetric.mae,
    ),
    PolarisBenchmark(
        dataset_id="polaris_pkis2_subset_cls",
        polaris_slug="polaris/drewry2017-pkis2-subset-v2",
        # Binary active/inactive labels thresholded from the regression
        # %-inhibition columns above (active ~= >=80% inhibition; CLS_KIT=1
        # iff KIT>=threshold etc.). Reuses the same dump as
        # polaris_pkis2_subset; ``parquet_name`` makes the build pipeline
        # read the same on-disk file rather than expecting a second dump.
        parquet_name="polaris_pkis2_subset.parquet",
        tasks=[
            _clf("CLS_KIT"),
            _clf("CLS_LOK"),
            _clf("CLS_RET"),
            _clf("CLS_SLK"),
            _clf("CLS_EGFR"),
        ],
        metric=EvalMetric.macro_auroc,
    ),
)


# --- Local extended-XYZ datasets (3D structures supplied directly) ----------
# Built by ``scripts/dataset_creation/build_chiral_cat.py`` (the QM9-style 3D
# path: no conformer generation). ChiralCat is single-label 5-class chirality-
# type classification over 17,023 molecules; balanced-accuracy is the headline
# given the extreme class imbalance (achiral/central dominate, helical/planar
# are rare). See ``/p/scratch/mace/wedig1/raw_datasets/chiral_cat/DATASET.md``.
LOCAL_BENCHMARKS: tuple[LocalXyzBenchmark, ...] = (
    LocalXyzBenchmark(
        dataset_id="chiral_cat",
        xyz_filename="chiral_structures.extxyz",
        tasks=[_mcls("chirality_type")],
        metric=EvalMetric.balanced_accuracy,
    ),
)


# --- Chiro docking (pairwise enantiomer ranking; 3D supplied directly) ------
# Built by ``scripts/dataset_creation/build_chiro_docking.py``. Source pickles:
# /p/scratch/mace/wedig1/raw_datasets/chiral_chiro_datasets (see its
# DATASETS_DESCRIPTION.md). Each constitution has exactly two enantiomers; the
# probe regresses the per-conformer docking top_score and the pairwise evaluator
# ranks the pair. ``margin3`` filtering upstream guarantees every pair has a
# clear (>=0.3 kcal/mol) winner.
CHIRO_BENCHMARKS: tuple[ChiroDockingBenchmark, ...] = (
    ChiroDockingBenchmark(
        dataset_id="chiral_docking",
        train_file="train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl",
        valid_file="validation_small_enantiomers_stable_full_screen_docking_MOL_margin3_49878_10368_5184.pkl",
        test_file="test_small_enantiomers_stable_full_screen_docking_MOL_margin3_50571_10368_5184.pkl",
        tasks=[_reg("docking_top_score", "top_score")],
    ),
)


_BY_ID: dict[str, Benchmark] = {
    b.dataset_id: b
    for b in (
        *MOLECULENET_BENCHMARKS,
        *TDC_BENCHMARKS,
        *POLARIS_BENCHMARKS,
        *LOCAL_BENCHMARKS,
        *CHIRO_BENCHMARKS,
    )
}


def get_benchmark(
    dataset_id: str,
) -> Benchmark:
    try:
        return _BY_ID[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark {dataset_id!r}. " f"Known: {sorted(_BY_ID)}"
        ) from exc


def select_benchmarks(
    benchmarks: tuple[Benchmark, ...],
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[Benchmark]:
    out = list(benchmarks)
    if ids:
        wanted = set(ids)
        out = [b for b in out if b.dataset_id in wanted]
    if limit is not None:
        out = out[:limit]
    return out


# --- Per-zarr eval-time sidecar --------------------------------------------
# Written next to each prepared zarr at ingest. Eval discovers benchmarks by
# walking a root and reading this manifest + the zarr's existing
# ``dataset_config.yaml`` (TaskSet). The eval reader never imports the
# registry — each zarr is self-describing.

BENCHMARK_MANIFEST_FILENAME = "benchmark_manifest.yaml"


class BenchmarkManifest(BaseModel):
    """Eval-time metadata sidecar for a prepared benchmark zarr."""

    dataset_id: str
    metric: EvalMetric
    split_variant: SplitVariant
    # Provenance only; the eval reader does not branch on this.
    source: Literal["moleculenet", "tdc", "polaris", "local", "local_chiro"]

    @classmethod
    def from_benchmark(
        cls,
        benchmark: Benchmark,
    ) -> BenchmarkManifest:
        return cls(
            dataset_id=benchmark.dataset_id,
            metric=benchmark.metric,
            split_variant=benchmark.split_variant,
            source=benchmark.source,
        )

    def to_zarr_dir(self, path: Path) -> None:
        import pydantic_yaml as pyd_yaml

        pyd_yaml.to_yaml_file(Path(path) / BENCHMARK_MANIFEST_FILENAME, self)

    @classmethod
    def from_zarr_dir(cls, path: Path) -> BenchmarkManifest:
        import pydantic_yaml as pyd_yaml

        return pyd_yaml.parse_yaml_file_as(
            cls, Path(path) / BENCHMARK_MANIFEST_FILENAME
        )


def discover_benchmark_zarrs(root: Path) -> list[tuple[Path, BenchmarkManifest]]:
    """Walk ``root``, yielding ``(zarr_path, manifest)`` for every subdirectory
    that carries a benchmark manifest. Directories without one are silently
    skipped so the eval root may hold unrelated zarrs.
    """
    out: list[tuple[Path, BenchmarkManifest]] = []
    root = Path(root)
    if not root.exists():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sub / BENCHMARK_MANIFEST_FILENAME).exists():
            out.append((sub, BenchmarkManifest.from_zarr_dir(sub)))
    return out
