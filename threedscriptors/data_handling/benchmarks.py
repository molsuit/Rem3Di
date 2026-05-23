"""Pydantic registry of the benchmark panel (MoleculeNet + TDC ADMET).

This replaces the junior eval001 frozen-dataclass / ``Literal`` specs with the
package idiom: pydantic models, an ``Annotated`` discriminated union on
``source``, and reuse of the canonical :class:`TaskConfig` / :class:`TaskSet`
vocabulary. One registry, consumed by both the dataset-creation generators
(which materialize the split into the zarr) and the descriptor evaluation.

Multi-label MoleculeNet datasets (ClinTox/SIDER/Tox21) are modelled as several
``classification`` :class:`BenchmarkTask` columns with a dataset-level
``macro-AUROC`` metric; the package ``TaskType`` only distinguishes
regression vs classification, and column count carries single- vs multi-label.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from threedscriptors.data_handling.dataset.tasks import (
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


class SplitVariant(StrEnum):
    # Deterministic DeepChem Bemis-Murcko scaffold split (eval001 splits.py).
    scaffold = "scaffold"
    # PyTDC admet_group official split (fixed scaffold test; seeded train/valid).
    tdc_default = "tdc_default"
    random = "random"


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


Benchmark = Annotated[
    MoleculeNetBenchmark | TdcBenchmark, Field(discriminator="source")
]


def _clf(name: str, source_column: str | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        name=name, source_column=source_column, task_type=TaskType.classification
    )


def _reg(name: str, source_column: str | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        name=name, source_column=source_column, task_type=TaskType.regression
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
    TdcBenchmark(dataset_id="BBB_Martins", tdc_name="BBB_Martins",
                 tasks=[_clf("BBB")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="HIA_Hou", tdc_name="HIA_Hou",
                 tasks=[_clf("HIA")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="Pgp_Broccatelli", tdc_name="Pgp_Broccatelli",
                 tasks=[_clf("Pgp")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="Bioavailability_Ma", tdc_name="Bioavailability_Ma",
                 tasks=[_clf("Bioavailability")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="DILI", tdc_name="DILI",
                 tasks=[_clf("DILI")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="hERG", tdc_name="hERG",
                 tasks=[_clf("hERG")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="AMES", tdc_name="AMES",
                 tasks=[_clf("AMES")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="CYP2C9_Veith", tdc_name="CYP2C9_Veith",
                 tasks=[_clf("CYP2C9-I")], metric=EvalMetric.auprc),
    TdcBenchmark(dataset_id="CYP2D6_Veith", tdc_name="CYP2D6_Veith",
                 tasks=[_clf("CYP2D6-I")], metric=EvalMetric.auprc),
    TdcBenchmark(dataset_id="CYP3A4_Veith", tdc_name="CYP3A4_Veith",
                 tasks=[_clf("CYP3A4-I")], metric=EvalMetric.auprc),
    TdcBenchmark(dataset_id="CYP2C9_Substrate_CarbonMangels",
                 tdc_name="CYP2C9_Substrate_CarbonMangels",
                 tasks=[_clf("CYP2C9-S")], metric=EvalMetric.auprc),
    TdcBenchmark(dataset_id="CYP2D6_Substrate_CarbonMangels",
                 tdc_name="CYP2D6_Substrate_CarbonMangels",
                 tasks=[_clf("CYP2D6-S")], metric=EvalMetric.auprc),
    TdcBenchmark(dataset_id="CYP3A4_Substrate_CarbonMangels",
                 tdc_name="CYP3A4_Substrate_CarbonMangels",
                 tasks=[_clf("CYP3A4-S")], metric=EvalMetric.auroc),
    TdcBenchmark(dataset_id="LD50_Zhu", tdc_name="LD50_Zhu",
                 tasks=[_reg("LD50")], metric=EvalMetric.mae),
    TdcBenchmark(dataset_id="Caco2_Wang", tdc_name="Caco2_Wang",
                 tasks=[_reg("Caco-2")], metric=EvalMetric.mae),
    TdcBenchmark(dataset_id="PPBR_AZ", tdc_name="PPBR_AZ",
                 tasks=[_reg("PPBR")], metric=EvalMetric.mae),
    TdcBenchmark(dataset_id="Lipophilicity_AstraZeneca",
                 tdc_name="Lipophilicity_AstraZeneca",
                 tasks=[_reg("Lipophilicity")], metric=EvalMetric.mae),
    TdcBenchmark(dataset_id="Solubility_AqSolDB",
                 tdc_name="Solubility_AqSolDB",
                 tasks=[_reg("Solubility")], metric=EvalMetric.mae),
    TdcBenchmark(dataset_id="VDss_Lombardo", tdc_name="VDss_Lombardo",
                 tasks=[_reg("VDss")], metric=EvalMetric.spearman),
    TdcBenchmark(dataset_id="Half_Life_Obach", tdc_name="Half_Life_Obach",
                 tasks=[_reg("Half-life")], metric=EvalMetric.spearman),
    TdcBenchmark(dataset_id="Clearance_Microsome_AZ",
                 tdc_name="Clearance_Microsome_AZ",
                 tasks=[_reg("CL-micro")], metric=EvalMetric.spearman),
    TdcBenchmark(dataset_id="Clearance_Hepatocyte_AZ",
                 tdc_name="Clearance_Hepatocyte_AZ",
                 tasks=[_reg("CL-hepa")], metric=EvalMetric.spearman),
)


_BY_ID: dict[str, MoleculeNetBenchmark | TdcBenchmark] = {
    b.dataset_id: b for b in (*MOLECULENET_BENCHMARKS, *TDC_BENCHMARKS)
}


def get_benchmark(dataset_id: str) -> MoleculeNetBenchmark | TdcBenchmark:
    try:
        return _BY_ID[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark {dataset_id!r}. "
            f"Known: {sorted(_BY_ID)}"
        ) from exc


def select_benchmarks(
    benchmarks: tuple[MoleculeNetBenchmark | TdcBenchmark, ...],
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[MoleculeNetBenchmark | TdcBenchmark]:
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
    source: Literal["moleculenet", "tdc"]

    @classmethod
    def from_benchmark(
        cls, benchmark: MoleculeNetBenchmark | TdcBenchmark
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
