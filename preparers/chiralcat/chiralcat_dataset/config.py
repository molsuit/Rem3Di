"""Pipeline configuration: one YAML file drives every stage.

All stage configs are pydantic models. Polymorphic entries (data sources, label
corrections) are Annotated unions discriminated on a ``kind`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field

from .taxonomy import normalize_class

# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #


class _BaseSource(BaseModel):
    path: str
    # Expected sha256 of the file; checked before it is read when set.
    sha256: str | None = None
    smiles_key: str = "SMILES"
    mol_key: str = "mol"


class FixedLabelSource(_BaseSource):
    """All molecules in the file share a single, fixed chiral class."""

    kind: Literal["fixed_label"]
    label: str


class TypedSource(_BaseSource):
    """Per-molecule class taken from a ``chiral type`` field in the file."""

    kind: Literal["typed"]
    type_key: str = "chiral type"
    drop_types: list[str] = Field(default_factory=lambda: ["unknown"])


SourceConfig = Annotated[FixedLabelSource | TypedSource, Field(discriminator="kind")]


# --------------------------------------------------------------------------- #
# Label corrections
# --------------------------------------------------------------------------- #


class RelabelCorrection(BaseModel):
    """Override a molecule's chiral class (keyed by canonical SMILES)."""

    kind: Literal["relabel"]
    smiles: str
    to_class: str
    reason: str = ""
    confidence: str = ""


class DeleteCorrection(BaseModel):
    """Remove a molecule from the dataset (keyed by canonical SMILES)."""

    kind: Literal["delete"]
    smiles: str
    reason: str = ""
    confidence: str = ""


class KeepCorrection(BaseModel):
    """Keep a molecule's label although the audit flags it (keyed by SMILES).

    The 3D stereocentre audit cannot perceive every kind of chirality: a P(III)
    phosphine, a selenoxide or a haptic organometallic is genuinely chiral but
    carries no tetrahedral tag RDKit can read off the geometry. Recording that
    review decision here keeps the audit quiet about molecules a curator has
    already judged, so a new finding actually means something.
    """

    kind: Literal["keep"]
    smiles: str
    reason: str = ""
    confidence: str = ""


Correction = Annotated[
    RelabelCorrection | DeleteCorrection | KeepCorrection,
    Field(discriminator="kind"),
]


class CorrectionSet(BaseModel):
    corrections: list[Correction] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> CorrectionSet:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw or {})


# --------------------------------------------------------------------------- #
# Stage configuration
# --------------------------------------------------------------------------- #


class StandardizeConfig(BaseModel):
    """RDKit standardization applied to each structure."""

    strip_salts: bool = True
    neutralize: bool = True

    @property
    def enabled(self) -> bool:
        return self.strip_salts or self.neutralize


class ExtractionConfig(BaseModel):
    """Stage 1: read the source pickles into labelled 3D structures."""

    data_dir: str = "data"
    sources: list[SourceConfig]
    conformer_id: int = 0
    add_hydrogens: bool = True
    standardize: StandardizeConfig = Field(default_factory=StandardizeConfig)


class ValidationConfig(BaseModel):
    """Stage 2: validate labels against 3D stereochemistry and correct them."""

    corrections_file: str | None = "corrections.yaml"
    # Re-run the RDKit stereocentre audit of the central class on every build.
    audit_central_stereo: bool = True
    # A central molecule with no 3D stereocentre that corrections.yaml does not
    # cover is a suspected mislabel: warn, or abort the build.
    on_uncovered_mislabel: Literal["warn", "error", "ignore"] = "warn"


class RepairConfig(BaseModel):
    """Stage 3: detect clashing geometries and repair what can be repaired."""

    enabled: bool = True
    embed_seed: int = 0xC0FFEE
    mmff_iters: int = 2000


class PinnedFile(BaseModel):
    """A pipeline input inside ``data_dir`` with its expected sha256."""

    path: str
    sha256: str | None = None


class OrganometallicConfig(BaseModel):
    """Stage 4: restore flagged organometallic complexes from frozen rebuilds.

    The complexes were assembled once with Architector and relaxed with xtb.
    Their geometries are read back from ``rebuilt_structures`` instead of being
    rebuilt on every run: Architector imports xtb-python and openbabel, neither
    of which has wheels for the Python this project runs on, and its build is
    not deterministic (28 to 30 of the 30 targets succeed from run to run).
    """

    enabled: bool = True
    target_class: str = "planar"
    rebuilt_structures: PinnedFile = Field(
        default_factory=lambda: PinnedFile(path="organometallic_rebuilds.extxyz")
    )


class OutputConfig(BaseModel):
    """Where the two datasets land."""

    directory: str = "output"
    dataset_index: str = "dataset.csv"
    dataset_extxyz: str = "dataset.extxyz"
    rejected_index: str = "rejected.csv"
    rejected_extxyz: str = "rejected.extxyz"
    run_report: str = "run_report.json"


class PipelineConfig(BaseModel):
    """The whole ingestion pipeline, from source pickles to the two datasets."""

    extraction: ExtractionConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    organometallic: OrganometallicConfig = Field(default_factory=OrganometallicConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Resolved at load time so every relative path in the config is interpreted
    # against the config file's own directory, not the current working dir.
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        path = Path(path).resolve()
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = cls.model_validate(raw)
        config.base_dir = path.parent
        config.validate_classes()
        return config

    def validate_classes(self) -> None:
        """Fail fast on an unknown class name anywhere in the config."""
        for source in self.extraction.sources:
            if isinstance(source, FixedLabelSource):
                normalize_class(source.label)
        normalize_class(self.organometallic.target_class)

    # --- resolved paths ---------------------------------------------------- #

    @property
    def data_dir(self) -> Path:
        return (self.base_dir / self.extraction.data_dir).resolve()

    @property
    def output_dir(self) -> Path:
        return (self.base_dir / self.output.directory).resolve()

    @property
    def rebuilt_structures_path(self) -> Path:
        return self.data_dir / self.organometallic.rebuilt_structures.path

    @property
    def corrections_path(self) -> Path | None:
        if self.validation.corrections_file is None:
            return None
        return (self.base_dir / self.validation.corrections_file).resolve()

    def load_corrections(self) -> CorrectionSet:
        path = self.corrections_path
        if path is None:
            return CorrectionSet()
        return CorrectionSet.from_yaml(path)
