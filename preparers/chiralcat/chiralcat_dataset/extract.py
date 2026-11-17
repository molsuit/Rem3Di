"""Stage 1: read the source pickles into labelled, standardized 3D structures.

Label corrections from ``corrections.yaml`` are applied here, because they key
on the raw canonical SMILES (before standardization) and must take effect before
deduplication decides which occurrence of a molecule wins.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from rdkit import Chem

from .chemistry import (
    add_hydrogens,
    canonical_smiles,
    conformer_coords,
    pick_conformer_id,
    single_conformer,
    standardize_mol,
)
from .config import (
    DeleteCorrection,
    ExtractionConfig,
    PipelineConfig,
    RelabelCorrection,
    TypedSource,
)
from .records import BROKEN, FILTERED, CorrectionRecord, RejectedRecord, Structure
from .sources import iter_source
from .taxonomy import CLASS_TO_LABEL, normalize_class


@dataclass
class ExtractionOutput:
    structures: list[Structure] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    # SMILES -> original (multi-fragment, pre-standardization) source mol.
    # The repair stage needs it to strip counter-ions without re-reading pickles.
    source_lookup: dict[str, Chem.Mol] = field(default_factory=dict, repr=False)

    def count(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1


def _standardized_graph_smiles(mol: Chem.Mol, standardize) -> str | None:
    """The SMILES the pipeline would store for this mol, ignoring geometry."""
    working = Chem.Mol(mol)
    working.RemoveAllConformers()
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(working)
    if standardize.enabled:
        standardized = standardize_mol(
            working,
            strip_salts=standardize.strip_salts,
            neutralize=standardize.neutralize,
        )
        if standardized is None:
            return None
        working = standardized
        with contextlib.suppress(Exception):
            Chem.SanitizeMol(working)
    try:
        return Chem.MolToSmiles(working)
    except Exception:
        return None


def extract(config: PipelineConfig) -> ExtractionOutput:
    """Run stage 1 and return usable structures plus everything rejected."""
    extraction: ExtractionConfig = config.extraction
    data_dir = config.data_dir
    output = ExtractionOutput()

    drop_lookup = {
        source.path: set(source.drop_types)
        for source in extraction.sources
        if isinstance(source, TypedSource)
    }

    # `relabel` and `delete` are extraction-time decisions: they change a
    # molecule's class or remove it, so they must land before dedup picks a
    # winner. They key on the raw canonical SMILES, which is what this stage
    # sees. `keep` is a validation-stage annotation and is applied there, keyed
    # on the standardized SMILES the audit actually reports.
    relabel_map: dict[str, RelabelCorrection] = {}
    delete_map: dict[str, DeleteCorrection] = {}
    for correction in config.load_corrections().corrections:
        key = canonical_smiles(correction.smiles)
        if key is None:
            raise ValueError(f"Correction SMILES does not parse: {correction.smiles!r}")
        if isinstance(correction, RelabelCorrection):
            normalize_class(correction.to_class)  # validate target class
            relabel_map[key] = correction
        elif isinstance(correction, DeleteCorrection):
            delete_map[key] = correction

    matched: set[str] = set()
    seen: set[str] = set()  # dedup on the raw canonical SMILES
    seen_standardized: set[str] = set()  # second net, after standardization
    next_index = 0

    def reject(
        smiles: str,
        class_name: str,
        source_path: str,
        reason: str,
        disposition: str,
        detail: str = "",
    ) -> None:
        output.count(reason)
        output.rejected.append(
            RejectedRecord(
                smiles=smiles,
                class_name=class_name,
                label=CLASS_TO_LABEL.get(class_name, -1),
                source_file=source_path,
                stage="extract",
                reason=reason,
                disposition=disposition,
                detail=detail,
            )
        )

    for source in extraction.sources:
        drop_types = drop_lookup.get(source.path, set())
        for raw_smiles, raw_class, mol in iter_source(source, data_dir):
            if raw_class in drop_types:
                reject(raw_smiles, "", source.path, "dropped_type", FILTERED, raw_class)
                continue
            class_name = normalize_class(raw_class)

            if mol is None:
                reject(raw_smiles, class_name, source.path, "no_mol", BROKEN)
                continue

            canonical = canonical_smiles(raw_smiles)
            if canonical is None:
                reject(raw_smiles, class_name, source.path, "bad_smiles", BROKEN)
                continue
            if canonical in seen:
                reject(canonical, class_name, source.path, "duplicate", FILTERED)
                continue

            # Index the source mol for the repair stage's counter-ion strip.
            if mol.GetNumConformers():
                for key in (
                    canonical,
                    _standardized_graph_smiles(mol, extraction.standardize),
                ):
                    if key is not None and key not in output.source_lookup:
                        output.source_lookup[key] = mol

            # --- validation + correction (manual overrides) ------------------
            if canonical in delete_map:
                matched.add(canonical)
                seen.add(canonical)
                correction = delete_map[canonical]
                output.corrections.append(
                    CorrectionRecord(
                        canonical,
                        "delete",
                        class_name,
                        "",
                        correction.confidence,
                        correction.reason,
                        True,
                    )
                )
                reject(
                    canonical,
                    class_name,
                    source.path,
                    "correction_delete",
                    FILTERED,
                    correction.reason,
                )
                continue

            correction_note = ""
            if canonical in relabel_map:
                matched.add(canonical)
                correction = relabel_map[canonical]
                new_class = normalize_class(correction.to_class)
                output.corrections.append(
                    CorrectionRecord(
                        canonical,
                        "relabel",
                        class_name,
                        new_class,
                        correction.confidence,
                        correction.reason,
                        True,
                    )
                )
                output.count("relabeled")
                correction_note = f"relabel {class_name}->{new_class}"
                class_name = new_class

            if mol.GetNumConformers() == 0:
                reject(canonical, class_name, source.path, "no_coords", BROKEN)
                continue

            conformer_id = pick_conformer_id(mol, extraction.conformer_id)
            working = single_conformer(mol, conformer_id)

            if extraction.standardize.enabled:
                working = standardize_mol(
                    working,
                    strip_salts=extraction.standardize.strip_salts,
                    neutralize=extraction.standardize.neutralize,
                )
                if working is None or working.GetNumAtoms() == 0:
                    reject(
                        canonical,
                        class_name,
                        source.path,
                        "standardize_emptied",
                        BROKEN,
                    )
                    continue
                with contextlib.suppress(Exception):
                    Chem.SanitizeMol(working)

            out_smiles = (
                Chem.MolToSmiles(working) if extraction.standardize.enabled else canonical
            )
            if extraction.standardize.enabled and out_smiles in seen_standardized:
                reject(
                    out_smiles,
                    class_name,
                    source.path,
                    "duplicate_after_standardize",
                    FILTERED,
                )
                continue

            if extraction.add_hydrogens:
                working = add_hydrogens(working)

            coords = conformer_coords(working, 0)
            symbols = [atom.GetSymbol() for atom in working.GetAtoms()]

            # Dummy / wildcard atoms (R-group attachment points) are not real
            # elements: they break ASE and do not describe a complete molecule.
            if "*" in symbols:
                reject(out_smiles, class_name, source.path, "dummy_atom", BROKEN)
                continue

            # An atom pinned at the exact origin signals an unrecoverable
            # coordinate failure (real coordinates are never exactly zero).
            if len(coords) > 1 and any(
                abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6 for x, y, z in coords
            ):
                reject(out_smiles, class_name, source.path, "bad_geometry", BROKEN)
                continue

            seen.add(canonical)
            seen_standardized.add(out_smiles)
            output.structures.append(
                Structure(
                    index=next_index,
                    smiles=out_smiles,
                    class_name=class_name,
                    label=CLASS_TO_LABEL[class_name],
                    symbols=symbols,
                    coords=coords,
                    source_file=source.path,
                    correction=correction_note,
                    mol=working,
                )
            )
            next_index += 1

    # Any correction whose SMILES never matched is surfaced rather than ignored:
    # it usually means a typo, or a molecule dropped further upstream.
    for key, correction in {**relabel_map, **delete_map}.items():
        if key in matched:
            continue
        action = "relabel" if isinstance(correction, RelabelCorrection) else "delete"
        to_class = (
            correction.to_class if isinstance(correction, RelabelCorrection) else ""
        )
        output.corrections.append(
            CorrectionRecord(
                correction.smiles,
                action,
                "",
                to_class,
                correction.confidence,
                correction.reason,
                False,
            )
        )
        output.count("correction_unmatched")

    output.counts["kept"] = len(output.structures)
    return output
