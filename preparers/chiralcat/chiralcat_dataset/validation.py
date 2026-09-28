"""Stage 2: validate the labels against the molecules' actual 3D stereochemistry.

This stage keeps ``corrections.yaml`` honest. It re-runs the RDKit stereocentre
audit over the ``central`` class and reports any molecule the audit believes is
mislabelled that the file does not account for, so a new mislabel surfaces on
every build instead of passing silently.

It also owns the ``keep`` corrections. The other two kinds belong to extraction:
``relabel`` and ``delete`` change a molecule's class or remove it, so they have
to land before deduplication picks a winner, and they key on the raw SMILES that
stage sees. A ``keep`` changes nothing — it records that a curator reviewed the
molecule and accepted the label — so it is applied here, against the
standardized SMILES the audit reports and the dataset ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from .chemistry import canonical_smiles, count_stereocenters_from_3d
from .config import KeepCorrection, PipelineConfig
from .records import CorrectionRecord, StereoAuditRecord, Structure


class UncoveredMislabelError(RuntimeError):
    """Raised when the audit finds a suspected mislabel and policy says to fail."""


@dataclass
class ValidationOutput:
    audit: list[StereoAuditRecord] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def audit_central_stereo(structures: list[Structure]) -> list[StereoAuditRecord]:
    """Read every ``central`` structure's 3D geometry for tetrahedral stereo.

    ``AssignStereochemistryFrom3D`` is the authoritative test: it reads the
    conformer rather than the SMILES graph, so it catches stereocentres defined
    only in 3D and settles the organometallic and heteroatom cases.
    """
    records: list[StereoAuditRecord] = []
    for structure in structures:
        if structure.class_name != "central":
            continue
        try:
            if structure.mol is None:
                raise ValueError("no in-memory mol carried from extraction")
            assigned, tagged = count_stereocenters_from_3d(Chem.Mol(structure.mol))
            error = ""
        except Exception as exc:
            assigned, tagged, error = -1, -1, type(exc).__name__
        records.append(
            StereoAuditRecord(
                smiles=structure.smiles,
                source_file=structure.source_file,
                assigned_rs_3d=assigned,
                tetrahedral_tagged_3d=tagged,
                error=error,
            )
        )
    return records


def validate(config: PipelineConfig, structures: list[Structure]) -> ValidationOutput:
    """Audit the central class and report suspected mislabels not yet reviewed.

    Also applies the ``keep`` corrections. Unlike ``relabel`` and ``delete``,
    which extraction applies to the raw SMILES, a ``keep`` is an annotation on
    the finished structure: it changes no label, it only records that a curator
    looked at this molecule and accepted it. It therefore keys on the
    standardized SMILES, which is what the audit reports and what ships.
    """
    output = ValidationOutput()

    # The keep annotations are applied whether or not the audit runs: they are a
    # record of a review decision, not an artefact of the check that prompted it.
    keep_map: dict[str, KeepCorrection] = {}
    for correction in config.load_corrections().corrections:
        if not isinstance(correction, KeepCorrection):
            continue
        key = canonical_smiles(correction.smiles)
        if key is None:
            raise ValueError(f"Correction SMILES does not parse: {correction.smiles!r}")
        keep_map[key] = correction

    matched: set[str] = set()
    for structure in structures:
        correction = keep_map.get(structure.smiles)
        if correction is None:
            continue
        matched.add(structure.smiles)
        note = f"keep {structure.class_name} (reviewed)"
        structure.correction = (
            f"{structure.correction}; {note}" if structure.correction else note
        )
        output.corrections.append(
            CorrectionRecord(
                structure.smiles,
                "keep",
                structure.class_name,
                structure.class_name,
                correction.confidence,
                correction.reason,
                True,
            )
        )

    # A stale `keep` is surfaced the same way a stale relabel/delete is: the
    # molecule it names is gone, so the entry should be removed from the file.
    for key, correction in keep_map.items():
        if key in matched:
            continue
        output.corrections.append(
            CorrectionRecord(
                correction.smiles,
                "keep",
                "",
                "",
                correction.confidence,
                correction.reason,
                False,
            )
        )

    if not config.validation.audit_central_stereo:
        output.counts = {"kept_by_review": len(matched), "audited": 0}
        return output

    output.audit = audit_central_stereo(structures)

    # `central` with no tetrahedral stereocentre anywhere in the geometry is the
    # signature of a mislabel: the molecule carries no central chirality at all.
    output.uncovered = [
        record.smiles
        for record in output.audit
        if not record.error
        and not record.has_stereocenter
        and record.smiles not in matched
    ]

    output.counts = {
        "audited": len(output.audit),
        "with_stereocenter": sum(1 for r in output.audit if r.has_stereocenter),
        "no_stereocenter": sum(
            1 for r in output.audit if not r.error and not r.has_stereocenter
        ),
        "perception_failed": sum(1 for r in output.audit if r.error),
        "kept_by_review": len(matched),
        "uncovered_mislabels": len(output.uncovered),
    }

    policy = config.validation.on_uncovered_mislabel
    if output.uncovered and policy == "error":
        listed = "\n  ".join(output.uncovered[:20])
        raise UncoveredMislabelError(
            f"{len(output.uncovered)} central molecule(s) have no 3D stereocentre "
            f"and are not covered by corrections.yaml:\n  {listed}"
        )
    return output
