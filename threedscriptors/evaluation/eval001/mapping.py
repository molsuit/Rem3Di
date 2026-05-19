from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from threedscriptors.data_handling.dataset_creation.generators.utils import (
    MACE_OFF_ELEMENTS,
    MACE_POLAR_ELEMENTS,
    filter_mol,
    standardize_mol,
)
from threedscriptors.evaluation.eval001.datasets import canonical_smiles


def is_polar_standardisation(config: dict, *, auto_path: str, auto_token: str) -> bool:
    """Whether the POLAR (salt-strip + uncharge) standardisation applies.

    Explicit `config["standardisation"]` of ``"polar"`` / ``"off24"`` wins. With
    ``"auto"`` (or unset) it falls back to the legacy path-substring inference,
    so existing invocations behave identically while a collaborator on custom
    paths can pin the standardisation explicitly instead of depending on a
    magic substring being present in their directory name.
    """
    mode = str(config.get("standardisation") or "auto").lower()
    if mode == "polar":
        return True
    if mode == "off24":
        return False
    return auto_token in str(auto_path).lower()


def tdc_mapping_for_task(config: dict, task_name: str):
    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

    zarr_path = Path(config["paths"]["tdc_root"]) / task_name / "zarr"
    # PORT NOTE (macepolar): open_existing_dataset_from_dir has no load_smiles
    # kwarg on this branch; it loads SMILES storage when the files are present.
    ds = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    exact = {}
    for i, smi in enumerate(ds.get_smiles_per_structure()):
        if smi and smi not in exact:
            exact[str(smi)] = i
    canon = {}
    for smi, idx in exact.items():
        c = canonical_smiles(smi)
        if c is not None and c not in canon:
            canon[c] = idx
    return exact, canon


def map_tdc_smiles(smiles: pd.Series, exact: dict[str, int], canon: dict[str, int]):
    indices = []
    mask = []
    for smi in smiles.astype(str).tolist():
        idx = exact.get(smi)
        if idx is None:
            c = canonical_smiles(smi)
            if c is not None:
                idx = canon.get(c)
        if idx is None:
            mask.append(False)
        else:
            indices.append(idx)
            mask.append(True)
    return np.asarray(indices, dtype=int), np.asarray(mask, dtype=bool)


def load_tdc_index_mapping(config: dict, task_name: str) -> dict | None:
    mapping_path = Path(config["paths"]["tdc_root"]) / task_name / "tdc_index_mapping.json"
    if not mapping_path.exists():
        return None
    return json.loads(mapping_path.read_text())


def tdc_source_to_zarr(mapping: dict) -> dict[int, int]:
    return {int(source_idx): zarr_idx for zarr_idx, source_idx in enumerate(mapping["accepted_tdc_indices"])}


def _is_polar_tdc_root(config: dict) -> bool:
    return is_polar_standardisation(
        config, auto_path=config["paths"].get("tdc_root", ""), auto_token="tdc_admet_polar"
    )


def load_moleculenet_source_to_zarr(out_root: Path, dataset: str) -> dict[str, int]:
    """Load the persisted raw-SMILES → zarr-row map for a MoleculeNet dataset."""
    path = out_root / dataset / "source_index_mapping.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for canonical MoleculeNet cached evaluation."
        )
    mapping = json.loads(path.read_text())
    source_to_zarr = mapping.get("source_smiles_to_zarr")
    if not isinstance(source_to_zarr, dict):
        raise KeyError(f"{path} is missing source_smiles_to_zarr")
    return {str(k): int(v) for k, v in source_to_zarr.items()}


def reconstruct_moleculenet_source_to_zarr(
    out_root: Path,
    dataset: str,
    raw_smiles: list[str],
) -> dict[str, int]:
    """Rebuild the raw-SMILES → zarr-row map from the zarr's stored SMILES.

    Counterpart of `reconstruct_tdc_source_to_zarr` for MoleculeNet. The zarr's
    ``isomeric_smiles.txt`` is already the post-standardisation SMILES, so the
    caller passes `raw_smiles` already routed through the matching canonicaliser
    (`canonical_smiles` / `canonical_smiles_polar`); we just intersect them.
    """
    smiles_path = out_root / dataset / "zarr" / "isomeric_smiles.txt"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"{smiles_path} is required to reconstruct stale MoleculeNet mapping."
        )
    zarr_smiles = smiles_path.read_text().splitlines()
    zarr_lookup: dict[str, int] = {}
    for idx, smi in enumerate(zarr_smiles):
        zarr_lookup.setdefault(str(smi), int(idx))
    return {
        str(smi): zarr_lookup[str(smi)]
        for smi in raw_smiles
        if str(smi) in zarr_lookup
    }


def _standardized_tdc_key(row, *, polar_mode: bool, max_atoms: int = 100) -> tuple[str, float] | None:
    mol = Chem.MolFromSmiles(str(row["Drug"]))
    mol = standardize_mol(mol, strip_salts=polar_mode, neutralize=polar_mode)
    allowed = MACE_POLAR_ELEMENTS if polar_mode else MACE_OFF_ELEMENTS
    # PORT NOTE (macepolar): the upgraded filter_mol defaults allow_charged /
    # allow_radicals to True to keep the existing generators bit-identical, so
    # EVAL-001 opts into strict MACE-OFF24 / MACE-POLAR semantics explicitly.
    if not filter_mol(
        mol,
        max_atoms=max_atoms,
        allowed_elements=allowed,
        allow_charged=False,
        allow_radicals=False,
    ):
        return None
    iso = Chem.MolToSmiles(Chem.RemoveAllHs(mol), isomericSmiles=True, canonical=True)
    return (iso, float(row["Y"]))


def reconstruct_tdc_source_to_zarr(
    config: dict,
    task_name: str,
    source_frame: pd.DataFrame,
    *,
    max_atoms: int = 100,
    max_unmatched_frac: float = 0.02,
) -> dict[int, int]:
    """Map original TDC source row positions to actual zarr row positions.

    `tdc_index_mapping.json` records rows accepted by the generator, but some
    rows can still fail in later construction stages such as conformer
    generation. Reconstructing from actual zarr rows prevents out-of-bounds
    indices and keeps coverage tied to the artifact we are scoring.

    A few zarr rows can be unmatchable for benign reasons (float32 target
    round-trip at the tolerance edge, a curated duplicate). Up to
    `max_unmatched_frac` of them are dropped (→ honest coverage < 1.0) instead
    of raising and silently switching the whole task to a different mapping
    method; only a systemic mismatch above that fraction raises.
    """

    from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset

    polar_mode = _is_polar_tdc_root(config)
    candidates: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for source_idx, (_, row) in enumerate(source_frame.reset_index(drop=True).iterrows()):
        key = _standardized_tdc_key(row, polar_mode=polar_mode, max_atoms=max_atoms)
        if key is not None:
            smi, y = key
            candidates[smi].append((source_idx, y))

    zarr_path = Path(config["paths"]["tdc_root"]) / task_name / "zarr"
    # PORT NOTE (macepolar): open_existing_dataset_from_dir has no load_smiles
    # kwarg on this branch; it loads SMILES storage when the files are present.
    ds = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    zarr_smiles = [str(s) for s in ds.get_smiles_per_structure()]
    zarr_targets = np.asarray(ds.targets_system[:]).reshape(-1)

    source_to_zarr: dict[int, int] = {}
    missing_keys: list[tuple[int, tuple[str, float]]] = []
    for zarr_idx, (smi, y) in enumerate(zip(zarr_smiles, zarr_targets, strict=True)):
        y_float = float(y)
        options = candidates.get(smi, [])
        if not options:
            missing_keys.append((zarr_idx, (smi, y_float)))
            continue
        best_i, (source_idx, source_y) = min(
            enumerate(options), key=lambda item: abs(item[1][1] - y_float)
        )
        # Targets persist in the zarr as float32, so match Y on a scale-aware
        # tolerance. A fixed 1e-3 absolute bound falsely rejects large-magnitude
        # targets (PPBR, Clearance, Half-life) once float32 ULP exceeds 1e-3,
        # while still being loose enough to cross-match near-duplicate small
        # targets. np.isclose tracks float32 precision across all TDC scales.
        if not np.isclose(source_y, y_float, rtol=1e-5, atol=1e-5):
            missing_keys.append((zarr_idx, (smi, y_float)))
            continue
        del options[best_i]
        source_to_zarr[int(source_idx)] = int(zarr_idx)

    n_zarr = max(len(zarr_smiles), 1)
    if missing_keys and len(missing_keys) / n_zarr > max_unmatched_frac:
        sample = "; ".join(f"{i}:{key[0]}:{key[1]}" for i, key in missing_keys[:3])
        raise KeyError(
            f"{task_name}: could not reconstruct {len(missing_keys)}/{len(zarr_smiles)} "
            f"zarr rows from official TDC source rows "
            f"(> {max_unmatched_frac:.0%} unmatched, systemic mismatch). "
            f"Examples: {sample}"
        )
    if missing_keys:
        print(
            f"{task_name}: reconstruct dropped {len(missing_keys)}/{len(zarr_smiles)} "
            f"zarr rows within the {max_unmatched_frac:.0%} tolerance; "
            "coverage is reported < 1.0 (coverage_limited), not silently re-mapped."
        )
    return source_to_zarr


def _tdc_row_key(row) -> tuple[str, str, str]:
    return (str(row["Drug_ID"]), str(row["Drug"]), repr(float(row["Y"])))


def map_tdc_frames_to_source_indices(
    frames: list[pd.DataFrame],
    source_frame: pd.DataFrame,
    offset: int = 0,
) -> list[np.ndarray]:
    """Recover source row indices for multiple PyTDC split frames at once.

    Mapping train and validation independently can reuse the first row of a
    duplicated (Drug_ID, Drug, Y) key. Building the source queues once and
    consuming them across all frames preserves disjoint split membership.
    """

    queues: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    for pos, (_, row) in enumerate(source_frame.reset_index(drop=True).iterrows()):
        queues[_tdc_row_key(row)].append(offset + pos)

    mapped: list[np.ndarray] = []
    for frame in frames:
        out: list[int] = []
        for _, row in frame.reset_index(drop=True).iterrows():
            key = _tdc_row_key(row)
            if not queues[key]:
                raise KeyError(f"Could not map PyTDC row back to source split: {key}")
            out.append(queues[key].popleft())
        mapped.append(np.asarray(out, dtype=int))
    return mapped


def map_source_indices_to_zarr(source_indices: np.ndarray, source_to_zarr: dict[int, int]):
    """Map source-row indices to zarr-row indices, returning (indices, keep_mask).

    Backbone-neutral: callers with a SMILES-keyed map first build an
    ``{raw_index: zarr_index}`` dict and pass integer indices here.
    """
    zarr_indices: list[int] = []
    mask: list[bool] = []
    for source_idx in np.asarray(source_indices, dtype=int).tolist():
        zarr_idx = source_to_zarr.get(int(source_idx))
        if zarr_idx is None:
            mask.append(False)
        else:
            zarr_indices.append(int(zarr_idx))
            mask.append(True)
    return np.asarray(zarr_indices, dtype=int), np.asarray(mask, dtype=bool)
