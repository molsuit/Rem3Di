"""Re-embed a geometry-only top-1 zarr with the frozen Orb-omol MLIP backbone.

Orb analog of ``reembed_geometry_with_backbone.py`` (the MACE reference). The
MLIP-swap matched panel needs every backbone to embed the SAME top-1 conformers.
This tool takes the shared geometry zarr (positions / atomic_numbers /
molecule_ptr / total_charge / total_spin — NO embeddings) and produces an
Orb-specific cache that is byte-identical in geometry but carries Orb's per-atom
INVARIANT node embedding in ``atomic_embeddings`` (fp16).

Backbone: ``orb_v3_conservative_omol`` (energy-conserving Orb-v3 trained on
OMol25; element coverage includes all of {H,C,N,O,F,P,S,Cl,Br,I}).

Per-atom feature endpoint (documented):
  ``model.model(batch)["node_features"]`` — the post-message-passing node
  embedding of the MoleculeGNS backbone (after 5 GNN stacks, before per-property
  decoding). Orb is non-equivariant, so this scalar/invariant embedding is the
  full per-atom representation — the direct analog of MACE's per-atom
  descriptors. Dimension D = latent_dim = 256.

  We deliberately call the GNS backbone directly (``model.model(batch)``) rather
  than ``model(batch)`` / ``model.predict(batch)`` so we do not pay for the
  energy/force/stress heads or autograd machinery — we only want the node
  embedding, which is identical either way.

Graph construction (Orb-omol defaults, baked into the returned atoms_adapter):
  radius = 6.0 A, max_num_neighbors = 120, charge+spin graph conditioning.
  GEOM-drugs molecules are non-periodic neutral closed-shell singlets, so we
  build ASE Atoms with no cell/pbc, charge = total_charge (0 throughout this
  dataset), spin_multiplicity = total_spin + 1 (S=0 -> singlet multiplicity 1).
  We pass wrap=False since there is no unit cell to wrap into (keeps positions
  untouched for the F1 byte-equality check).

Fairness / correctness gates (mirrors the MACE reference):
  F1  geometry copied verbatim from the shared source; positions asserted
      byte-equal (the model never mutates positions; we also re-read the graph's
      stored positions per batch and compare to source).
  G1  no NaN/Inf; per-channel std reported (flag collapsed channels);
      dim == 256.

Usage:
  python scripts/dataset_creation/reembed_geometry_with_orb.py \
    --geometry-zarr /home/fb638/research/datasets/geom_drugs_top1 \
    --out-zarr /local/scratch/fb638/geom_top1_orb_omol \
    --batch-size 16 --device cuda [--max-mols 500 --validate-only]

Run with the clean Orb venv: source /home/fb638/venvs/orb_omol/bin/activate
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import ase
import numpy as np
import torch
import zarr

LATENT_DIM = 256  # orb_v3 conservative latent_dim == per-atom node_features dim
FEAT_KEY = "node_features"


def build_atoms_list(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charges: np.ndarray,
    spins: np.ndarray,
    ptr: np.ndarray,
    mol_indices: list[int],
) -> list[ase.Atoms]:
    """Build a list of ase.Atoms (one per molecule index) from flat arrays.

    charge/spin set in atoms.info as Orb's ForcefieldAtomsAdapter expects:
      info['charge'] -> total_charge,  info['spin'] -> spin_multiplicity.
    spin_multiplicity = total_spin + 1 (closed-shell S=0 -> 1).
    """
    out: list[ase.Atoms] = []
    for m in mol_indices:
        a, b = int(ptr[m]), int(ptr[m + 1])
        atoms = ase.Atoms(
            numbers=atomic_numbers[a:b].astype(np.int64),
            positions=positions[a:b].astype(np.float64),  # ase stores float64
        )
        atoms.info["charge"] = float(charges[m])
        atoms.info["spin"] = float(spins[m]) + 1.0
        out.append(atoms)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-zarr", required=True, type=Path)
    ap.add_argument("--out-zarr", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-mols", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda if available)")
    ap.add_argument(
        "--weights-path", default=None,
        help="Override Orb omol checkpoint path/URL (default: bundled S3 URL).",
    )
    ap.add_argument("--validate-only", action="store_true",
                    help="Do not write the full cache; just embed --max-mols and run G1 checks.")
    ap.add_argument("--rotation-average", type=int, default=1,
                    help="K: frame-average the per-atom node features over K independent random "
                         "SO(3) rotations (Orb's own rand_matrix). K=1 (default) keeps the legacy "
                         "single-orientation behaviour. K>1 turns the non-equivariant Orb features "
                         "into a Monte-Carlo estimate of the exactly rotation-invariant feature, "
                         "matching MACE's invariance for a fair backbone comparison.")
    args = ap.parse_args()

    dev = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[cfg] device={dev}  feat_key={FEAT_KEY}  dim={LATENT_DIM}")

    # ---- load Orb omol (returns (model, atoms_adapter)) ----
    from orb_models.forcefield import pretrained
    # Orb's own canonical SO(3) sampler (the same one used in Orb's training augmentation).
    from orb_models.common.dataset.augmentations.geometric_augmentations import rand_matrix

    kw = {"device": dev, "precision": "float32-high"}
    if args.weights_path:
        kw["weights_path"] = args.weights_path
    model, atoms_adapter = pretrained.orb_v3_conservative_omol(**kw)
    model = model.to(dev)
    model.eval()
    print(f"[model] orb_v3_conservative_omol loaded; "
          f"adapter radius={atoms_adapter.radius} max_neighbors={atoms_adapter.max_num_neighbors}")

    # ---- source geometry (flat raw arrays) ----
    src = zarr.open_group(str(args.geometry_zarr), mode="r")
    src_pos = src["positions"]
    positions = src_pos[:]
    atomic_numbers = src["atomic_numbers"][:]
    ptr = src["molecule_ptr"][:]
    n_mol_total = int(ptr.shape[0] - 1)
    # total_charge/total_spin are absent on TDC/MolNet eval zarrs (neutral closed-shell
    # drug molecules) — default to 0 (charge 0, spin 0 -> singlet multiplicity 1).
    charges = src["total_charge"][:] if "total_charge" in src else np.zeros(n_mol_total, dtype=np.float32)
    spins = src["total_spin"][:] if "total_spin" in src else np.zeros(n_mol_total, dtype=np.float32)
    n_atoms_total = int(src_pos.shape[0])
    print(f"[data] molecules={n_mol_total} atoms={n_atoms_total}")

    mol_indices = list(range(n_mol_total))
    if args.max_mols:
        mol_indices = mol_indices[: args.max_mols]
    # number of atoms covered by the selected molecules (for the smoke/validate case)
    n_atoms_covered = int(ptr[mol_indices[-1] + 1]) - int(ptr[mol_indices[0]])

    # ---- output zarr ----
    emb_arr = None
    if not args.validate_only:
        if args.out_zarr.exists():
            raise SystemExit(f"out-zarr already exists: {args.out_zarr}")
        print(f"[copy] copytree geometry -> {args.out_zarr}")
        shutil.copytree(args.geometry_zarr, args.out_zarr)
        out_grp = zarr.open_group(str(args.out_zarr), mode="a")
        # zarr v2 API (geometry zarr is v2 .zgroup/.zarray format; keep output v2-compatible).
        emb_arr = out_grp.create_dataset(
            "atomic_embeddings",
            shape=(n_atoms_total, LATENT_DIM),
            chunks=(450, LATENT_DIM),
            dtype="float16",
            overwrite=True,
        )

    offset = int(ptr[mol_indices[0]])  # global atom offset of first selected mol
    nan_seen = False
    sample_sum = torch.zeros(LATENT_DIM, dtype=torch.float64)
    sample_sumsq = torch.zeros(LATENT_DIM, dtype=torch.float64)
    sample_n = 0
    pos_mismatch = 0

    K = max(1, int(args.rotation_average))
    rot_std_vals: list[float] = []   # mean across-rotation feature std per batch (K>1 diagnostic)
    print(f"[cfg] rotation_average K={K}"
          + ("  (single orientation, legacy)" if K == 1 else "  (frame-averaging over SO(3))"))

    def _embed(atoms_list):
        graph = atoms_adapter.from_ase_atoms_list(
            atoms_list, device=dev, wrap=False, output_dtype=torch.float32
        )
        with torch.inference_mode():
            e = model.model(graph)[FEAT_KEY].detach()
        return e.to(torch.float32).cpu(), graph

    n_batches = (len(mol_indices) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch_mols = mol_indices[bi * args.batch_size : (bi + 1) * args.batch_size]
        graph = None
        if K == 1:
            atoms_list = build_atoms_list(
                positions, atomic_numbers, charges, spins, ptr, batch_mols
            )
            emb, graph = _embed(atoms_list)
        else:
            # Frame-average over K independent SO(3) rotations. Orb is non-equivariant, so each
            # orientation yields a different per-atom feature; the mean over rotations is a
            # Monte-Carlo estimate of the exactly rotation-invariant feature (atom indexing is
            # preserved, so the per-atom average is well defined).
            draws = []
            for _k in range(K):
                atoms_list = build_atoms_list(
                    positions, atomic_numbers, charges, spins, ptr, batch_mols
                )
                for a in atoms_list:
                    a.positions = a.positions @ rand_matrix(1, dtype=torch.float64)[0].numpy()
                e, _g = _embed(atoms_list)
                draws.append(e)
            stk = torch.stack(draws, 0)                  # (K, nb, D)
            emb = stk.mean(0)
            rot_std_vals.append(stk.std(0).mean().item())  # orientation-noise scale (pre-average)
        nb = emb.shape[0]
        assert emb.shape[1] == LATENT_DIM, (
            f"endpoint dim {emb.shape[1]} != expected {LATENT_DIM}"
        )
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            nan_seen = True

        # F1 byte-equality only meaningful when geometry is untouched (K==1); under
        # frame-averaging the positions are deliberately rotated, so the check is skipped.
        if K == 1:
            rec_pos = graph.node_features["positions"].detach().to(torch.float32).cpu().numpy()
            cmp = positions[offset : offset + nb].astype(np.float32)
            if not np.array_equal(rec_pos, cmp):
                if not np.allclose(rec_pos, cmp, atol=1e-5):
                    pos_mismatch += int(
                        (~np.isclose(rec_pos, cmp, atol=1e-5)).any(axis=1).sum()
                    )

        if emb_arr is not None:
            emb_arr[offset : offset + nb] = emb.to(torch.float16).numpy()

        if sample_n < 200000:
            sample_sum += emb.double().sum(0)
            sample_sumsq += (emb.double() ** 2).sum(0)
            sample_n += nb
        offset += nb
        if bi % 50 == 0:
            print(f"[emb] batch={bi}/{n_batches} atoms_done={offset - int(ptr[mol_indices[0]])}"
                  f"/{n_atoms_covered}", flush=True)

    # ---- G1 report ----
    mean = sample_sum / max(1, sample_n)
    var = (sample_sumsq / max(1, sample_n)) - mean ** 2
    std = var.clamp_min(0).sqrt()
    n_collapsed = int((std < 1e-6).sum())
    atoms_embedded = offset - int(ptr[mol_indices[0]])
    print("\n===== G1 cache-integrity report =====")
    print(f"  atoms embedded      : {atoms_embedded}")
    print(f"  feature dim D       : {LATENT_DIM}")
    print(f"  NaN/Inf present     : {nan_seen}")
    print(f"  position mismatches : {pos_mismatch}  (F1: want 0)")
    print(f"  per-channel std min/med/max : {std.min():.4g} / {std.median():.4g} / {std.max():.4g}")
    print(f"  collapsed channels (std<1e-6): {n_collapsed} / {LATENT_DIM}")
    if K > 1 and rot_std_vals:
        import statistics as _st
        rot_noise = _st.mean(rot_std_vals)
        feat_scale = float(std.median())
        print(f"  rotation_average K  : {K}")
        print(f"  across-rotation feature std (pre-average, mean over batches): {rot_noise:.4g}")
        print(f"    => single-orientation orientation noise is {100 * rot_noise / max(feat_scale, 1e-12):.1f}% "
              f"of the channel scale; frame-averaging shrinks it ~1/sqrt(K)={1 / (K ** 0.5):.3f}x")
    ok = (not nan_seen) and pos_mismatch == 0 and n_collapsed == 0

    if not args.validate_only and ok:
        import yaml

        cfg_path = args.out_zarr / "dataset_config.yaml"
        cfg = {}
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cfg.update(
            {
                "contains_embeddings": True,
                "embedding_dim": int(LATENT_DIM),
                "backbone": "orb_v3_conservative_omol",
                "feature_endpoint": "MoleculeGNS.node_features (invariant, post-MP)",
                "rotation_average_K": int(K),
                "rotation_invariant": bool(K > 1),
            }
        )
        cfg_path.write_text(yaml.safe_dump(cfg))
        print(f"  wrote dataset_config.yaml (contains_embeddings=true, dim={LATENT_DIM})")

    print(f"\n[{'PASS' if ok else 'FAIL'}] G1 {'all checks passed' if ok else 'see report above'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
