# Using an Orb backbone

Rem3Di's MACE path runs the foundation model **online**, recomputing features every
training step. Orb does not fit that path, so it is supported through a **precomputed
cache** instead: embed the geometry once with Orb, write the per-atom features into the
zarr, and train reading from that cache.

This is how the published Orb results were produced, so it needs no re-validation.

## Why precomputed rather than online

The online path is built on `torch-sim` `SimState` — per-system centering, cell padding,
charge and spin plumbing — none of which Orb consumes; it wants
`atoms_adapter.from_ase_atoms_list(...)`. Supporting Orb online would mean a second,
parallel state path with a CPU-side per-molecule loop inside every training step, and
frame averaging would then cost K forward passes *per batch* rather than K passes once.

## Producing the cache

```bash
python scripts/dataset_creation/reembed_geometry_with_orb.py \
    --src  /path/to/geometry_only_zarr \
    --dst  /path/to/orb_embedded_zarr \
    --batch-size 64
```

The source must be a geometry-only store (positions, atomic_numbers, molecule_ptr,
total_charge, total_spin — no embeddings). The output is byte-identical in geometry and
adds Orb's per-atom invariant node embedding in `atomic_embeddings` as fp16.

Backbone: `orb_v3_conservative_omol`. Features are taken from
`model.model(batch)["node_features"]` — the post-message-passing node embedding of the
GNS backbone, dimension **256**. The GNS backbone is called directly rather than through
`model(batch)`, to skip the energy/force/stress heads; the node embedding is identical
either way.

## Frame averaging

`--rotation-average K` averages the node features over K random SO(3) rotations using
Orb's own sampler.

This matters because **Orb's per-atom latents are not rotation-invariant.** Re-embedding
a rotated molecule shifts them by roughly 67% of per-channel scale, where MACE scalars are
exactly O(3)-invariant. A single-orientation Orb cache therefore carries orientation noise
that the MACE backbones do not.

Measured effect, evaluating a single-orientation-trained checkpoint on K=16 averaged
features: classification AUROC **0.726 → 0.766**, against a MACE band of 0.768–0.774.

⚠️ **Retraining on averaged features does not work.** Pretraining val loss drops to
1.1×10⁻⁴, better than any MACE backbone, but downstream collapses to AUROC 0.615 — worse
than the unaveraged baseline. Averaging smooths the features enough that the denoising
pretext task becomes trivial, so nothing is forced through the descriptor bottleneck.
Reconstruction quality is not descriptor quality. Use frame averaging at inference, not to
build a training cache.

## Architecture config differences

An Orb config differs from a MACE one in three ways:

```yaml
embedding_preprocess_config:
  input_irreps: 256x0e            # Orb node features are invariant scalars only
  invariant_projection_dim: 256   # native Orb width; no up-projection
  pseudoscalars: false            # no l=1 features, so no chiral branch
```

**`pseudoscalars: false` is not optional.** The chiral encoder builds pseudoscalars from
l≥1 channels, which Orb does not produce. An Orb-backbone Rem3Di therefore ships without
the chirality contribution.

The `mace_config` block is still required by the schema but is a **dummy** on this path —
it supplies `r_max` for the pair graph and nothing else. `_run_mace` is never called,
because the preprocessor takes the cached-embeddings branch.

## Requirements

Orb comes from `orb-models`, which is a separate install and is not a declared dependency
of `remedi`. Install it in the environment you use to build the cache; it is not needed at
training time, since training only reads the zarr.
