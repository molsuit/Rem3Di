---
library_name: remedi
tags:
  - chemistry
  - molecular-descriptors
  - drug-discovery
  - admet
  - mace
license: other
license_name: rem3di-weights
license_link: LICENSE
---

# Rem3Di — MACE-POLAR descriptor encoder

A learned 3D molecular descriptor. Rem3Di takes per-atom features from a frozen
atomistic foundation potential and pools them into a single fixed-length vector per
molecule that varies smoothly with 3D structure and is sensitive to chirality.

Paper: [arXiv:2607.19977](https://arxiv.org/abs/2607.19977) ·
Code: [molsuit/Rem3Di](https://github.com/molsuit/Rem3Di) ·
Docs: [molsuit.github.io/Rem3Di](https://molsuit.github.io/Rem3Di/)

## ⚠️ Read this before downloading

**These weights do not contain a MACE model, and are useless without one.** The
backbone is a *runtime dependency*: Rem3Di calls a frozen MACE-POLAR-1-M at inference
and pools its output. You download MACE separately from
[ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations) and accept its
licence directly from its authors.

**MACE-POLAR-1 and MACE-OFF are distributed under the ASL (Academic Software Licence):
academic use only, no commercial use.** So although the Rem3Di code is Apache-2.0 and
no MACE weights are redistributed here, *running this model requires a licence you must
obtain yourself*, and in practice that makes this checkpoint academic-use-only. If you
need an unencumbered option, an Orb-v3 backbone (Apache-2.0) variant is planned — see
the limitations section for what you trade away.

## Usage

```python
from remedi.evaluation.benchmark.descriptors import RemediCalculator
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from huggingface_hub import snapshot_download

model_dir = snapshot_download("REPO_ID_PLACEHOLDER")

calc = RemediCalculator(
    model_dir=model_dir,
    device="cuda",                       # or "cpu"
    mace_model_path="/local/MACE-POLAR-1-M.model",   # required: see below
)

dataset = MoleculeDataset.open_existing_dataset_from_dir("/path/to/dataset_zarr")
descriptors = calc.calculate(dataset)    # (N_molecules, 640)
```

`mace_model_path` is **not optional in practice.** The config carries the absolute
path the model was trained with on our cluster, which will not exist on your machine.

To build a `MoleculeDataset` from your own SMILES, see
[Prepare a dataset](https://molsuit.github.io/Rem3Di/prepare-a-dataset/).

## What you get

| | |
|---|---|
| Descriptor dimension | **640** |
| Backbone | MACE-POLAR-1-M (frozen, 4096-d per-atom features) |
| Encoder | 4 pair-biased self-attention layers, width 1088 |
| Pooling | PMA, 4 learned seed queries |
| Parameters (encoder + pooler) | 77.5 M |
| Chirality | yes — pseudoscalar channels from the l≥1 features |

The descriptor is permutation-invariant, O(3)-invariant up to the chiral channels, and
of fixed length regardless of molecule size.

## Training

Self-supervised denoising. Atomic embeddings are corrupted, the molecule is encoded to
a single descriptor, and a decoder reconstructs the clean embeddings — so geometry must
survive the molecule-level bottleneck.

| | |
|---|---|
| Corpus | GEOM-Drugs, top-1 Boltzmann conformer per molecule |
| Size | 281,071 conformers / 278,679 unique molecules |
| Elements | H, C, N, O, F, P, S, Cl, Br, I |
| Molecule size | 3–128 atoms (mean 44.5) |
| Filters | neutral, single fragment, no radicals or isotopes |
| Split | molecule-level random 0.95/0.05 → 267,007 / 14,064 |
| Schedule | 24 epochs, AdamW, LR 1e-4, weight decay 1e-3, noise σ 0.3 |

## Evaluation

Benchmarked on TDC ADMET and MoleculeNet with a 5-seed MLP head on the frozen
descriptor, official splits. Against other frozen backbones under an identical recipe
(30-task relative score, min–max over the four backbones):

| Backbone | Regression (12) | Classification (18) | All (30) |
|---|---|---|---|
| MACE-MP-0 | 0.92 ± 0.10 | 0.70 ± 0.37 | 0.79 ± 0.31 |
| **MACE-POLAR-1-M** | 0.84 ± 0.14 | 0.70 ± 0.33 | **0.76 ± 0.28** |
| MACE-OFF24-medium | 0.70 ± 0.30 | 0.67 ± 0.28 | 0.68 ± 0.29 |
| Orb-v3 | 0.08 ± 0.28 | 0.08 ± 0.25 | 0.08 ± 0.26 |

MP-0 and POLAR are statistically indistinguishable here. POLAR is the released backbone
because it also supplies the l≥1 channels the chiral encoder needs.

**Adapting beats using it frozen.** Relative score across a 13-task ladder:

| | frozen | + LoRA | + full fine-tune |
|---|---|---|---|
| Pretrained | 0.655 | 0.814 | 0.958 |
| Random init | 0.018 | 0.447 | 0.598 |

LoRA (r=16, lr 5e-5) recovers most of full fine-tuning at a fraction of the cost. It is
sensitive to learning rate — at 1e-4 it collapses to chance.

## Limitations

- **Requires an ASL-licensed backbone at runtime.** Academic use only in practice.
- **Element coverage is limited** to the ten elements above. Molecules outside that set
  cannot be featurised.
- **Conformer-dependent.** The descriptor is a function of the 3D structure you give it.
  We pretrained on a single Boltzmann-weighted conformer per molecule; results on
  ensembles or on poor conformers may differ.
- **Orb-backbone caveat.** Orb-v3 is Apache-2.0 and would remove the licence problem,
  but its per-atom latents are not rotation-invariant — re-embedding a rotated molecule
  shifts them by ~67% of per-channel scale. With 16-rotation test-time averaging an Orb
  variant reaches 0.766 classification AUROC against a 0.768–0.774 MACE band, but
  regression stays behind and the chiral channels are unavailable entirely.
- **No formal significance testing.** "Tied" above means within one seed-level standard
  deviation across 5 seeds; no confidence intervals are reported.

## Citation

```bibtex
@article{wedig2026rem3di,
  title  = {Rem3Di: Learning smooth, chiral 3D molecular descriptors from
            atomistic foundation models},
  author = {Wedig, Steffen and Burton, Felix and Elijo{\v{s}}ius, Rokas and
            Schran, Christoph and Schaaf, Lars L.},
  journal = {arXiv preprint arXiv:2607.19977},
  year   = {2026}
}
```

Please also cite MACE-POLAR-1 ([arXiv:2602.19411](https://arxiv.org/abs/2602.19411)) if
you use this checkpoint, since it cannot run without it.
