# `scripts/release/` — publishing model weights to Hugging Face

Tooling for putting a trained Rem3Di model on the Hub, plus a gate that blocks an
upload that users could not load.

## Why the gate exists

The docs and the README both presuppose a **published model directory**
([Quickstart](https://molsuit.github.io/Rem3Di/quickstart/),
[Concepts](https://molsuit.github.io/Rem3Di/concepts/#model-directory)), and
`RemediCalculator(model_dir=...)` is the entry point. No such directory is published
anywhere today, which is what [#2](https://github.com/molsuit/Rem3Di/issues/2) is asking
for.

The risk when we do publish is subtle. A model directory has the right *four filenames*
long before its *weights* are loadable by the current package, because the layout is
stable while module internals are not. A checkpoint from before a refactor will look
correct on disk and fail at `load_state_dict`. `verify_model_dir.py` moves that failure
from the user to us.

## Contents

| File | Purpose |
|---|---|
| `verify_model_dir.py` | Load a candidate directory with the installed `remedi` and print exact key and shape mismatches. Exit 0 = publishable. |
| `upload_to_hf.py` | Run the gate, refuse on failure, then create the repo and upload. Private by default. |
| `MODEL_CARD.md` | Model-card template for a MACE-POLAR checkpoint. |

## Usage

```bash
pip install "remedi[cpu]" huggingface_hub

python scripts/release/verify_model_dir.py /path/to/model_dir \
    --mace /local/MACE-POLAR-1-M.model

export HF_TOKEN=hf_...
python scripts/release/upload_to_hf.py /path/to/model_dir --repo <owner>/rem3di-polar
```

`HF_TOKEN` comes from the environment only — never a CLI argument, never a file.
Repositories are created **private**; `--public` is a separate deliberate flag.

## What a publishable model directory is

`EncoderOnlyArchitectureConfig.from_encoder_yaml` reads the config, drops
`decoder_config`, and forces `kind: encoder_only`, so a pretraining checkpoint works
directly. Four files:

```
model_dir/
├── post_training_architecture_config.yaml
├── encoder.pth
├── atomic_preprocessor.pth
└── geometric_preprocessor.pth
```

**`mace_model_path` is effectively required for any downloaded model.** The config
carries the absolute MACE path from the training machine, which will not exist on a
user's system. The model-card template says so; the docs should too.

## Licensing, stated plainly

A Rem3Di checkpoint contains **no MACE tensors** — the backbone is a runtime dependency
resolved through `mace_config.model_path`, and users download MACE from ACEsuit and
accept its licence there. Publishing Rem3Di weights therefore does not redistribute
ASL-licensed material, and the model card says so.

What remains open is whether the ASL reaches *derived* models: Rem3Di is trained on
MACE-POLAR features, and `atomic_preprocessor.pth` stores precomputed statistics of that
feature distribution. Worth a direct answer from the MACE authors before any repository
is made public. An Orb-v3 backbone (Apache-2.0) would sidestep it, at the cost of the
chiral channels, which need `l≥1` features Orb does not provide.

## A compatibility note for whoever publishes first

Checkpoints trained before the PMA rewrite will not load. In `PMAAggregator`, `head_dim`
is now the **per-head** Q/K dimension, so the total is `num_heads * head_dim`; older
checkpoints were trained when it was the total. The visible symptom on one such
checkpoint:

| Tensor | Old checkpoint | Current package |
|---|---|---|
| `seeds` | `(1, 4, 320)` | `(1, 4, 640)` |
| `W_Q.weight` | `(320, 320)` | `(2560, 640)` |
| `W_K.weight` | `(320, 1088)` | `(2560, 1088)` |
| `W_O` | absent | required |

`ChiralEmbeddingModel` changed too — `lin0/lin1/lin2/tp_cross/tp_dot` became
`Rem3DiPseudoScalarTP`, so 14 of 21 preprocessor keys have no counterpart. Conversion
recovers the transformer body but not the seeds, Q/K reshape, or `W_O`; a retrain against
the released package is the honest route.

One robustness suggestion while this is fresh: `PMAAggregatorConfig` does not set
`extra="forbid"`, so an older config's `d_v_out` and `reduction` fields are **silently
dropped** rather than rejected. Failing loudly there would make stale checkpoints
diagnose themselves.

---

## Clean-room install, 2026-08-02 — what actually happens today

Run in a fresh Python 3.12 venv on Linux, exactly as an external user would:
`pip install "remedi[cpu]"`. Three independent failures, in the order you hit them.

**1. `pip install` produces a package that cannot import.** `remedi` declares
`torch>=2.5.1` with no upper bound, so pip resolves torch 2.13.0, while `mace-torch`
hard-pins `e3nn==0.4.4`. e3nn 0.4.4's `o3/_wigner.py` calls `torch.load(constants.pt)`
without `weights_only=False`, and torch ≥2.6 defaults that to `True`:

```
_pickle.UnpicklingError: Weights only load failed.
  Unsupported global: GLOBAL slice was not an allowed global by default
```

Any import that reaches `e3nn.o3` dies, which includes
`remedi.configuration.architecture_config`. Upgrading to `e3nn>=0.5` fixes it but
conflicts with mace-torch's pin, so the actionable fix is a `torch<2.6` bound (or a
mace-torch that accepts newer e3nn).

**2. The wheel is missing `remedi.data_handling.dataset` entirely.** A one-character
typo — `__init_.py` — means hatchling does not treat the directory as a package. The
published wheel has 142 `.py` files against 147 in the repo. `RemediCalculator`,
`MoleculeDataset` and `TrainingMoleculeDataset` are all unreachable, so **every code
path in the README and the docs site fails at import from PyPI.** Fixed on branch
`fix-dataset-package-init`; verified by dropping the five files into the installed
wheel, after which the documented quickstart imports cleanly.

**3. Pre-refactor checkpoints do not even parse.** With 1 and 2 worked around, the gate
run against a MACE-POLAR checkpoint from the pre-PMA-rewrite lineage fails earlier than
the shape mismatch documented above:

```
ok    all 4 required files present
FAIL  config does not parse: ValidationError for EncoderOnlyArchitectureConfig
      Value error, too many values to unpack (expected 2)
```

So the config schema diverged as well as the weights. Publishing such a checkpoint would
give users a validation error before they ever reached `load_state_dict`.

**Net:** 1 and 2 affect every user of the released package today and are worth fixing
regardless of any weights release. 3 confirms that the route to publishing weights is a
retrain against the current package, not a conversion.
