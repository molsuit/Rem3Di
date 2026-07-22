# Installation

This page covers installing `remedi` and its dependencies.

## Prerequisites

- Python **≥ 3.12**
- A CUDA **12.6** capable GPU for training and for computing descriptors at scale
  (CPU works for small runs).
- [`uv`](https://docs.astral.sh/uv/) is recommended, since it resolves the pinned
  torch/CUDA and git dependencies for you.

## Install with uv (recommended)

```bash
git clone https://github.com/molsuit/Rem3Di.git
cd Rem3Di
uv sync                 # core install
uv sync --extra eval    # + downstream learners (LightGBM, scikit-learn)
```

`uv sync` uses the pins in `pyproject.toml`: the CUDA-12.6 torch wheels
(`pytorch-cu126` index) and the `torch-sim` MACE backend from git
(`torch-sim-atomistic`).

## Install with pip

```bash
pip install -e .            # core
pip install -e ".[eval]"    # + downstream learners
```

With pip you must install the git-sourced dependencies yourself:

```bash
pip install \
    "torch-sim-atomistic @ git+https://github.com/TorchSim/torch-sim.git@main"
```

## Verify

```bash
python - <<'PY'
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
print("ok")
PY
```

## Gotchas

- **zarr v3 only.** Datasets use `zarr>=3.2`. `polaris-lib` pins `zarr<3` and
  cannot share this environment; ingest Polaris data out-of-venv via
  `scripts/dataset_download/dump_polaris.py`.
- **numba.** Pinned to `numba>=0.61` (older versions fail to build on Python 3.12).
- **MACE weights.** The model uses a MACE foundation model as its frontend. Point
  the checkpoint's `mace_config.model_path` at a MACE model file on your machine
  (see [Concepts](concepts.md)).
- **`torch.load` weights-only error on import** (e3nn `constants.pt`, PyTorch ≥ 2.6):
  export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Next steps

- [Quickstart](quickstart.md): get descriptors in 5 minutes.
