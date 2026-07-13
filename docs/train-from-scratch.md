# Train from scratch

**What you'll do:** pretrain a new Rem3Di descriptor model with self-supervised
denoising, producing a model directory you can then use for
[evaluation](evaluate-a-model.md).

The objective: noise the atomic embeddings, encode to a molecular descriptor,
and have a decoder reconstruct the clean embeddings — so the descriptor must
capture the structure. An optional VICReg term keeps descriptor dimensions
non-degenerate. (`remedi/training/pretraining.py`.)

## Prerequisites

- [Installed](installation.md) with a CUDA-12.6 GPU.
- A [MoleculeDataset](prepare-a-dataset.md) zarr to train on.
- A MACE foundation-model file for the frontend.

## 1. Create a training directory with two YAMLs

```
my_run/
├── training_config.yaml       # optimization + data loading
└── architecture_config.yaml   # model architecture (kind: encoder_decoder)
```

Copy a working pair from
`configs/training/pcqm_ablation_novicreg/pcqm_baseline/` and edit the paths.

`training_config.yaml` — the keys you'll usually touch:

```yaml
training_name: my_run
run_group: my_experiments
dataset_path: /path/to/dataset_zarr
model_config_path: /abs/path/to/my_run/architecture_config.yaml
output_base: /path/to/training_runs  # timestamped run dir created here
epochs: 15
learning_rate: 0.0003
weight_decay: 0.001
max_grad_norm: 1.0
noise_level: 0.3                          # denoising noise std
split_config:
  strategy: single
  train_val_ratios: [0.98, 0.02]
  shuffle: true
dataloader:
  batch_sampling:
    kind: bucket_batch                    # pack batches by atom count
    max_atoms_per_batch: 5250
    bucket_size: 512
  num_workers: 12
  persistent_workers: true
  pin_memory: true
  prefetch_factor: 4
vicreg:
  enabled: false  # set true + weights to regularize descriptors
  variance_weight: 25.0
  covariance_weight: 1.0
wandb_active: false
```

`architecture_config.yaml` (`kind: encoder_decoder`) defines the MACE frontend,
the encoder/decoder transformers, the pairwise positional encoding, and the
pooling aggregator. Edit the example directly, or generate one in Python — see
`scripts/write_config_file.py` for a template using `EncoderDecoderArchitectureConfig`
and friends (`remedi/configuration/architecture_config.py`). Set
`mace_config.model_path` to your MACE file.

## 2. Smoke-test first

Cap the run to a handful of molecules and one epoch to check everything wires up:

```bash
python scripts/run_online_embedding_denoising_pretraining.py \
    --training_config my_run/training_config.yaml \
    --max_train 100 --max_val 20 --max_epochs 1
```

Other smoke flags: `--max_train_batches`, `--max_val_batches`, and
`--dataset_path` to override the dataset without editing the yaml.

## 3. Run the full training

```bash
python scripts/run_online_embedding_denoising_pretraining.py \
    --training_config my_run/training_config.yaml
```

## Outputs

A timestamped run directory under `output_base/` containing exactly the files an
[evaluation model directory](concepts.md#model-directory) needs — the best
checkpoint is saved whenever validation improves:

- `encoder.pth`, `atomic_preprocessor.pth`, `geometric_preprocessor.pth`
- `post_training_architecture_config.yaml`

## Next steps

- [Evaluate a model](evaluate-a-model.md) — point `model_dir` at the run directory.
- Notebook: `examples/04_pretrain_mini.ipynb`.
