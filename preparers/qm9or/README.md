# QM9-OR — author-created train/validation/test splits

These are the exact train/validation/test splits used for the chirality experiments in the
Rem3Di paper (R/S configuration and optical-rotation-sign prediction). They are **author-created**
and are provided here so the reported numbers can be reproduced.

## Source dataset

QM9-OR pairs the QM9 (GDB-9) molecules with TD-DFT optical rotations and Cahn–Ingold–Prelog
configuration labels.

- Underlying dataset: **QM9-OR**, Zhou et al. (2025), "QM9-OR: DFT Optimized Geometries and
  Optical Rotations for Selected QM9 Molecules", Zenodo record
  [13380412](https://zenodo.org/records/13380412) (`qm9-or.npy`, md5
  `d7a24d28e5f611acefb9a07ff9831efd`, sha256
  `cbe1c0def10a57e3d29c33e666b9fec30c04cf8cf3ddd8f59f642e4a1772e9ba`). Downloaded to the gitignored
  `benchmark_data/raw/qm9or/`; the bundle preparer is not written yet.
- Molecules: **117,625** unique 2D molecules (`molecule_id`). The experiments additionally use each
  molecule together with its mirror image (labels flipped) for a mirror-balanced task; only one row
  per `molecule_id` is stored here, and both enantiomers of a molecule always share its split.

## File: `qm9or_splits.csv`

One row per molecule; SMILES stored once, one column per split assignment.

| column | meaning |
|---|---|
| `molecule_id` | stable integer id (groups all stereoisomers/enantiomers of a 2D molecule) |
| `isomeric_smiles` | canonical isomeric SMILES |
| `n_chiral` | number of chiral centres |
| `rs` | R/S configuration label |
| `or_sign_589` | sign of the optical rotation at 589.3 nm (−1 / 0 / +1) |
| `random_s42 … random_s45` | seeded random split, value ∈ {`train`,`valid`,`test`} |
| `scaffold_s0 … scaffold_s2` | Bemis–Murcko scaffold split, value ∈ {`train`,`valid`,`test`} |

### Split families

- **Random** (`random_s42`–`random_s45`): seeded random partition keyed by `molecule_id`
  (`numpy.random.default_rng(seed).permutation`), `test = 20%`, `valid = 10%`, `train = 70%`.
  Counts per seed: **train 82,338 / valid 11,762 / test 23,525**.
- **Scaffold** (`scaffold_s0`–`scaffold_s2`): Bemis–Murcko scaffold split
  (RDKit `MurckoScaffoldSmiles`, `includeChirality=True`), whole scaffold groups assigned atomically
  (no scaffold leakage), `train/valid/test = 0.8/0.1/0.1`, seeds 0–2.
  Counts per seed: **train 94,100 / valid 11,762 / test 11,763**.

Both families are keyed by `molecule_id`, so a molecule and all of its stereoisomers/enantiomers
always land in the **same** partition — there is no chirality leakage across train/test.

## Provenance

- Source table `qm9or_molecules-use.csv`, sha256
  `38386849432b81390454748dbda9ccb5ac11c260e621ba848b86a469266be2d6`.
- Exact author generators are kept under `generators/` for reference:
  - `make_splits.py` — random splits (seeds 42–45).
  - `make_scaffold_splits.py` — scaffold splits (seeds 0–2).
- The random splits in `qm9or_splits.csv` were verified bit-exact against `make_splits.py`; the
  scaffold columns are the frozen canonical partitions the paper trained on.

## Load

```python
import pandas as pd
df = pd.read_csv("qm9or_splits.csv")

# e.g. scaffold split, seed 0
train = df[df.scaffold_s0 == "train"]
test  = df[df.scaffold_s0 == "test"]
X_train, y_train = train.isomeric_smiles, train.or_sign_589
```
