# Pseudoscalars

**What you'll do:** enumerate the pseudoscalar paths available from a set of e3nn
irreps, and convert equivariant features into pseudoscalar (chirality-sensitive)
invariants.

## Why pseudoscalars

An ordinary invariant (`0e`) is unchanged by *any* rotation or reflection — so it
gives a molecule and its mirror image the **same** value and cannot see chirality.
A **pseudoscalar** (`0o`) is rotation-invariant but **flips sign under reflection**,
so it distinguishes enantiomers.

Pseudoscalars can't be read off directly: from ordinary (polar) tensors, `0o` is
reachable only via **two** tensor products — you need an axial (`l ≥ 1`)
intermediate that contracts with a third input to `0o` (equivalently, an odd
`l₁+l₂+l₃` path). The classic example is the scalar triple product of three
vectors, `(a × b) · c`.

## The universal function

`find_pseudoscalar_paths(irreps)` enumerates **every** valid 2-tensor-product path
to `0o` for a given irrep set — the same selection rules (triangle + parity) e3nn
and MACE use. It's pure and cheap; use it to check what's reachable.

```python
from remedi.utils.pseudoscalar_paths import find_pseudoscalar_paths

# e.g. MACE-style node irreps
paths = find_pseudoscalar_paths("8x0e + 8x1o + 8x2e")
for p in paths:
    print(p)   # ir_in1 ⊗ ir_in2 → ir_mid ; ir_mid ⊗ ir_in3 → 0o
```

Each result is a `PseudoscalarPath` (`ir_in1, ir_in2, ir_mid, ir_in3, ir_out=0o`).
An empty list means no pseudoscalar is reachable from those irreps.

## Producing pseudoscalars from features

`Rem3DiPseudoScalarTP` is an `nn.Module` that realizes *all* discovered paths as two
fused tensor products and maps equivariant features `[..., irreps.dim]` to `K`
pseudoscalars `[..., K]`:

```python
import torch
from e3nn.o3 import Irreps
from remedi.model.preprocessing.pseudoscalar_tp import Rem3DiPseudoScalarTP

irreps = Irreps("8x0e + 8x1o + 8x2e")
ps_tp = Rem3DiPseudoScalarTP(
    input_irreps=irreps,
    pseudoscalar_dimension=4,      # K
    invariant_conditioned=False,   # standalone: no invariant conditioning
)

x = torch.randn(16, irreps.dim)    # a batch of equivariant features
ps = ps_tp(x)                      # -> (16, 4) pseudoscalars
```

By construction the output is `K x 0o`, so reflecting the input negates it — that
sign flip *is* the chirality signal (the notebook checks it explicitly).

> `invariant_conditioned=True` (the default in the model) makes the tensor-product
> weights a function of invariant features — pass `invariant_irreps=...` at
> construction and `invariants=...` to `forward`. Use `False` for a standalone,
> self-contained block.

## Outputs

- `find_pseudoscalar_paths` → a list of `PseudoscalarPath` (possibly empty).
- `Rem3DiPseudoScalarTP(...)(x)` → a `[..., K]` tensor of pseudoscalars.

## Next steps

- Notebook: `examples/05_pseudoscalars.ipynb` — enumerate paths, build the block,
  and verify the reflection sign-flip (runs on CPU, no model needed).
- [Concepts → descriptor shape](concepts.md#descriptor-shape).
