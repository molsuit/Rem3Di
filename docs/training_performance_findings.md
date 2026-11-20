# Training performance and deployment: findings (2026-11-20)

Measured on an RTX 5080 (16 GB), torch 2.15.0+cu132, on the stand-in corpora of
`scripts/gpu_checks/make_test_datasets.py` (1,954 small molecules, 15 drugs of
60-120 atoms), drug_like_small architecture with MACE-POLAR-1-M (float32, cueq).
Not comparable in absolute terms with the H200 runs on the real corpora.

## Where a training step goes

`scripts/profile_pretraining_step.py` (torch.profiler over the real training step):

| setting (compile on) | step ms | atoms/s |
|---|---|---|
| float32 | 54.5 | 70.7k |
| bfloat16 autocast (MACE featurizer opts out) | 40.6 | 94.9k |
| TF32 matmuls | crashes (see below) | - |
| float32, compile off | 76.8 | 50.1k |

- Per step (compile on, float32): MACE featurizer 17 ms (31 %), the rest of the
  preprocessing 1.5 ms, encoder + decoder forward/backward and optimizer 37 ms
  (67 %). The pair transformer, not the MLIP, is the bottleneck now.
- Every float32 matmul runs as `simt_sgemm` (no tensor cores). bf16 autocast moves
  the encoder/decoder to tensor cores: -25 % step time. The MACE featurizer keeps
  float32 (it disables autocast inside).
- Host-device synchronisations (`--find_synchronisations`): the model's step had
  five per step (the statistics flag read in every forward, a host-to-device copy
  of the channel indices and two boolean-mask selections in the preprocessor, and
  a tensor built from Python numbers in the compiled featurizer). All are removed;
  bf16 + compile went from 40.6 to ~37.5 ms per step (103k atoms/s), GPU idle
  15 % -> 8 %. The training loop's two per-step NaN guards (the loss finiteness
  check and `clip_grad_norm_(error_if_nonfinite=True)`) are replaced by
  `FiniteStepGuard`: the finiteness of loss and gradient norm is collected on the
  device and checked every 50 steps and at the end of every epoch, before
  validation and any checkpoint. A NaN now surfaces up to 50 steps late, with the
  weights of those steps lost, but the run still stops before saving anything
  from them (test `test_a_non_finite_step_stops_the_run_before_any_state_is_saved`).
  Counted on the real script: no synchronisation per step remains; the rest are
  per 50 steps (logging) or per epoch (validation, saving). On a faster GPU (H200,
  49 % utilisation in job 915761) host stalls matter more.

## Open issues

1. **Every training config has `compile.enabled: false`.** Compile gives -26 %
   step time (76 -> 56 ms) and is now correct (`PairLinear`); the configs predate
   the fix. Not yet validated on an H200.
2. **bf16 autocast is not wired into the training script.** Only the profiler has
   the switch (`--autocast bfloat16`). Needs a config field, and a check that the
   loss curve matches float32 over a real run.
3. **TF32 is ruled out.** It is a global switch, so it also reaches the frozen MACE
   featurizer: TF32 shifts the MACE features by up to 4e-3 (1.7e-3 relative)
   against ~1e-6 in float32. The featurizer now refuses to run unless
   `torch.get_float32_matmul_precision()` is `"highest"`. (TF32 with
   torch.compile also crashed the training step with `CUDA error: an illegal
   memory access`; eager TF32 ran.)
4. **Four recompiles per compiled run** (all in epoch 0): an inductor fusion
   heuristic guard on the pair-tensor size (`>= 5242880` elements) splits small and
   large batches into two graphs per module, and validation (no grad) adds one
   graph per module. Harmless; the cache holds them.

## Is bf16 autocast safe?

- The MACE featurizer disables autocast inside itself: under bf16 autocast its
  output is float32 and bit-identical to the float32 run (test
  `test_featurizer_stays_in_float32_under_autocast`).
- Parameters, gradients and AdamW states stay float32 (autocast casts per
  operation); LayerNorm, softmax and reductions run in float32 under autocast;
  bf16 needs no gradient scaler.
- One batch, eval mode, same initialisation: gradient cosine similarity float32
  vs bf16 0.99999, relative gradient-norm difference 0.4 %. (The losses agree to
  five digits only because the zero-initialised decoder head outputs zero.)
- 300 training steps from the same initialisation and batches (compile on, mean
  loss per 50 steps): float32 0.916 / 0.838 / 0.808 / 0.787 / 0.757 / 0.735,
  bf16 0.919 / 0.837 / 0.806 / 0.784 / 0.768 / 0.739. Differences of -0.003 to
  +0.010, with no drift in one direction; the run-to-run spread of float32 alone
  was not measured.
- Not yet verified: a full run on the real corpora (final validation loss, probe
  and benchmark scores against float32).

## AOTInductor export of the descriptor model

Prototype (scratchpad, not yet a script): the encoder-only model (MACE featurizer,
preprocessing, encoder) wrapped to take `(atomic_numbers, positions, system_index,
padding_mask, edge_index)`, exported with `torch.export` (dynamic atoms, molecules,
atom slots and pairs), compiled with `torch._inductor.aoti_compile_and_package`.

| featurizer | eager | AOTInductor | loads without Python packages |
|---|---|---|---|
| cuequivariance | 7.6 ms | 4.9 ms | no |
| e3nn | 87 ms | 31 ms | yes (torch only) |

(37 molecules per batch; max difference to eager 8e-7.)

- cuequivariance kernels are `torch.library.custom_op` (Python) operators, which
  an AOTInductor package cannot run; only the e3nn featurizer gives a
  self-contained package. `PairLinear` takes the plain linear under export for the
  same reason.
- Compile-time autotuning has to be off
  (`inductor_configs={"triton.autotune_at_compile_time": False}`): it runs the
  kernels on generated inputs that fail the model's device-side consistency asserts.
- The neighbour list is an input: a deployment must build it (or the export must
  compute it inside, e.g. on the dense intra-molecule graph).
- Fixed for the export: the statistics check read a buffer (`stats_set`) in every
  forward (data-dependent; also a device sync per step), and `quantize_pad_length`
  / `max()` in `pad_molecules` specialised the molecule and atom-slot counts.
