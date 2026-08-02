#!/usr/bin/env python3
"""Gate a Rem3Di model directory before it is published.

Checks that a directory really is loadable by the *released* `remedi` package —
the thing an external user will `pip install`. Publishing weights that only load
against a private branch is worse than publishing nothing, because the failure
lands on the user rather than on us.

Usage
-----
    python verify_model_dir.py /path/to/model_dir [--mace /path/to/MACE.model]

Exit codes
----------
    0  loadable — safe to publish
    1  not loadable — do not publish
    2  could not run the check (remedi not installed, etc.)

The four files a published model directory must contain, per
`remedi.evaluation.benchmark.descriptors.RemediCalculator`:

    post_training_architecture_config.yaml
    encoder.pth
    atomic_preprocessor.pth
    geometric_preprocessor.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED = [
    "post_training_architecture_config.yaml",
    "encoder.pth",
    "atomic_preprocessor.pth",
    "geometric_preprocessor.pth",
]


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"ok    {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument(
        "--mace",
        type=Path,
        default=None,
        help="local MACE foundation-model path, overriding the absolute path "
        "baked into the checkpoint config at training time",
    )
    args = ap.parse_args()
    d: Path = args.model_dir

    print(f"Verifying {d}\n")

    missing = [f for f in REQUIRED if not (d / f).is_file()]
    if missing:
        for f in missing:
            _fail(f"missing required file: {f}")
        return 1
    _ok(f"all {len(REQUIRED)} required files present")

    try:
        import torch
        from remedi.configuration.architecture_config import (
            EncoderOnlyArchitectureConfig,
        )
    except ImportError as e:
        print(f"\nCannot run the check: {e}")
        print("Install the released package first:  pip install 'remedi[cpu]'")
        return 2

    # 1. Config parses, and the decoder half is stripped as from_encoder_yaml expects.
    try:
        cfg = EncoderOnlyArchitectureConfig.from_encoder_yaml(d)
        _ok("architecture config parses as encoder_only")
    except Exception as e:
        _fail(f"config does not parse: {type(e).__name__}: {e}")
        return 1

    if args.mace is not None:
        if cfg.mace_config is None:
            _fail("--mace given but the config has no mace_config to apply it to")
            return 1
        cfg.mace_config.model_path = args.mace
        _ok(f"MACE path overridden to {args.mace}")
    elif cfg.mace_config is not None:
        p = Path(str(cfg.mace_config.model_path))
        if not p.exists():
            print(
                f"warn  MACE path in the config does not exist here: {p}\n"
                f"      External users will hit this too. Pass --mace, and make sure\n"
                f"      the model card tells users to set mace_model_path."
            )

    # 2. The model actually builds from that config.
    try:
        model = cfg.build()
        _ok("model builds from config")
    except Exception as e:
        _fail(f"cfg.build() failed: {type(e).__name__}: {e}")
        return 1

    # 3. The three state dicts load strictly. This is where a checkpoint from a
    #    pre-refactor lineage dies, and it must die here rather than for a user.
    targets = [
        ("encoder.pth", model.encoder),
        ("atomic_preprocessor.pth", model.preprocessor.atomic_preprocessor),
        ("geometric_preprocessor.pth", model.preprocessor.geometric_preprocessor),
    ]
    failed = False
    for fname, module in targets:
        sd = torch.load(d / fname, map_location="cpu", weights_only=False)
        try:
            module.load_state_dict(sd)
            _ok(f"{fname} loads (strict)")
        except Exception as e:
            failed = True
            _fail(f"{fname} does not load")
            have = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
            want = {k: tuple(v.shape) for k, v in module.state_dict().items()}
            only_ckpt = sorted(set(have) - set(want))
            only_model = sorted(set(want) - set(have))
            shape_diff = sorted(k for k in set(have) & set(want) if have[k] != want[k])
            if only_ckpt:
                print(f"      in checkpoint, not in model ({len(only_ckpt)}):")
                for k in only_ckpt[:8]:
                    print(f"        {k}  {have[k]}")
            if only_model:
                print(f"      in model, not in checkpoint ({len(only_model)}):")
                for k in only_model[:8]:
                    print(f"        {k}  {want[k]}")
            if shape_diff:
                print(f"      shape mismatches ({len(shape_diff)}):")
                for k in shape_diff[:8]:
                    print(f"        {k}  checkpoint {have[k]}  vs model {want[k]}")
            print(f"      ({type(e).__name__})")

    print()
    if failed:
        print("VERDICT: not loadable by the released remedi package. Do not publish.")
        print("A checkpoint from a pre-refactor lineage needs retraining against the")
        print("released code, not conversion — see RELEASE_PLAN.md section 3.")
        return 1

    print("VERDICT: loadable. Safe to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
