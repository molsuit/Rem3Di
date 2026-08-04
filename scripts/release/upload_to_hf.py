#!/usr/bin/env python3
"""Publish a verified Rem3Di model directory to the Hugging Face Hub.

Refuses to upload anything that `verify_model_dir.py` rejects, because a model
that cannot be loaded by `pip install remedi` is worse than no model at all —
the failure lands on the user.

Credentials come from the environment only:

    export HF_TOKEN=hf_...
    python upload_to_hf.py /path/to/model_dir --repo Felixb7/rem3di-polar-exp050

Never pass a token on the command line and never commit one. If a token has
been pasted anywhere shared, rotate it at https://huggingface.co/settings/tokens

Repositories are created **private by default**. Going public is a separate,
deliberate `--public` flag, because publishing weights derived from an
ASL-licensed backbone is a decision for the authors, not for a script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--repo", required=True, help="e.g. Felixb7/rem3di-polar-exp050")
    ap.add_argument("--card", type=Path, default=HERE / "MODEL_CARD.md")
    ap.add_argument(
        "--public",
        action="store_true",
        help="create/flip the repo to public. Do not use until the licensing "
        "question in RELEASE_PLAN.md section 5d has an answer.",
    )
    ap.add_argument("--mace", type=Path, default=None, help="local MACE path for verification")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="bypass the load check. Only for uploading a card to an empty repo.",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set in the environment. Refusing to continue.")
        return 2

    if not args.skip_verify:
        cmd = [sys.executable, str(HERE / "verify_model_dir.py"), str(args.model_dir)]
        if args.mace:
            cmd += ["--mace", str(args.mace)]
        print("Running the pre-publish load check...\n")
        rc = subprocess.call(cmd)
        if rc != 0:
            print("\nVerification did not pass. Nothing was uploaded.")
            return 1
        print()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub")
        return 2

    api = HfApi(token=token)

    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    visibility = "PUBLIC" if args.public else "private"
    print(f"Repository {args.repo} ready ({visibility}).")

    if args.card and args.card.is_file():
        api.upload_file(
            path_or_fileobj=str(args.card),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"Uploaded model card from {args.card.name}")

    if args.model_dir.is_dir() and not args.skip_verify:
        api.upload_folder(
            folder_path=str(args.model_dir),
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=["*.log", "__pycache__/*", ".DS_Store"],
        )
        print(f"Uploaded {args.model_dir}")

    print(f"\nhttps://huggingface.co/{args.repo}")
    if not args.public:
        print("Still private. Flip it in the repo settings, or re-run with --public,")
        print("once the ASL derived-work question has an answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
