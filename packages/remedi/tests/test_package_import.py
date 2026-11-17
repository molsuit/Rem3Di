"""The model modules import without TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD."""

from __future__ import annotations

import os
import subprocess
import sys


def test_e3nn_modules_import_with_weights_only_loading() -> None:
    """e3nn reads its Wigner constants with ``torch.load`` at import; ``remedi``
    allowlists the ``slice`` they hold, so a clean environment (no
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD) can import the model."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import remedi.model.remedi_model, remedi.utils.pseudoscalar_paths",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
