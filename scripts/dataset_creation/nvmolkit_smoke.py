"""Tiny GPU sanity test for the nvMolKit backend used by DatasetComparison.

Exercises the exact code path `_nn_tanimoto_nvmolkit` would take: build Morgan
fingerprints for a handful of SMILES, run crossTanimotoSimilarity, take the
per-row max, and pull the result back to CPU. Prints diagonal == 1.0 (self
similarity) as the smoke check.
"""

from __future__ import annotations

import sys

import torch
from rdkit import Chem

from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarity


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() == False (no GPU?)", file=sys.stderr)
        return 1

    smiles = [
        "CCO",
        "CCN",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "COc1ccc(CC(N)C(=O)O)cc1",
    ]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols)

    fpgen = MorganFingerprintGenerator(radius=2, fpSize=2048)
    fps = fpgen.GetFingerprints(mols).torch()
    print(
        f"fingerprints: shape={tuple(fps.shape)} dtype={fps.dtype} device={fps.device}"
    )

    sim = crossTanimotoSimilarity(fps, fps)
    torch.cuda.synchronize()
    sim_t = sim.torch() if hasattr(sim, "torch") else sim
    print(
        f"similarity:   shape={tuple(sim_t.shape)} dtype={sim_t.dtype} device={sim_t.device}"
    )
    diag = sim_t.diagonal().cpu().numpy()
    print("diagonal (self-similarity, expect ~1.0):", diag)
    nn = sim_t.max(dim=1).values.cpu().numpy()
    print("NN max per query (each is itself, so 1.0):", nn)

    ok = bool(diag.min() > 0.999 and diag.max() < 1.001)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
