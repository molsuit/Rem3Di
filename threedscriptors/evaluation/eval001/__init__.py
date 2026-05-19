"""EVAL-001: the apples-to-apples descriptor benchmark panel.

"EVAL-001" is a stable evaluation-registry identifier (see logs/decisions.md
DEC-043), not a version number. This package scores any molecular descriptor
(ECFP, REM3DI, or a bring-your-own .npz) on the **TDC ADMET** (22 tasks, PyTDC
official admet_group splits + per-task official metrics) and **MoleculeNet**
(8 datasets, deterministic DeepChem scaffold split) panels, with honest
coverage accounting. Entry point: scripts/evaluation/run_eval_001.py; usage in
scripts/evaluation/README_eval001.md.
"""

