#!/bin/bash
# tmQM, pinned to the 2025-03-17 commit (108k TMCs, 2024 release, with SMILES in tmQM_y.csv).
# Usage: download_tmqm.sh <target directory>
set -euo pipefail
commit=ece824419c9006fe66436daf233822b75533b82f
mkdir -p "$1/tmqm"
cd "$1/tmqm"
for f in tmQM_X1.xyz.gz tmQM_X2.xyz.gz tmQM_X3.xyz.gz tmQM_y.csv; do
    wget -c "https://raw.githubusercontent.com/uiocompcat/tmQM/$commit/tmQM/$f" -O "$f"
done
echo "$commit" > COMMIT
sha256sum tmQM_* > SHA256SUMS
