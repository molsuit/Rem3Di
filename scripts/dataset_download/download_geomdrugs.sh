#!/bin/bash
# GEOM (rdkit_folder: drugs and qm9 pickles). Dataverse names the file .tar.gz, but it is a plain
# tar of 50,137,579,520 bytes.
# Usage: download_geomdrugs.sh <target directory>
set -euo pipefail
mkdir -p "$1/geom_drugs"
cd "$1/geom_drugs"
wget -c https://dataverse.harvard.edu/api/access/datafile/4327252 -O rdkit_folder.tar
sha256sum rdkit_folder.tar > SHA256SUMS
tar -xf rdkit_folder.tar
