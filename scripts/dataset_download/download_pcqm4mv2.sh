#!/bin/bash
# PCQM4Mv2: 3D structures of the 3,378,606 training molecules plus SMILES and HOMO-LUMO gaps.
# The OGB server (ogb-data.stanford.edu) refused connections on 2026-11-02. The SDF comes from a
# Hugging Face copy and is checked against OGB's published md5; the zip is the AWS mirror the
# ogb package itself downloads from.
# Usage: download_pcqm4mv2.sh <target directory>
set -euo pipefail
mkdir -p "$1/pcqm4mv2"
cd "$1/pcqm4mv2"
wget -c https://huggingface.co/datasets/liuganghuggingface/moldata-pcqm4mv2/resolve/main/structures/pcqm4m-v2-train.sdf.tar.gz
echo "fd72bce606e7ddf36c2a832badeec6ab  pcqm4m-v2-train.sdf.tar.gz" | md5sum -c -
wget -c https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip
sha256sum pcqm4m-v2-train.sdf.tar.gz pcqm4m-v2.zip > SHA256SUMS
tar -xzf pcqm4m-v2-train.sdf.tar.gz
unzip -q -o pcqm4m-v2.zip
