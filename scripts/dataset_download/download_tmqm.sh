#!/bin/bash

cd $1

mkdir tmqm
cd tmqm

wget https://github.com/uiocompcat/tmQM/raw/refs/heads/master/tmQM/tmQM_X1.xyz.gz
gzip -d tmQM_X1.xyz.gz

wget https://github.com/uiocompcat/tmQM/raw/refs/heads/master/tmQM/tmQM_X2.xyz.gz
gzip -d tmQM_X2.xyz.gz


wget https://github.com/uiocompcat/tmQM/raw/refs/heads/master/tmQM/tmQM_X3.xyz.gz
gzip -d tmQM_X3.xyz.gz

wget https://github.com/uiocompcat/tmQM/raw/refs/heads/master/tmQM/tmQM_y.csv

# Strip blank lines between frames — ASE's xyz parser treats them as EOF
# and only yields the first frame per file.
for f in tmQM_X1.xyz tmQM_X2.xyz tmQM_X3.xyz; do
    sed -i '/^[[:space:]]*$/d' "$f"
done
