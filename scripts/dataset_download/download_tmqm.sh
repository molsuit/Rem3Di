#!\bin\bash

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
