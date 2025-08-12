#!\bin\bash

conda init
conda activate remedi

python scripts/run_antiviral_cross_val.py --n_conf 1
python scripts/run_antiviral_cross_val.py --n_conf 4
python scripts/run_antiviral_cross_val.py --n_conf 32
python scripts/run_antiviral_cross_val.py --n_conf 16
python scripts/run_antiviral_cross_val.py --n_conf 8
python scripts/run_antiviral_cross_val.py --n_conf 64

