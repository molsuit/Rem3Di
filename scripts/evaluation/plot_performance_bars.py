import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

run_dict = {
    "No": "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/207-2025_08_09_17_21_01-qm9_no_pos",
    "2D": "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/200-2025_08_09_16_47_39-QM9 With RW",
    "3D": "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/205-2025_08_09_16_24_02-RelativePosQm9",
}

rc_params = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"""
\usepackage{helvet}     % Helvetica
\usepackage{sansmath}   % use sans serif in math mode as well
\sansmath
""",
"font.size": 14
}
mpl.rcParams.update(rc_params)

def read_val_loss(dir):

    training_results_file = glob.glob(f"{dir}/valset_results/training_results*.yaml")[0]
    with open(training_results_file) as f:
        results = yaml.safe_load(f)

    mae = results["Regression Eval Stats"]["gap"]["mean absolute error"]
    mae = mae * 27211.385
    return mae


plt.figure()
plt.bar(x=list(run_dict.keys()), height=[read_val_loss(d) for d in run_dict.values()])
plt.ylabel("MAE HOMO-LUMO gap (meV)")
plt.savefig("bar_chart_qm9_pos.svg")

maes = [read_val_loss(d) for d in run_dict.values()]

maes_ratio = [(maes[0]-m)/(maes[0]) for m in maes]
print(maes_ratio)
