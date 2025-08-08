from threedscriptors.evaluation.training_metadata import TrainingMetadata
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.cm as cmx

dirs = [
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/245-2025_08_04_19_21_39-QM9NoPretrain",
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/246-2025_08_04_22_53_54-QM9FromPCQM",
]

labels = ["From Scratch", "Pretrained"]


runs = [TrainingMetadata.from_dir(d) for d in dirs]



min_val_loss = {}

import matplotlib as mpl


rc_params = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"""
\usepackage{helvet}     % Helvetica
\usepackage{sansmath}   % use sans serif in math mode as well
\sansmath
""",
}
mpl.rcParams.update(rc_params)

mpl.rcParams.update(rc_params)

cmap = plt.cm.Blues
cNorm = LogNorm(vmin=0.1, vmax=128)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)


fig = plt.figure(figsize=(5.5, 2.75))

max_epoch = 0
for i, run in enumerate(runs):

    epochs, val_loss = run.get_validation_loss()
    epochs = np.array(epochs) + 1
    max_epoch = max(max_epoch, max(epochs))
    N_confs = run.config.N_conformers

    min_val_loss[N_confs] = min(val_loss)

    plt.plot(epochs, val_loss, label=labels[i])



plt.xlabel("Training Epoch")
plt.ylabel("Validation Loss (a.u)")
plt.yscale("log")
plt.legend()

ax = plt.gca()  # or use the axes you created
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    title=r"Pretraining",
    title_fontsize="medium",  # legend‐title styling (optional)
    frameon=False,  # turn off legend box if preferred
    loc="center left",  # move as needed
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
)


plt.xlim([1, max_epoch])
plt.xticks(ticks=[1, 10, 20, 30, 40, 50], labels=["1", "10", "20","30", "40", "50"])


# Leave room on the right for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("pretraining.svg", bbox_inches="tight")

breakpoint()


plt.figure(figsize=(3, 2.75))

n_confs = np.array(list(min_val_loss.keys()))
min_val_losses = np.array(list(min_val_loss.values()))


# Linear regression:  log (y) ≈ (−ν) * log(x) + const
slope, _ = -np.polyfit(np.log(n_confs), np.log(min_val_losses), deg=1)
print(slope)
plt.plot(n_confs, min_val_losses)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Min. Validation Loss")
plt.xlabel(r"$N_{\text{sampled}}$ Conformers")
plt.tight_layout()
plt.savefig("log_log_dataaugmentation.svg")
