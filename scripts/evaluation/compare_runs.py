from threedscriptors.evaluation.training_metadata import TrainingMetadata
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

dirs = [
    "/home/snw30/rds/hpc-work/3DMolecularDescriptors/training_runs/10-2025_08_16_21_16_42-qm9_scratch_finetune10k",
    "/home/snw30/rds/hpc-work/3DMolecularDescriptors/training_runs/17-2025_08_16_21_05_59-qm9_pre100k_finetune10k",
    "/home/snw30/rds/hpc-work/3DMolecularDescriptors/training_runs/13-2025_08_16_20_39_16-qm9_pre200k_finetune10k",
    "/home/snw30/rds/hpc-work/3DMolecularDescriptors/training_runs/15-2025_08_16_20_55_23-qm9_pre400k_finetune10k",
]
labels = ["From Scratch", "100k" , "200k", "400k"]
pretrain_size = [0, 100_000, 200_000, 400_000]

runs = [TrainingMetadata.from_dir(d) for d in dirs]

# LaTeX styling (remove if LaTeX not available)
rc_params = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{helvet}\usepackage{sansmath}\sansmath",
    "font.size" : 11
}
mpl.rcParams.update(rc_params)

# Color map by pretraining size for lines (scratch in gray)
cmap = plt.cm.Blues
norm = mpl.colors.LogNorm(vmin=100_000, vmax=400_000)

min_val_loss = {}

fig = plt.figure(figsize=(5.5, 2.75))
ax = plt.gca()

max_epoch = 0
for i, (size, run) in enumerate(zip(pretrain_size, runs)):
    epochs, val_loss = run.get_validation_loss()
    epochs = np.asarray(epochs) + 1
    val_loss = np.asarray(val_loss, dtype=float)
    # drop NaNs if present
    mask = np.isfinite(epochs) & np.isfinite(val_loss)
    epochs, val_loss = epochs[mask], val_loss[mask]
    if len(epochs) == 0:
        continue

    max_epoch = max(max_epoch, int(epochs.max()))
    min_val_loss[size] = float(np.nanmin(val_loss))

    if size == 0:
        color = "0.5"
    else:
        color = cmap(norm(size))
    ax.plot(epochs, val_loss, label=labels[i], lw=1.6, color=color)

# --- Figure 1 formatting
ax.set_xlabel("Training Epoch")
ax.set_ylabel("Validation Loss (a.u.)")
ax.set_yscale("log")

# dynamic x ticks up to max_epoch
tick_candidates = [1, 10, 20, 30, 40, 50, 75, 100]
xt = [t for t in tick_candidates if t <= max_epoch]
if 1 not in xt:
    xt = [1] + xt
ax.set_xlim(1, max_epoch)
ax.set_xticks(xt)
ax.set_xticklabels([str(t) for t in xt])

handles, leg_labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    leg_labels,
    title=r"Pretraining",
    title_fontsize="medium",
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
)
ax.grid(True, which="both", alpha=0.2)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("pretraining.png", bbox_inches="tight")
plt.close()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


sizes = np.array(sorted(min_val_loss.keys())) 
losses = np.array([min_val_loss[s] for s in sizes], dtype=float)
scratch = float(min_val_loss[0])
ratios = losses / scratch

print(ratios)

mask = sizes > 0

fig, ax = plt.subplots(figsize=(3, 2.75))

ax.plot(sizes[mask], ratios[mask], marker="o", lw=1.6, label=r"$\min\mathrm{MSE}/\mathrm{scratch}$")
ax.axhline(1.0, ls="--", lw=1.4, color="0.4", label="Scratch baseline")

y_ticks = [0.90, 0.94, 0.98, 1.00]
ax.yaxis.set_major_locator(FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.yaxis.set_minor_formatter(NullFormatter())   # hide minor labels
ax.set_ylim(0.89, 1.01)

# X: show 1, 2, 4 with ×10^5 in the axis label
x_ticks = [1e5, 2e5, 4e5]
ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.set_xticklabels([r"$1\times 10^{5}$", r"$2\times 10^{5}$", r"$4\times 10^{5}$"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Pretraining Dataset Size $N $")
ax.set_ylabel(r"Relative min. validation MSE")
ax.legend(frameon=False)
plt.tight_layout()


plt.savefig("loglog_ratio_to_scratch.svg", bbox_inches="tight")

# optional: slope on N>0 (same as absolute since division by constant)
m, b = np.polyfit(np.log(sizes[mask].astype(float)),
                  np.log(ratios[mask].astype(float)), 1)
nu = -m
print({"nu": float(nu)})