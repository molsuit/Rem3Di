import os
import re

import numpy as np
import pandas as pd
import pingouin as pg

top_dir = "/path/to/3DMolecularDescriptors/training_runs/0-av_potency_conformal_sampling"


# capture the number at the end of the line after a colon
LOSS_RE = re.compile(r":\s*([+-]?\d+(?:\.\d+)?)(?=\s*$)")


def extract_loss_value(line: str):
    m = LOSS_RE.search(line)
    return float(m.group(1)) if m else None


runs = {}
for item in os.listdir(top_dir):
    print(item)
    item_path = os.path.join(top_dir, item)

    if os.path.isdir(item_path):
        val_loss_file = os.path.join(item_path, "lowest_val_losses.txt")
        if os.path.exists(val_loss_file):
            with open(val_loss_file) as f:
                losses = [
                    v for line in f if (v := extract_loss_value(line)) is not None
                ]
            if losses:
                runs[item] = losses


print(runs)

for k, v in runs.items():
    print(k)
    print(np.mean(np.array(v)))


n_repeats = 10
model_names = list(runs.keys())

# Replace these with your actual lowest-error values:

# Build a long-form DataFrame
df = pd.DataFrame(
    {
        "Model": np.repeat(model_names, n_repeats),
        "Repeat": list(range(1, n_repeats + 1)) * len(model_names),
        "Error": np.concat(list(runs.values())),
    }
)


print(df)
breakpoint()
aov = pg.rm_anova(data=df, dv="Error", within="Model", subject="Repeat", detailed=True)


print("\nRepeated-Measures ANOVA:")
print(aov)

# ---------------------------------------------------------------------
# 3) Tukey HSD Post-Hoc
# ---------------------------------------------------------------------
# For two models this is simple, but Pingouin provides pairwise_tukey()
tukey = pg.pairwise_tukey(data=df, dv="Error", between="Model")
print("\nTukey HSD Post-Hoc:")
print(tukey)


breakpoint()


min_val_loss = {}

rc_params = {
    "text.usetex": True,  # Enable LaTeX rendering
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Computer Modern Roman"],  # Specify the default LaTeX font
    "axes.unicode_minus": False,  # Avoid Unicode minus problems
    # Include LaTeX packages as needed
    "text.latex.preamble": r"\usepackage{amsmath}",
}

mpl.rcParams.update(rc_params)

cmap = plt.cm.Blues
cNorm = LogNorm(vmin=0.1, vmax=128)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)


fig = plt.figure(figsize=(5.5, 2.75))

max_epoch = 0
for run in runs:
    epochs, val_loss = run.get_validation_loss()
    epochs = np.array(epochs) + 1
    max_epoch = max(max_epoch, max(epochs))
    N_confs = run.dataset_config.N_conformers

    min_val_loss[N_confs] = min(val_loss)

    plt.plot(epochs, val_loss, c=scalarMap.to_rgba(N_confs), label=f"{N_confs}")


plt.xlabel("Training Epoch")
plt.ylabel("Validation Loss (a.u)")
plt.yscale("log")
plt.legend()

ax = plt.gca()  # or use the axes you created
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    title=r"Number of \\ conformers",
    title_fontsize="medium",  # legend-title styling (optional)
    frameon=False,  # turn off legend box if preferred
    loc="center left",  # move as needed
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
)


plt.xlim([1, max_epoch])
plt.xticks(ticks=[1, 10, 20, 30, 40, 50], labels=["1", "10", "20", "30", "40", "50"])

# Leave room on the right for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("conformal_dataugmentation.svg", bbox_inches="tight")


plt.figure(figsize=(3, 2.75))

n_confs = np.array(list(min_val_loss.keys()))
min_val_losses = np.array(list(min_val_loss.values()))


# Linear regression:  log (y) ≈ (-nu) * log(x) + const
slope, _ = -np.polyfit(np.log(n_confs), np.log(min_val_losses), deg=1)
print(slope)
plt.plot(n_confs, min_val_losses)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Min. Validation Loss")
plt.xlabel(r"$N_{\text{sampled}}$ Conformers")
plt.tight_layout()
plt.savefig("log_log_dataaugmentation.svg")
