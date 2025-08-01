import numpy as np
import pandas as pd
import pingouin as pg


single_conf_runs = [0.3589017540216446, 0.5200750231742859, 0.46499399840831757, 0.4767233729362488, 0.8825672566890717, 0.48990803956985474, 0.4465969055891037, 0.6080317497253418, 0.4679517298936844, 0.710316002368927, 0.3674023151397705, 0.44915957748889923, 0.669556051492691, 0.585677832365036, 0.810077890753746, 0.8736345618963242, 0.4967687577009201, 0.41552169620990753, 0.5237913131713867, 0.4911726713180542, 0.704640805721283, 0.7704883515834808, 0.6055524796247482, 0.45815540850162506, 0.3445603996515274]


ten_conf_run = [0.47684661032898085, 0.399101481373821, 0.6798874520297561, 0.27765623001115664, 0.5597120406372207, 0.42825370920555933, 0.34375108911522795, 0.4215369629008429, 0.34727243706583977, 0.581163156245436, 0.2642854933759996, 0.7668485183800969, 0.485358684190682, 0.3236162755638361, 0.4811415145439761, 0.6775454985243934, 0.3788979340876852, 0.28883263282477856, 0.29587043821811676, 0.5237886375481529, 0.30123806478721754, 0.356673905027232, 0.7501553520560265, 0.3920990526676178, 0.5566212050616741]


print("Single configuration runs mean:", np.mean(single_conf_runs))
print("Single configuration runs std:", np.std(single_conf_runs))

print("Ten configuration runs mean:", np.mean(ten_conf_run))
print("Ten configuration runs std:", np.std(ten_conf_run))



n_repeats = 25
model_names = ['SingleConf', '10Conf']

# Replace these with your actual lowest‐error values:

# Build a long‐form DataFrame
df = pd.DataFrame({
    'Model': np.repeat(model_names, n_repeats),
    'Repeat': list(range(1, n_repeats+1)) * len(model_names),
    'Error': single_conf_runs + ten_conf_run
})
print(df)

aov = pg.rm_anova(data=df, dv='Error', within='Model', subject='Repeat', detailed=True)
print("\nRepeated‐Measures ANOVA:")
print(aov)

# ---------------------------------------------------------------------
# 3) Tukey HSD Post‐Hoc
# ---------------------------------------------------------------------
# For two models this is simple, but Pingouin provides pairwise_tukey()
tukey = pg.pairwise_tukey(data=df, dv='Error', between='Model')
print("\nTukey HSD Post‐Hoc:")
print(tukey)




runs = [TrainingMetadata(d) for d in dirs]


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



fig = plt.figure(figsize= (5.5, 2.75))

max_epoch = 0
for run in runs:

    epochs, val_loss = run.get_validation_loss()
    epochs = np.array(epochs) + 1
    max_epoch = max(max_epoch, max(epochs))
    N_confs = run.dataset_config.N_conformers


    min_val_loss[N_confs] = min(val_loss)

    plt.plot(epochs,val_loss,c = scalarMap.to_rgba(N_confs), label = f"{N_confs}")



plt.xlabel("Training Epoch")
plt.ylabel("Validation Loss (a.u)")
plt.yscale("log")
plt.legend()

ax = plt.gca()                           # or use the axes you created
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    title=r"Number of \\ conformers",
    title_fontsize="medium",             # legend-title styling (optional)
    frameon=False,                       # turn off legend box if preferred
    loc="center left",                   # move as needed
    bbox_to_anchor=(1.02, 0.5), borderaxespad=0
)


plt.xlim([1, max_epoch])
plt.xticks(ticks= [1,10,20,30,40,50], labels=["1","10","20","30","40","50"])

# Leave room on the right for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("conformal_dataugmentation.svg", bbox_inches="tight")




plt.figure(figsize=(3,2.75))

n_confs = np.array(list(min_val_loss.keys()))
min_val_losses = np.array(list(min_val_loss.values()))



# Linear regression:  log (y) ≈ (-nu) * log(x) + const
slope, _ = - np.polyfit(np.log(n_confs), np.log(min_val_losses), deg=1)
print(slope)
plt.plot(n_confs, min_val_losses)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Min. Validation Loss")
plt.xlabel(r"$N_{\text{sampled}}$ Conformers")
plt.tight_layout()
plt.savefig("log_log_dataaugmentation.svg")
