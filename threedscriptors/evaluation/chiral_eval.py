import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from threedscriptors.data_handling.dataset import RegressionWithAuxDataset
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.model.regression_models import MultiTaskRegressionModel

rc_params = {
    "text.usetex": True,  # Enable LaTeX rendering
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Computer Modern Roman"],  # Specify the default LaTeX font
    "axes.unicode_minus": False,  # Avoid Unicode minus problems
    "text.latex.preamble": r"\usepackage{amsmath}",  # Include LaTeX packages as needed
}

mpl.rcParams.update(rc_params)


def plot_molecule_pseudoscalar_comparison(
    dataset: RegressionWithAuxDataset,
    ps_model: MultiTaskRegressionModel,
    no_ps_model: MultiTaskRegressionModel,
):
    dataloader = DataLoader(
        dataset,
        batch_size=dataset.dataset_config.N_conformers,
        shuffle=False,
        collate_fn=sample_collate_fn,
    )

    device = "cuda"

    enantiomer_pred_ps = np.zeros(
        (len(dataloader), dataset.dataset_config.N_conformers)
    )

    enantiomer_pred_nops = np.zeros(
        (len(dataloader), dataset.dataset_config.N_conformers)
    )

    for batch_idx, samples in enumerate(dataloader):
        embeddings = samples.embeddings.to(device)
        padding_mask = samples.padding_mask.to(device)
        auxillary_data = samples.auxillary_data

        enantiomer_pred_ps[batch_idx, :] = (
            ps_model(
                embeddings, padding_mask=padding_mask, auxillary_data=auxillary_data
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        enantiomer_pred_nops[batch_idx, :] = (
            no_ps_model(
                embeddings, padding_mask=padding_mask, auxillary_data=auxillary_data
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

    fig = plt.figure()

    regression_targets = (
        dataset.regression_targets.reshape(-1, dataset.dataset_config.N_conformers)
        .detach()
        .cpu()
        .numpy()
    )

    y_min = min(
        [
            np.min(a=regression_targets),
            np.min(enantiomer_pred_ps),
            np.min(enantiomer_pred_nops),
        ]
    )

    y_max = max(
        [
            np.max(regression_targets),
            np.max(enantiomer_pred_ps),
            np.max(enantiomer_pred_nops),
        ]
    )

    n_confs_per_enantionmer = int(dataset.dataset_config.N_conformers / 2)
    x = np.ones(shape=(n_confs_per_enantionmer))

    plt.ylim([y_min, y_max])

    plt.xlim([0, 8 * 10])

    for idx, ps_pred, no_ps_pred, gt_label in zip(
        range(10),
        enantiomer_pred_ps,
        enantiomer_pred_nops,
        regression_targets,
        strict=False,
    ):
        class_pos = idx * 8

        e0_ps = ps_pred[:n_confs_per_enantionmer]
        e1_ps = ps_pred[n_confs_per_enantionmer:]

        e0_nps = no_ps_pred[:n_confs_per_enantionmer]
        e1_nps = no_ps_pred[n_confs_per_enantionmer:]

        center = np.mean(gt_label)

        plt.vlines(x=class_pos, ymin=-5, ymax=5, colors="k")
        plt.scatter(
            (class_pos + 2) * x, e0_ps - center, c="tab:blue", marker="*", label="E0PS"
        )
        plt.scatter(
            (class_pos + 3) * x,
            e1_ps - center,
            c="tab:orange",
            marker="*",
            label="E1PS",
        )

        plt.scatter(class_pos + 4, gt_label[0] - center, c="tab:blue", label="GT_E0")
        plt.scatter(
            class_pos + 4,
            gt_label[n_confs_per_enantionmer] - center,
            c="tab:orange",
            label="GT_E1",
        )
        plt.scatter(
            (class_pos + 5) * x,
            e0_nps - center,
            c="tab:blue",
            marker="x",
            label="E0NPS",
        )
        plt.scatter(
            (class_pos + 6) * x,
            e1_nps - center,
            c="tab:orange",
            marker="x",
            label="E1NPS",
        )

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="orange", linestyle="None", label="Enantiomer 0"
        ),
        Line2D(
            [0], [0], marker="o", color="blue", linestyle="None", label="Enantiomer 1"
        ),
        Line2D(
            [0], [0], marker="o", color="black", linestyle="None", label="Reference"
        ),
        Line2D([0], [0], marker="*", color="black", linestyle="None", label="With PS"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="No PS"),
    ]

    plt.xticks(ticks=np.linspace(4, 76, 10), labels=[str(i) for i in range(10)])
    plt.yticks([])
    plt.ylabel("Retention Time [a.u]")
    plt.xlabel("Molecule")
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
    )

    return fig
