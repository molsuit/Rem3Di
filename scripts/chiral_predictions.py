import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic_yaml as pyaml
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.data_handling.dataset import (
    RegressionAtomEmbeddingDatasetWithAuxillaryData,
)
from threedscriptors.data_handling.dataset_builder import DatasetBuildingDirector
from threedscriptors.model.model_builder import ModelBuilder
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
    cmrt_dataset: RegressionAtomEmbeddingDatasetWithAuxillaryData,
    ps_model: MultiTaskRegressionModel,
    no_ps_model: MultiTaskRegressionModel,
):
    dataloader = DataLoader(
        cmrt_dataset, batch_size=cmrt_dataset.dataset_config.N_conformers, shuffle=False
    )

    device = "cuda"

    enantiomer_pred_ps = np.zeros(
        (len(dataloader), cmrt_dataset.dataset_config.N_conformers)
    )

    enantiomer_pred_nops = np.zeros(
        (len(dataloader), cmrt_dataset.dataset_config.N_conformers)
    )

    for batch_idx, (
        embeddings,
        padding_mask,
        _,
        _,
        auxillary_data,
    ) in enumerate(dataloader):
        embeddings = embeddings.to(device)
        padding_mask = padding_mask.to(device)

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


if __name__ == "__main__":
    _, dataset = DatasetBuildingDirector.reload_dataset(
        directory="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt",
        return_normalized_targets=True,
        return_normalized_inputs=True,
    )

    ps_architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig,
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_ps/architecture_config.yaml",
    )
    ps_architecture_config.reload_full_model_weights = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_ps/regression_model.pth"

    ps_model_builder = ModelBuilder(architecture_config=ps_architecture_config)
    ps_model = ps_model_builder.build_model()
    ps_model.to("cuda")
    ps_model.eval()

    nops_architecture_config = pyaml.parse_yaml_file_as(
        ArchitectureConfig,
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_nops/architecture_config.yaml",
    )
    nops_architecture_config.reload_full_model_weights = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_nops/regression_model.pth"

    nops_model_builder = ModelBuilder(architecture_config=nops_architecture_config)
    nops_model = nops_model_builder.build_model()
    nops_model.to("cuda")
    nops_model.eval()

    fig = plot_molecule_pseudoscalar_comparison(dataset, ps_model, nops_model)

    fig.savefig("ps_vs_nops_pred.pdf", bbox_inches="tight")
