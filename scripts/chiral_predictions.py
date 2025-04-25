import pydantic_yaml as pyaml

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.data_handling.dataset import RegressionWithAuxDataset
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.evaluation.chiral_eval import plot_molecule_pseudoscalar_comparison
from threedscriptors.model.model_builder import ModelBuilder

if __name__ == "__main__":
    dataset = load_data_from_disk(
        directory="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt",
        dataset_cls=RegressionWithAuxDataset,
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
