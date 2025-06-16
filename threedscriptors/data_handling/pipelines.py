from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_build_pipeline import (
    AtomicEmbeddingStage,
    AuxillaryDataStage,
    ChiralConformalEmbeddingStage,
    ConformalEmbeddingStage,
    InitializeBuildPipeline,
    InsertMoleculeStage,
    InsertSmilesStage,
    NormalizationStage,
    PipelineOrchestrator,
    RegressionLabelingStage,
    RelaxStage,
    ReloadFromDiskStage,
    SimilarityLabelingStage,
    AtomicPositionsStage
)
from threedscriptors.data_handling.dataset import (
    AtomicEmbeddingDataset,
    RegressionDataset,
    SimilarityScreeningDataset,
    RegressionWithAuxDataset
)


def regression_training_pipeline(
    dataset_config: DatasetConfig, smiles, regression_targets, regression_masks
):
    stages = [
        InitializeBuildPipeline(dataset_config, RegressionDataset),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
        AtomicPositionsStage()
    ]

    return PipelineOrchestrator(stages)


def chiral_regression_training_pipeline(
    dataset_config: DatasetConfig,
    smiles,
    regression_targets,
    regression_masks,
    auxillary_data,
):
    stages = [
        InitializeBuildPipeline(dataset_config, RegressionWithAuxDataset),
        InsertSmilesStage(smiles=smiles),
        ChiralConformalEmbeddingStage(),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
        AuxillaryDataStage(auxillary_data),
    ]

    return PipelineOrchestrator(stages)


def pretraining_pipeline(dataset_config: DatasetConfig, smiles):
    stages = [
        InitializeBuildPipeline(dataset_config, AtomicEmbeddingDataset),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
    ]

    return PipelineOrchestrator(stages)


def similarity_screening_pipeline(
    dataset_config: DatasetConfig, smiles, target_class_labels, active_decoy_labels
):
    stages = [
        InitializeBuildPipeline(dataset_config, SimilarityScreeningDataset),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        SimilarityLabelingStage(target_class_labels, active_decoy_labels),
    ]

    return PipelineOrchestrator(stages)


def reload_dataset_pipeline(
    directory, mean_targets = None, std_targets = None
) -> PipelineOrchestrator :


    stages = [
        ReloadFromDiskStage(directory),
        AtomicPositionsStage(), 
        NormalizationStage(
            mean_targets, std_targets,
        ),
    ]
    return PipelineOrchestrator(stages)


def regression_training_from_structures_pipeline(
    dataset_config: DatasetConfig,
    molecules,
    mol_ids,
    regression_targets,
    regression_masks,
):
    stages = [
        InitializeBuildPipeline(dataset_config, RegressionDataset),
        InsertMoleculeStage(molecules=molecules, mol_ids=mol_ids),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
    ]

    return PipelineOrchestrator(stages)
