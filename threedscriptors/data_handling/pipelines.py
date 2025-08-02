from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.data_build_pipeline import (
    AddRandomWalkTransitionProbabilityMatrixStage,
    AtomicEmbeddingStage,
    AtomicPositionsStage,
    AuxillaryDataStage,
    CanonicalizeStructureIDStage,
    ChiralConformalEmbeddingStage,
    ConformalEmbeddingStage,
    InitializeBuildPipeline,
    InsertMoleculeStage,
    InsertSmilesStage,
    PipelineOrchestrator,
    RegressionLabelingStage,
    RelaxStage,
    ReloadFromDiskStage,
    SanitizeLogLabels,
    SimilarityLabelingStage,
)


def regression_training_pipeline(
    dataset_config: DatasetConfig, smiles, regression_targets, regression_masks
):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
    ]

    return PipelineOrchestrator(stages)



def regression_training_with_pos_pipeline(
    dataset_config: DatasetConfig, smiles, regression_targets, regression_masks
):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
        AtomicPositionsStage(),
        AddRandomWalkTransitionProbabilityMatrixStage(),
        CanonicalizeStructureIDStage(),

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
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ChiralConformalEmbeddingStage(),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),
        AuxillaryDataStage(auxillary_data),
        AtomicPositionsStage()
    ]

    return PipelineOrchestrator(stages)


def pretraining_pipeline(dataset_config: DatasetConfig, smiles):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
    ]

    return PipelineOrchestrator(stages)


def pretraining_pipeline_with_positions(dataset_config: DatasetConfig, smiles):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        AtomicPositionsStage()
    ]

    return PipelineOrchestrator(stages)

def pretraining_pipeline_from_structures(dataset_config: DatasetConfig, molecules, structure_ids):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertMoleculeStage(molecules=molecules, structure_ids=structure_ids),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        AtomicPositionsStage(),
        AddRandomWalkTransitionProbabilityMatrixStage(),
    ]

    return PipelineOrchestrator(stages)





def similarity_screening_pipeline(
    dataset_config: DatasetConfig, smiles, target_class_labels, active_decoy_labels
):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertSmilesStage(smiles=smiles),
        ConformalEmbeddingStage(),
        RelaxStage(dataset_config.embedding_model_config.mace_calc),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        SimilarityLabelingStage(target_class_labels, active_decoy_labels),
    ]

    return PipelineOrchestrator(stages)

def regression_training_from_structures_pipeline(
    dataset_config: DatasetConfig,
    molecules,
    structure_ids,
    regression_targets,
    regression_masks,
):
    stages = [
        InitializeBuildPipeline(dataset_config),
        InsertMoleculeStage(molecules=molecules, structure_ids=structure_ids),
        AtomicEmbeddingStage(dataset_config.embedding_model_config.mace_calc),
        RegressionLabelingStage(
            regression_targets=regression_targets, regression_masks=regression_masks
        ),AtomicPositionsStage(),
        #AddRandomWalkTransitionProbabilityMatrixStage(),
    ]

    return PipelineOrchestrator(stages)




def reload_dataset_pipeline(
    directory
) -> PipelineOrchestrator :


    stages = [
        ReloadFromDiskStage(directory),
        AtomicPositionsStage(),
    ]
    return PipelineOrchestrator(stages)

def reload_regression_dataset_with_log_sanitation_pipeline(
    directory
) -> PipelineOrchestrator :


    stages = [
        ReloadFromDiskStage(directory),
        AtomicPositionsStage(),
        SanitizeLogLabels()
    ]
    return PipelineOrchestrator(stages)




