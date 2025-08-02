import logging
import time
from abc import ABC, abstractmethod

from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.data_handling.dataset import BaseDataset
from threedscriptors.data_handling.dataset_builder import DatasetBuilder
from threedscriptors.data_handling.dataset_io import (
    load_data_from_disk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define the core Stage interface
class BuildStage(ABC):
    def run(self, builder: DatasetBuilder) -> DatasetBuilder:
        """
        Take a DatasetBuilder (or intermediate builder) and perform one step,
        returning the (possibly modified) builder.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("➡️ Starting")
        t0 = time.time()
        try:
            result = self._run(builder)
            elapsed = time.time() - t0
            logger.info(f"✅ Finished in {elapsed:.2f}s")
            return result
        except Exception:
            logger.exception("💥 Failed")
            raise

    @abstractmethod
    def _run(self, builder: DatasetBuilder): ...


class InitializeBuildPipeline(BuildStage):
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.dataset_type = dataset_config.dataset_type.value

    def _run(self, _):
        dataset = self.dataset_type(dataset_config=self.dataset_config)
        builder = DatasetBuilder(dataset=dataset)
        return builder


class ReloadFromDiskStage(BuildStage):
    def __init__(self, directory):
        self.directory = directory

    def _run(self, _: DatasetBuilder):
        dataset = load_data_from_disk(self.directory)

        builder = DatasetBuilder(dataset=dataset)
        return builder


class InsertSmilesStage(BuildStage):
    def __init__(self, smiles):
        self.smiles = smiles

    def _run(self, builder: DatasetBuilder):
        builder.add_smiles_data(self.smiles)
        return builder


class InsertMoleculeStage(BuildStage):
    def __init__(self, molecules, structure_ids):
        self.molecules = molecules
        self.structure_ids = structure_ids

    def _run(self, builder):
        builder.add_molecules(self.molecules, self.structure_ids)
        return builder


class ConformalEmbeddingStage(BuildStage):
    def _run(self, builder):
        builder.embed_structures_from_smiles()
        return builder


class RelaxStage(BuildStage):
    def __init__(self, mace_calculator: MACECalculator):
        self.mace_calculator = mace_calculator

    def _run(self, builder):
        print("Started Relaxing")
        builder.relax_structures(self.mace_calculator)
        print("Finished Relaxing")
        return builder


# class TorchSimRelaxStage(BuildStage):
#
#    def __init__(self, mace_calculator: MACECalculator):
#
#        self.mace_calculator = mace_calculator
#        self.torch_sim_mace_calculator = MaceModel(mace_calculator.models[0])
#
#    def _run(self, builder):
#        builder.relax_structures_torchsim(self.torch_sim_mace_calculator)


class ChiralConformalEmbeddingStage(BuildStage):
    def _run(self, builder):
        builder.load_pairwise_chiral_structures_from_smiles()
        return builder


class NormalizationStage(BuildStage):
    def __init__(self, mean_targets, std_targets):
        self.mean_targets = mean_targets
        self.std_targets = std_targets

    def _run(self, builder):

        builder.normalize_regression_targets(self.mean_targets, self.std_targets)
        return builder


class AtomicEmbeddingStage(BuildStage):
    def __init__(self, mace_calculator: MACECalculator):
        self.mace_calculator = mace_calculator

    def _run(self, builder: DatasetBuilder):

        builder.calculate_atomic_embeddings(
            calculator=self.mace_calculator
        )
        return builder




class MolecularDescriptorStage(BuildStage):
    def __init__(self, descriptor_calculator):
        self.descriptor_calculator = descriptor_calculator

    def _run(self, builder: DatasetBuilder):
        builder.add_molecular_descriptor(self.descriptor_calculator)
        return builder


class RegressionLabelingStage(BuildStage):
    def __init__(self, regression_targets, regression_masks):
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks

    def _run(self, builder: DatasetBuilder):

        builder.add_regression_data(
            regression_targets=self.regression_targets,
            regression_masks=self.regression_masks,
        )

        return builder


class AtomicPositionsStage(BuildStage):

    def _run(self, builder : DatasetBuilder):
        builder.add_atomic_positions()
        return builder


class AddRandomWalkTransitionProbabilityMatrixStage(BuildStage):

    def _run(self,builder: DatasetBuilder):

        builder.add_random_walk_matrices()

        return builder

class AuxillaryDataStage(BuildStage):
    def __init__(self, auxillary_data):
        self.aux_data = auxillary_data

    def _run(self, builder: DatasetBuilder):
        builder.add_auxillary_data(self.aux_data)
        return builder


class SimilarityLabelingStage(BuildStage):
    def __init__(self, target_class_labels, active_decoy_labels):
        self.target_class_labels = target_class_labels
        self.active_decoy_labels = active_decoy_labels

    def _run(self, builder):
        builder.add_similarity_screening_data(
            self.target_class_labels, self.active_decoy_labels
        )

        return builder


class CanonicalizeStructureIDStage(BuildStage):

    def _run(self, builder: DatasetBuilder):
        builder.canonicalize_structure_ids()
        return builder


class SanitizeLogLabels(BuildStage):

    def _run(self,builder: DatasetBuilder):
        builder.sanitize_log_scaled_regression_targets()
        return builder




class PipelineOrchestrator:
    def __init__(self, stages: list[BuildStage]):
        self.stages = stages
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> BaseDataset:
        self.logger.info("🔨 Building pipeline")
        self.builder = self.stages[0].run(None)
        for stage in self.stages[1:]:
            self.builder = stage.run(self.builder)

        dataset = self.builder.dataset
        self.logger.info("🎉 Pipeline complete")
        return dataset
