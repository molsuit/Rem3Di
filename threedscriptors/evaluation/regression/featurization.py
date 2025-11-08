from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Literal

from molfeat.trans.fp import FPVecTransformer
from pydantic import BaseModel, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.model.remedi_model import REM3DIModel


class _DescriptorCalculatorConfigBase(BaseModel, ABC):
    """Abstract base for discriminated-union. Do not instantiate directly."""
    calculator_id: Literal["molfeat", "remedi"]

    @abstractmethod
    def get_descriptor_calculator(self):
        pass

class MolfeatDescriptorCalculatorConfig(_DescriptorCalculatorConfigBase):
    calculator_id: Literal["molfeat"] = "molfeat"
    descriptor_molfeat_name: str

    def get_descriptor_calculator(self):
        return MolfeatDescriptorCalculator(self.descriptor_molfeat_name)

class RemediDescriptorCalculatorConfig(_DescriptorCalculatorConfigBase):
    calculator_id: Literal["remedi"] = "remedi"
    model_dir: Path

    def get_descriptor_calculator(self):
        remedi_model = ModelBuilder.from_directory(self.model_dir).build_remedi_model()

        return RemediDescriptorCalculator(remedi_model)

# Discriminated union type alias (pydantic v2 style)
DescriptorCalculatorConfig = Annotated[
    MolfeatDescriptorCalculatorConfig | RemediDescriptorCalculatorConfig,
    Field(discriminator="calculator_id"),
]



class DescriptorCalculator(ABC):
    @abstractmethod
    def calculate_descriptors(self, dataset):
        pass

class MolfeatDescriptorCalculator(DescriptorCalculator):
    def __init__(self, descriptor_name):
        super().__init__()
        self.descriptor_name = descriptor_name
        self.featurizer = FPVecTransformer(kind=self.descriptor_name)

    def calculate_descriptors(self, dataset: MoleculeDataset):
        return self.featurizer(dataset.get_smiles_per_structure())


class RemediDescriptorCalculator(DescriptorCalculator):
    def __init__(
        self,
        remedi_model : REM3DIModel,

    ):
        super().__init__()

        self.model = remedi_model.eval()
        self.descriptor_name = "remedi"

    def calculate_descriptors(self, dataset: MoleculeDataset):

        train_ds =  TrainingMoleculeDataset.from_molecule_dataset(dataset, get_item=pos_emb_getitem)
        descriptors = evaluate_molecular_descriptor_on_dataset(self.model, train_ds)

        descriptors = descriptors.numpy()

        return descriptors
