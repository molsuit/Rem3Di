from abc import ABC, abstractmethod

from molfeat.trans.fp import FPVecTransformer

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)


from threedscriptors.model.remedi_model import REM3DIModel


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
