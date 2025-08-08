from pathlib import Path

import pydantic_yaml as pyaml
import yaml

from threedscriptors.configuration.data_config import DatasetConfig


class TrainingMetadata:


    def __init__(self, loss_data, dataset_config):
        self.loss_data = loss_data
        self.config = dataset_config


    @classmethod
    def from_dir(cls, training_data_dir: Path):
        with open(f"{training_data_dir}/train/training_losses.yaml") as f:
            training_loss_data = yaml.safe_load(f)

        dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, f"{training_data_dir}/dataset_config.yaml")

        return cls(loss_data = training_loss_data, dataset_config = dataset_config)


    def get_validation_loss(self):

        validation_loss = []

        epochs = []

        for epochal_data in self.loss_data:
            epochs.append(epochal_data["epoch"])
            validation_loss.append(epochal_data["validation_loss"])

        return epochs, validation_loss


