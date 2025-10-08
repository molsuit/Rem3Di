import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import yaml as vanilla_yaml

from threedscriptors.model.regression_models import MultiTaskRegressionModel


class BaseEvalTask(ABC):
    @abstractmethod
    def __init__(self):
        self.results = {}
        self.figs = {}

    @abstractmethod
    def run(self, model: MultiTaskRegressionModel):
        pass

    @abstractmethod
    def plot(self, model_name: str):
        pass



class EvalPipelineRunner:
    def __init__(
        self, tasks: list[BaseEvalTask], dataset_name
    ):

        self.tasks = tasks
        self.dataset_label = f"{dataset_name}"

    def evaluate(self, model: MultiTaskRegressionModel):
        model.eval()

        for task in self.tasks:
            task.run(model)

    def output_results(self, output_directory: str, model_name: str):

        os.makedirs(output_directory, exist_ok=True)
        figs = self.visualize(output_directory, model_name)

        report_str = self.write_results(output_directory, model_name)

        return figs, report_str

    def visualize(self, output_directory: str, model_name: str):
        figs: dict[str : plt.Figure] = {}  # taskname : Figure

        os.makedirs(output_directory, exist_ok=True)
        for task in self.tasks:
            task.plot()
            figs.update(task.figs)

        for fig_name, fig in figs.items():
            fig.savefig(
                f"{output_directory}/{fig_name}_{model_name}_{self.dataset_label}.pdf"
            )

        return figs

    def write_results(self, output_directory, model_name):


        os.makedirs(output_directory,exist_ok=True)
        result_dict = {}

        for task in self.tasks:
            if any(task.results):
                result_dict.update(task.results)

        with open(
            f"{output_directory}/training_results_{model_name}_{self.dataset_label}.yaml",
            "w",
        ) as f:
            vanilla_yaml.dump(result_dict, f)

        return result_dict
