from abc import ABC, abstractmethod
from pathlib import Path

from remedi.evaluation.results import EvalResult
from remedi.model.regression_models import MultiTaskRegressionModel


class BaseEvalTask(ABC):
    @abstractmethod
    def __init__(self):
        self.results: list[EvalResult] = []

    @abstractmethod
    def run(self, model: MultiTaskRegressionModel):
        pass

    @abstractmethod
    def plot(self):
        pass


class EvalPipelineRunner:
    def __init__(self, tasks: list[BaseEvalTask], dataset_name):
        self.tasks = tasks
        self.dataset_label = f"{dataset_name}"
        self.results: list[EvalResult] = []

    def evaluate(self, model: MultiTaskRegressionModel, model_name):
        model.eval()

        self.results.clear()

        for task in self.tasks:
            task.results.clear()
            task.run(model)
            task.plot(model_name)
            self.results.extend(task.results)

    def output_results(self, output_directory: str):
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in self.results:
            result.serialize_to(output_dir)
