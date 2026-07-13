from pydantic import BaseModel

from remedi.evaluation.regression.featurization import (
    DescriptorCalculatorConfig,
)


class PredictionEvalTaskConfig(BaseModel):
    learner_config: ...
    dataset_path: ...
    featurizer_config: DescriptorCalculatorConfig
    cv_params: ...


class PredictionEvalTaks:
    def __init__(self):
        pass
