import torch.nn as nn

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    RegressionHeadConfig,
)


class ModelBuilder:
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config

    def count_trainable_parameters(self):
        pass

    def build_model(self):
        pass

    def build_preprocess(self):
        pass

    def build_encoder(self):
        pass

    def build_global_aggregator(self):
        pass

    def build_regression_head(self):
        pass

    def build_head(
        head_config: RegressionHeadConfig,
        input_dim: int,
    ) -> nn.Module:
        """
        Build a regression head based on the provided configuration.

        Args:
            head_config (RegressionHeadConfig): Configuration for the regression head.
            input_dim (int): Input dimension for the regression head.

        Returns:
            nn.Module: The constructed regression head.
        """
        raise NotImplementedError("The build_head function is not implemented yet.")
        # for _ in task_list:
        #    head = nn.Sequential()


#
#    for idx, dim in enumerate(regression_head_config.hidden_dimensions[:-1]):
#        head.add_module(
#            f"linear_{idx}",
#            nn.Linear(dim, regression_head_config.hidden_dimensions[idx + 1]),
#        )
#        head.add_module("activation", self.activation)
#
#    head.add_module(
#        f"linear_{idx + 1}",
#        nn.Linear(regression_head_config.hidden_dimensions[-1], 1),
#    )
#
#    self.task_heads.append(head)
#
