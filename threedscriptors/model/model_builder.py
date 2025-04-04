
from threedscriptors.model.architecture_config import RegressionHeadConfig
import torch.nn as nn

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
    for _ in task_list:
            head = nn.Sequential()

            for idx, dim in enumerate(regression_head_config.hidden_dimensions[:-1]):
                head.add_module(
                    f"linear_{idx}",
                    nn.Linear(dim, regression_head_config.hidden_dimensions[idx + 1]),
                )
                head.add_module("activation", self.activation)

            head.add_module(
                f"linear_{idx + 1}",
                nn.Linear(regression_head_config.hidden_dimensions[-1], 1),
            )

            self.task_heads.append(head)