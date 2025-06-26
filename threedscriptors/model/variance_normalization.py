import torch
from torch.nn import Parameter, Module



class VarianceNormalization(Module):
    def __init__(self, num_features, eps=1e-3):

        super(VarianceNormalization, self).__init__()
        self.num_features = num_features
        self.running_var = Parameter(torch.ones(num_features), requires_grad=False)
        self.num_batches_tracked = 0
        self.eps = eps

    def forward(self, inp):

        exponential_average_factor = 0.0

        mask = torch.ones_like(inp[..., 0])
        mask = mask.unsqueeze(-1)
        shape = inp.shape
        shape_mask = mask.shape

        n = mask.sum()

        mask = mask / (n + 1e-8)

        inp = inp.reshape((-1, shape[-1]))

        mask = mask.reshape((-1, 1))

        if self.training:

            self.num_batches_tracked += 1
            exvf = 1.0 / (1.0 + float(self.num_batches_tracked))
            exponential_average_factor = exvf

        var = (mask * inp**2).sum(dim=0)  # - mean ** 2

        var = torch.clamp(var, self.eps, 1e8)
        if self.training:
            with torch.no_grad():
                self.running_var.data = (
                    exponential_average_factor * var * (n / (n - 1.0))
                    + (1.0 - exponential_average_factor) * self.running_var.data
                )

        inp = (inp) / (self.running_var[None, :] ** (1.0 / 2.0))
        inp = inp.reshape(shape)
        mask = mask.reshape(shape_mask)

        return inp