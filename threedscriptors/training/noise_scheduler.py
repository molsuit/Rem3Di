from abc import ABC, abstractmethod

import torch
from torch import nn


class NoiseScheduler(ABC):
    @abstractmethod
    def step(self): ...
    @property
    @abstractmethod
    def value(self): ...


class ConstantSchedule(NoiseScheduler):
    def __init__(self, level):
        self._lvl = level

    @property
    def value(self):
        return self._lvl

    def step(self):
        pass



class NoiseModule(nn.Module):
    def __init__(self, scheduler: NoiseScheduler):
        super().__init__()
        self.scheduler = scheduler
        # register a buffer to track level on correct device
        self.register_buffer("noise_level", torch.tensor(self.scheduler.value))

    def forward(self, S, padding_mask=None):
        # sync buffer
        self.noise_level.fill_(self.scheduler.value)
        noise = torch.randn_like(S) * self.noise_level
        if padding_mask is not None:
            noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0)

        S_noised = S.clone().detach() + noise
        return S_noised

    def step(self):
        self.scheduler.step()
