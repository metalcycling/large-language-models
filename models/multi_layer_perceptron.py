# %% Modules

import math
import torch
import torch.nn as nn

from torch.nn import Module, Linear
from torch.nn.init import normal_

import torch.nn.functional as F

# %% Classes

class MultiLayerPerceptron(Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()

        self.d_model = d_model
        self.expansion = expansion

        self.expand = Linear(d_model, expansion * d_model)
        self.compress = Linear(expansion * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(F.gelu(self.expand(x)))

# %% Testing

if __name__ == "__main__":
    d_model = 4

    multi_layer_perceptron = MultiLayerPerceptron(d_model)

    x = torch.empty(d_model)
    normal_(x, mean=0.0, std=0.02)

    y = multi_layer_perceptron(x)

    print(y)

# %% End of script
