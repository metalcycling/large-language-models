# %% Modules

import math
import torch
import torch.nn as nn

from torch.nn import Module, Linear, LayerNorm
from torch.nn.init import normal_

import torch.nn.functional as F

from multihead_attention import CausalSelfAttention
from multi_layer_perceptron import MultiLayerPerceptron

# %% Classes

class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.layer_norm_1 = LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.layer_norm_2 = LayerNorm(d_model)
        self.multi_layer_perceptron = MultiLayerPerceptron(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.multi_layer_perceptron(self.layer_norm_2(x))

        return x

# %% Testing

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 4
    num_heads = 2

    transformer_block = TransformerBlock(d_model, num_heads)

    x_1 = torch.empty(batch_size, seq_len, d_model)
    normal_(x_1, mean=0.0, std=0.02)

    x_2 = x_1.clone()
    x_2[:, -1, :] = torch.randn(d_model)

    y_1 = transformer_block(x_1)
    y_2 = transformer_block(x_2)

    print("max diff first T-1 positions:", (y_1[:, :-1, :] - y_2[:, :-1, :]).abs().max().item())
    print("diff last position:", (y_1[:, -1, :] - y_2[:, -1, :]).abs().max().item())

# %% End of script
