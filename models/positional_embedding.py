# %% Modules

import torch

from torch.nn import Module, Parameter
from torch.nn.init import normal_

# %% Class definition

class PositionalEmbedding(Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.weights = Parameter(torch.empty(max_seq_len, d_model))

        normal_(self.weights, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.weights[:seq_len]

# %% Testing

if __name__ == "__main__":
    seq_len = 5
    max_seq_len = 128
    d_model = 64

    positional_embedding = PositionalEmbedding(max_seq_len, d_model)

    print(positional_embedding(seq_len).shape)

# %% End of script

