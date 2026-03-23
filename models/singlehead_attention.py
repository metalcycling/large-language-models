# %% Modules

import math
import torch
import torch.nn as nn

from torch.nn import Module
from torch.nn.init import normal_

import torch.nn.functional as F

# %% Classes

class CausalSelfAttentionSingleHead(Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head)
        self.W_k = nn.Linear(d_model, d_head)
        self.W_v = nn.Linear(d_model, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        _, seq_len, _ = x.shape
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        A = F.softmax(scores, dim=-1)

        output = A @ V

        return output

# %% Testing

if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 4
    d_head = 2

    x = torch.empty(batch_size, seq_len, d_model)
    normal_(x, mean=0.0, std=0.02)

    attention = CausalSelfAttentionSingleHead(d_model, d_head)

    print(attention(x).shape)

# %% End of script
