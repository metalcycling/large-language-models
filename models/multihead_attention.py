# %% Modules

import math
import torch
import torch.nn as nn

from torch.nn import Module
from torch.nn.init import normal_

import torch.nn.functional as F

# %% Classes

class CausalSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view((batch_size, seq_len, self.num_heads, self.d_head)).transpose(1, 2)
        K = self.W_k(x).view((batch_size, seq_len, self.num_heads, self.d_head)).transpose(1, 2)
        V = self.W_v(x).view((batch_size, seq_len, self.num_heads, self.d_head)).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        A = F.softmax(scores, dim=-1)

        output = A @ V
        output = output.transpose(1, 2).contiguous().view((batch_size, seq_len, self.d_model))

        return self.W_o(output)

# %% Testing

if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 4
    num_heads = 2

    x = torch.empty(batch_size, seq_len, d_model)
    normal_(x, mean=0.0, std=0.02)

    attention = CausalSelfAttention(d_model, num_heads)
    self = attention

    print(attention(x).shape)

# %% End of script
