# %% Modules

import torch

from torch.nn import Module, Parameter
from torch.nn.init import normal_

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# %% Class definition

class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int | None = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.weights = Parameter(torch.empty(vocab_size, d_model))

        normal_(self.weights, mean=0.0, std=0.02)

        if self.pad_id is not None:
            with torch.no_grad():
                self.weights[self.pad_id] = 0.0

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.weights[token_ids]

        if self.pad_id is not None:
            with torch.no_grad():
                self.weights[self.pad_id].zero_()

        return output

# %% Testing

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    vocab_size = 1000
    d_model = 64
    pad_id = 0

    token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    embedding = Embedding(vocab_size, d_model, pad_id)

    print(embedding(token_ids).shape)

# %% End of script
