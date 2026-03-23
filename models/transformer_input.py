# %% Modules

import torch

from torch.nn import Module, Parameter
from torch.nn.init import normal_

from embedding import Embedding
from positional_embedding import PositionalEmbedding

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# %% Class definition

class TransformerInput(Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, pad_id: int | None = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embedding = Embedding(self.vocab_size, self.d_model, self.pad_id)
        self.positional_embedding = PositionalEmbedding(self.max_seq_len, self.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape

        embedding = self.embedding(token_ids)
        positional_embedding = self.positional_embedding(seq_len)

        return embedding + positional_embedding


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    max_seq_len = 128
    vocab_size = 1000
    d_model = 64
    pad_id = 0

    token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    transformer_input = TransformerInput(vocab_size, d_model, max_seq_len, pad_id)
    transformer_output = transformer_input(token_ids)

    print(transformer_output.shape)

# %% End of script


