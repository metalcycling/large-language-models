# %% Modules

import torch

from tokenizer import CharTokenizer
from torch.utils.data import Dataset

# %% Classes

class NextTokenDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.tokens[idx     : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y

# %% Testing

if __name__ == "__main__":
    corpus = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.'\"-"
    tokenizer = CharTokenizer(corpus)

    samples = [
        "We're careful about orange ping pong balls because people might think they're fruit.",
        "There are no heroes in a punk rock band.",
        "I covered my friend in baby oil.",
    ]

    dataset = []
    block_size = 6

    for sample in samples:
        dataset.append(
            NextTokenDataset(
                tokenizer.encode(sample),
                block_size,
            )
        )

        x, y = dataset[-1][30]
        print(f"{tokenizer.decode(x.tolist())} -> {tokenizer.decode(y.tolist())}")

# %% End of script

