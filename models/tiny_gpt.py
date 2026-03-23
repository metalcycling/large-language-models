# %% Modules

import math
import shutil
import utils
import torch
import kagglehub

from datetime import datetime
from pathlib import Path

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear, LayerNorm, ModuleList
from torch.nn.init import normal_, zeros_, ones_
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer
from next_token import NextTokenDataset
from embedding import Embedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock

# %% Classes

class TinyGPT(Module):
    def __init__(
        self, 
        vocab_size: int,
        block_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(block_size, d_model)
        self.layers = ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

        self.lm_head_bias = Parameter(torch.zeros(vocab_size))

        self.apply(self._init_weights)

    def _init_weights(self, module: Module) -> None:
        if isinstance(module, Linear):
            normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, LayerNorm):
            ones_(module.weight)
            zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None):
        batch_size, seq_len = tokens.shape

        if seq_len > self.block_size:
            raise ValueError(f"Sequence length '{seq_len}' exceeds block size '{self.block_size}'")

        embedding = self.embedding(tokens)
        positional_embedding = self.positional_embedding(seq_len)
        x = embedding + positional_embedding

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)

        logits = x @ self.embedding.weights.t() + self.lm_head_bias

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()

        for _ in range(max_new_tokens):
            tokens_cond = tokens[:, - self.block_size:]
            logits, _ = self(tokens_cond)
            logits = logits[:, -1, :] / max(temperature, 1.0e-06)

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

# %% Functions

def train(args):
    corpus = Path(args.input).read_text("utf-8")
    tokenizer = CharTokenizer(corpus)
    tokens = tokenizer.encode(corpus)

    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]

    train_dataset = NextTokenDataset(train_tokens, args.block_size)
    test_dataset = NextTokenDataset(test_tokens, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_path.with_stem(f"{checkpoint_path.stem}-{timestamp}")
        shutil.copy2(checkpoint_path, backup_path)

        print(f"backed up checkpoint to {backup_path}")

        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint["config"]

        model = TinyGPT(**config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("resumed training from existing checkpoint")

    else:
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"device={device}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"train_tokens={len(train_tokens)} test_tokens={len(test_tokens)}")
    print(f"num_params={sum(parameters.numel() for parameters in model.parameters()):,}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            _, loss = model(x, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if global_step % args.eval_every == 0:
                train_loss = loss.item()
                test_loss = estimate_loss(model, test_loader, device)

                test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")

                print(
                    f"epoch={epoch} step={global_step} train_loss={train_loss:.4f} "
                    f"test_loss={test_loss:.4f} test_ppl={test_ppl:.2f} best_loss={best_loss:.4f}"
                )

                if test_loss < best_loss:
                    best_loss = test_loss

                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "vocab_size": tokenizer.vocab_size,
                            "block_size": args.block_size,
                            "d_model": args.d_model,
                            "num_heads": args.num_heads,
                            "num_layers": args.num_layers,
                        },
                        "stoi": tokenizer.stoi,
                        "itos": tokenizer.itos,
                    }

                    torch.save(checkpoint, args.checkpoint)
                    print(f"saved checkpoint to {args.checkpoint}")

            global_step += 1

    print("Training complete")

    prompt = args.prompt or "\n"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_tokens = model.generate(prompt_tokens, max_new_tokens=5)
    print(tokenizer.decode(output_tokens[0].tolist()))

@torch.no_grad()
def estimate_loss(model: Module, loader: DataLoader, device: torch.device, max_batches: int = 50):
    model.eval()

    losses = []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / max(1, len(losses))

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = TinyGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def encode(text: str) -> list[int]:
        unknown = [char for char in text if char not in tokenizer.stoi]

        if unknown:
            raise ValueError(f"Prompt contains unseen characters: {unknown[:10]}")

        return [tokenizer.stoi[char] for char in text]

    def decode(tokens: list[int]) -> str:
        return "".join(tokenizer.itos[idx] for idx in tokens)

    prompt = args.prompt or "\n"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_tokens = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output_tokens[0].tolist()))

# %% Testing

if __name__ == "__main__":
    import sys

    sys.argv = [
        sys.argv[0],
        "train",
        "--input", kagglehub.dataset_download("edenbd/children-stories-text-corpus") + "/cleaned_merged_fairy_tales_without_eos.txt",
        "--checkpoint", "../checkpoints/tiny_gpt.pt",
        "--epochs", "1",
    ]

    parser = utils.build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)

    if args.command == "evaluate":
        evaluate(args)

# %% End of script
