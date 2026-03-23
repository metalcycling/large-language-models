# %% Modules

# %% Classes

class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(list(set(corpus)))

        self.vocab_size = len(chars)
        self.stoi = {}
        self.itos = {}

        for char in corpus:
            if char not in self.stoi:
                self.stoi[char] = len(self.stoi)

        for char, idx in self.stoi.items():
            self.itos[idx] = char

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[token] for token in tokens])

# %% Testing

if __name__ == "__main__":
    corpus = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.'\"-"
    tokenizer = CharTokenizer(corpus)

    text = "This is a test I'm trying"
    print(tokenizer.encode(text))

# %% End of script
