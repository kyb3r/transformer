from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Config:
    batch_size: int = 32
    num_layers: int = 1
    num_heads: int = 1
    embedding_size: int = 32
    hidden_size: int = 30
    vocab_size: int = 100
    block_size: int = 10
    device: str = "cpu"


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        block_size = config.block_size
        head_size = config.embedding_size // config.num_heads
        self.key = nn.Linear(config.embedding_size, head_size)
        self.query = nn.Linear(config.embedding_size, head_size)
        self.value = nn.Linear(config.embedding_size, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.position_embeddings = nn.Embedding(config.block_size, config.embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)

        self.register_buffer("position_ids", torch.arange(config.block_size).to(self.device))

        print(self.position_ids)
        print(self.position_ids.shape)


    def forward(self, x):
        B, T = x.shape # (batch_size, block_size)
        x = self.position_embeddings(self.position_ids) + self.token_embeddings(x)
        # x --> (batch_size, block_size, embedding_size)

        return x

if __name__ == "__main__":
    from utils import get_dataloader


    config = Config()

    train, validation, test = get_dataloader(config)


    model = Transformer(config)

    for x, y in train:
        print(x.shape)
        print(model(x).shape)
        break

    tril = torch.tril(torch.ones(10, 10))