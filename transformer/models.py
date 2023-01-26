from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import vocab_size


@dataclass
class Config:
    batch_size: int = 64
    num_layers: int = 6
    num_heads: int = 6
    embedding_size: int = 384
    hidden_size: int = 4 * embedding_size
    vocab_size: int = vocab_size
    block_size: int = 256
    dropout: float = 0.2
    device: str = "cuda"
    learning_rate: int = 1e-3
    epochs: int = 40


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_size = config.embedding_size // config.num_heads
        self.key = nn.Linear(config.embedding_size, head_size, bias=False)
        self.query = nn.Linear(config.embedding_size, head_size, bias=False)
        self.value = nn.Linear(config.embedding_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, block_size, embedding_size)

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        affinity = query @ key.transpose(-1, -2) / (C**0.5)
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        affinity = torch.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)

        return affinity @ value

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.project = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        heads = [head(x) for head in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.project(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multi_headed_attention = MultiHeadedAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embedding_size)
        self.ln2 = nn.LayerNorm(config.embedding_size)
    
    def forward(self, x):
        x = x + self.multi_headed_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = cfg = config
        self.device = config.device
        self.pos_embeddings = nn.Embedding(config.block_size, config.embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.transformer_blocks = nn.Sequential(*[
            Block(config) 
            for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, x):
        B, T = x.shape  # (batch_size, block_size)

        position_indexes = torch.arange(T, device=self.device)
        x = self.pos_embeddings(position_indexes) + self.token_embeddings(x)
        # x --> (batch_size, block_size, embedding_size)
        x = self.transformer_blocks(x)
        logits = self.lm_head(x)
        return logits

    def loss_function(self, logits, y):
        B, T, C = logits.shape
        logits = logits.view(-1, C)
        targets = y.view(-1)
        return F.cross_entropy(logits, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)
        return optimizer, scheduler

    def generate(self, idx, max_new_tokens=100):
        block_size = self.config.block_size
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    from utils import get_dataloader, decode, vocab_size

    config = Config()
    config.vocab_size = vocab_size

    train, validation = get_dataloader(config)

    model = Transformer(config).to(config.device)
    print(f"Number of parameters: {model.num_parameters}")
    optim, scheduler = model.configure_optimizers()

    train_losses_all = []
    val_losses_all = []

    for epoch in range(config.epochs):
        losses = []
        with tqdm(train, unit="batch", desc=f"Epoch - {epoch}") as train:
            for x, y in train:
                x = x.to(config.device) 
                y = y.to(config.device)
                
                logits = model(x)
                loss = model.loss_function(logits, y)

                optim.zero_grad()
                loss.backward()
                optim.step()
                

                losses.append(loss.item())
            print(f"Epoch - {epoch}: Loss: {np.mean(losses)}")
            train_losses_all.append(np.mean(losses))

        val_losses = []
        with torch.no_grad():
            for x, y in validation:
                x = x.to(config.device)
                y = y.to(config.device)
                logits = model(x)
                loss = model.loss_function(logits, y)
                val_losses.append(loss.item())

        print(f"Epoch - {epoch}: Val Loss: {np.mean(val_losses)}")
        val_losses_all.append(np.mean(val_losses))

    import matplotlib.pyplot as plt

    # plot losses and save image to file:
    plt.plot(train_losses_all, label="train")
    plt.plot(val_losses_all, label="val")
    plt.legend()
    plt.savefig("losses.png")

    starting_context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    tokens = model.generate(starting_context, max_new_tokens=1000)[0]
    print(decode(tokens))
