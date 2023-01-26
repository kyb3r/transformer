import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

data = Path(__file__).parent / "data"

with open(data / "input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars) # 28 in this case

str_to_idx = {s: i for i, s in enumerate(chars)}
idx_to_str = {i: s for s, i in str_to_idx.items()}
    
def encode(text):
    return torch.tensor([str_to_idx[s] for s in text], dtype=torch.long)

def decode(encoded):
    return "".join(idx_to_str[i] for i in encoded.tolist())

def build_dataset(text, block_size):
    data = encode(text)

    X, Y = [], []
    for i in range(0, len(data) - block_size, block_size):
        X.append(data[i:i+block_size])
        Y.append(data[i+1:i+block_size+1])

    X = torch.stack(X)
    Y = torch.stack(Y)

    return TensorDataset(X, Y)


def get_dataloader(config):
    """Returns train, validation, and test dataloaders"""
    dataset = build_dataset(text, config.block_size)

    train_ds, _test_val = random_split(dataset, [0.8, 0.2])
    test_ds, val_ds = random_split(_test_val, [0.5, 0.5])

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    validation_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    return train_dl, validation_dl, test_dl
