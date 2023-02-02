from models import Config, Block

from torch import nn
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from dataclasses import dataclass

dataset = MNIST(root="data/", download=True, transform=ToTensor(), train=True)
test_dataset = MNIST(root="data/", download=True, transform=ToTensor(), train=False)

print(dataset[0][0].shape)

train_ds, val_ds = random_split(dataset, [0.9, 0.1])

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=256, shuffle=True)


@dataclass
class VisionConfig:
    num_layers = 6
    num_heads: int = 6
    embedding_size: int = 120
    hidden_size: int = 4 * embedding_size

    patch_size = 4
    use_mask = False
    n_patches = (28 // patch_size) ** 2
    block_size = n_patches + 1
    
    dropout = 0.2

    device: str = "cuda"
    learning_rate: int = 3e-4
    epochs: int = 35


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, emb_dimension):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dimension = emb_dimension
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=emb_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        x = self.conv(images)
        x = x.flatten(start_dim=2)  # Flatten the patches into a single vector
        x = x.transpose(
            1, 2
        )  # Transpose so that each patch gets a embedding of size emb_dimension
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_to_patches = PatchEmbedding(config.patch_size, config.embedding_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_size))
        self.pos_embeddings = nn.Parameter(
            torch.randn(config.n_patches + 1, config.embedding_size)
        )
        self.transformer_blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(config.embedding_size * 2, 10),
        )


    
    def forward(self, images):
        B, C, H, W = images.shape

        x = self.image_to_patches(images) # (batch_size, n_patches, emb_dimension)

        # Add the cls token to the beginning of the sequence
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add the position embeddings
        x = x + self.pos_embeddings

        # Pass through the transformer blocks
        x = self.transformer_blocks(x)

        cls_embedding = x[:, 0]
        logits = self.mlp_head(cls_embedding)
        return logits
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

@torch.no_grad()
def check_accuracy(model, dataloaders: list):
    num_correct = 0
    num_samples = 0
    model.eval()

    for dataloader in dataloaders:
        for x, y in dataloader:
            x = x.to(config.device)
            y = y.to(config.device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%"
    )

    model.train()



if __name__ == "__main__":
    config = VisionConfig()

    model = VisionTransformer(config).to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=config.learning_rate, steps_per_epoch=len(train_dl), epochs=config.epochs
    )

    print(model.num_parameters)

    from tqdm import tqdm

    for epoch in range(config.epochs):
        losses = []
        for images, labels in tqdm(train_dl, desc=f"Epoch {epoch}"):
            images = images.to(config.device)
            labels = labels.to(config.device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)


            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.item())
        
        with torch.no_grad():
            model.eval()
            val_losses = []
            for images, labels in val_dl:
                images = images.to(config.device)
                labels = labels.to(config.device)

                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                val_losses.append(loss.item())
            model.train()

            check_accuracy(model, [test_dl])

        print(f"Train Loss: {sum(losses)/len(losses)}")
        print(f"Val Loss: {sum(val_losses)/len(val_losses)}")

