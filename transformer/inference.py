import torch
from utils import get_dataloader, decode
from models import Config, Transformer


config = Config()
model = Transformer(config).to(config.device)
model.eval()

# load the model from the checkpoint:
model.load_state_dict(torch.load("model.pt"))
print("Loaded model from checkpoint")

ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)

for token in model.generate(ctx, max_new_tokens=10000, stream=True):
    print(decode(token[0]), end="", flush=True)