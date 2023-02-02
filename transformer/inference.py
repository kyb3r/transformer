import torch
from utils import get_dataloader, decode, encode
from models import Config, Transformer


config = Config()
model = Transformer(config).to(config.device)
model.eval()

# load the model from the checkpoint:
model.load_state_dict(torch.load("model.pt"))
print("Loaded model from checkpoint")

# ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)

# for token in model.generate(ctx, max_new_tokens=100, stream=True):
#     print(decode(token[0]), end="", flush=True)

# Get sentence embedding:
ctx = encode("Hello world").unsqueeze(0).to(config.device)
print(ctx)
hidden_states = model(ctx, return_hidden_states=True)  # (1, 11, 384)

last_hidden_embedding = hidden_states[:, -1]
print(last_hidden_embedding.shape)


mean_hidden_embedding = hidden_states.mean(dim=1)
print(mean_hidden_embedding.shape)  # ( 384)
