from utils import get_dataloader
from models import Config


config = Config()
train, validation, test = get_dataloader(config)
