import torchvision.models as models
import torch.nn as nn
from config import Config

def get_model(weights=None):
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)
    return model