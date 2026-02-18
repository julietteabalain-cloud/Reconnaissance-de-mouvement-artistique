""" Deep learning models"""

import torch.nn as nn
from torchvision import models as models

import random
import numpy as np
import torch

# fixer seeds pour initialisation (reproductibilité)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# modele resnet : baseline 

def get_resnet18(num_classes, device, freeze_backbone=True, drop=False):
    model = models.resnet18(weights="IMAGENET1K_V1")

    model.fc = nn.Sequential(
        nn.Dropout(0.2) if drop else nn.Identity(),
        nn.Linear(model.fc.in_features, 512)
    )
    model = model.to(device)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model