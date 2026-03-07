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


##################### Modele resnet : baseline #################################

def get_resnet18(num_classes, device, freeze_backbone=True, drop=False):
    model = models.resnet18(weights="IMAGENET1K_V1")

    model.fc = nn.Sequential(
        nn.Dropout(0.2) if drop else nn.Identity(),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


################### Comparaison des architectures ################################

def get_model(model_name, num_classes=23, dropout_p=0.2, freeze_backbone=True):

    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnext50":
        model = models.resnext50_32x4d(weights=models.ResNext50_32X4D_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError("Model name not recognized")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layer
        if "resnet" in model_name or "resnext" in model_name:
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


########################### Approche hybride ###############################
# Pour le modèle avec fusion des 3 branches (normal, haute fréquence, basse fréquence)

class MultiBranchResNet18(nn.Module):
    def __init__(self, num_classes, drop=False):
        super().__init__()
        
        # 3 backbones ResNet18 indépendants, chacun pré-entraîné
        resnet_orig = models.resnet18(weights="IMAGENET1K_V1")
        resnet_hf   = models.resnet18(weights="IMAGENET1K_V1")
        resnet_lf   = models.resnet18(weights="IMAGENET1K_V1")
        
        # On retire la dernière couche FC de chaque branche
        self.branch_orig = nn.Sequential(*list(resnet_orig.children())[:-1])
        self.branch_hf   = nn.Sequential(*list(resnet_hf.children())[:-1])
        self.branch_lf   = nn.Sequential(*list(resnet_lf.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2) if drop else nn.Identity(),
            nn.Linear(512 * 3, num_classes)
        )

    def forward(self, x_orig, x_hf, x_lf):
        feat_orig = self.branch_orig(x_orig).flatten(1)  
        feat_hf   = self.branch_hf(x_hf).flatten(1)      
        feat_lf   = self.branch_lf(x_lf).flatten(1)     
        
        feat = torch.cat([feat_orig, feat_hf, feat_lf], dim=1)  
        return self.classifier(feat)


def get_multibranch_resnet18(num_classes, device, freeze_backbone=True, drop=False):
    model = MultiBranchResNet18(num_classes=num_classes, drop=drop)
    model = model.to(device)
    
    if freeze_backbone:
        for branch in [model.branch_orig, model.branch_hf, model.branch_lf]:
            for param in branch.parameters():
                param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model