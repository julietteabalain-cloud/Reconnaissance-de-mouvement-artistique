""" Création d'un dataset adapté au deep learning avec Pytorch """

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ArtDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.image_root / str(row["style"]) / row["filename"]

        image = Image.open(img_path).convert("RGB")

        label = row["style_encoded"]  
        
        if self.transform:
            image = self.transform(image)

        return image, label

#Pour le modèle multibranches :

class ArtDatasetMultiBranch(Dataset):
    """
    Dataset qui retourne 3 versions de chaque image (orig, HF, LF)
    """
    def __init__(self, df, image_root, transform_orig, transform_hf, transform_lf):
        # On réutilise ArtDataset 3 fois
        self.dataset_orig = ArtDataset(df, image_root, transform=transform_orig)
        self.dataset_hf   = ArtDataset(df, image_root, transform=transform_hf)
        self.dataset_lf   = ArtDataset(df, image_root, transform=transform_lf)

    def __len__(self):
        return len(self.dataset_orig)

    def __getitem__(self, idx):
        img_orig, label = self.dataset_orig[idx]
        img_hf,   _     = self.dataset_hf[idx]
        img_lf,   _     = self.dataset_lf[idx]

        return img_orig, img_hf, img_lf, label