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
