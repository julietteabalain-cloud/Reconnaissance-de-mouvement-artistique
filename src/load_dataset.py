"""Script pour charger un sous-ensemble du dataset WikiArt et l'enregistrer sur Google Drive
   Ne pas utiliser c'est le code utilisé dans Google Colab
"""


from pathlib import Path
import csv
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
from datasets import load_dataset
from google.colab import drive

########### enregistrement d'un sous-ensemble du dataset WikiArt sur google Drive ###########

drive.mount('/content/drive')

dataset = load_dataset(
    "huggan/wikiart",
    split="train",
)

MAX_PER_STYLE = 400

# Compteur par style
counter = defaultdict(int)

def select_fn(example):
    style = example["style"]
    if counter[style] < MAX_PER_STYLE:
        counter[style] += 1
        return True
    return False

subset = dataset.shuffle(seed=42).filter(select_fn)

# Récupère les indices du dataset filtré
indices = list(range(len(subset)))

# Shuffle pour mélanger
random.seed(42)
random.shuffle(indices)

# Split 70/15/15
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

BASE_DIR = Path("/content/drive/MyDrive/DeepLearning/WikiArt_Subset")
BASE_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

for split_name, split_indices in splits.items():
    csv_path = BASE_DIR / f"{split_name}_metadata.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "style", "artist", "genre"]
        )
        writer.writeheader()

        for idx in tqdm(split_indices, desc=f"Saving {split_name}"):
            item = subset[idx]

            style = str(item["style"])
            artist = item["artist"]
            genre = item["genre"]
            image = item["image"].convert("RGB")

            out_dir = BASE_DIR / split_name / style
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{idx}.jpg"
            image_path = out_dir / filename

            image.save(
                image_path,
                format="JPEG",
                quality=90,
                subsampling=2,
                optimize=True
            )

            writer.writerow({
                "filename": filename,
                "style": style,
                "artist": artist,
                "genre": genre
            })

