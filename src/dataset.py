from matplotlib import pyplot as plt
import pandas as pd
import json
from pathlib import Path
from PIL import Image

# Racine du projet (deepl-projet/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


################## ACCES AU DATASET #################

def load_df_train_test_val():
    df_train = pd.read_csv(DATA_DIR  / "train_metadata.csv")
    df_val   = pd.read_csv(DATA_DIR  / "val_metadata.csv")
    df_test  = pd.read_csv(DATA_DIR  / "test_metadata.csv")

    with open(DATA_DIR  / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    df_train["style_name"]  = df_train["style"].apply(lambda x: mapping["style"][x])
    df_train["artist_name"] = df_train["artist"].apply(lambda x: mapping["artist"][x])
    df_train["genre_name"]  = df_train["genre"].apply(lambda x: mapping["genre"][x])
    df_val["style_name"]  = df_val["style"].apply(lambda x: mapping["style"][x])
    df_val["artist_name"] = df_val["artist"].apply(lambda x: mapping["artist"][x])
    df_val["genre_name"]  = df_val["genre"].apply(lambda x: mapping["genre"][x])
    df_test["style_name"]  = df_test["style"].apply(lambda x: mapping["style"][x])
    df_test["artist_name"] = df_test["artist"].apply(lambda x: mapping["artist"][x])
    df_test["genre_name"]  = df_test["genre"].apply(lambda x: mapping["genre"][x])

    return df_test, df_train, df_val

def load_df():
    df_test, df_train, df_val = load_df_train_test_val()
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    with open(DATA_DIR  / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    df["style_name"]  = df["style"].apply(lambda x: mapping["style"][x])
    df["artist_name"] = df["artist"].apply(lambda x: mapping["artist"][x])
    df["genre_name"]  = df["genre"].apply(lambda x: mapping["genre"][x])
    return df

def load_image(row, split="train", data_root=None):
    img_path = DATA_DIR / split / str(row["style"]) / row["filename"]

    if not img_path.exists():
        raise FileNotFoundError(f"Image non trouvée : {img_path}")

    return Image.open(img_path).convert("RGB")

################### EXPLORATION DES DONNEES ###################

def show_images_by_style(dataset, target_style, n=5):
    count = 0
    for item in dataset:
        if item["style"] == target_style:
            plt.figure(figsize=(3,3))
            plt.imshow(item["image"])
            plt.axis("off")
            plt.title(f"{item['artist']} – {item['style']}")
            plt.show()
            count += 1
            if count == n:
                break