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

def visualize_data(df):
    sample = df.sample(9, random_state=42)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        img = load_image(row, split="train")
        ax.imshow(img)
        ax.set_title(row["style_name"], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def variability_inter_style(style, df, n):
    sample = df[df["style_name"] == style].sample(n)
    fig, axes = plt.subplots(3, 3, figsize=(8,8))
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        ax.imshow(load_image(row))
        ax.axis("off")
    plt.suptitle(f"Variabilité intra-style : {style}")
    plt.show()

def variation_inter_style(df):
    styles = df["style_name"].unique()[:4]
    fig, axes = plt.subplots(len(styles), 4, figsize=(10,8))

    for i, style in enumerate(styles):
        samples = df[df["style_name"] == style].sample(4)
        for j, (_, row) in enumerate(samples.iterrows()):
            axes[i, j].imshow(load_image(row))
            axes[i, j].axis("off")
        axes[i,0].set_ylabel(style)

    plt.show()

def visualize_style_repartition(df):
    plt.figure(figsize=(12,4))
    df["style_name"].value_counts().plot.bar()
    plt.title("Distribution des styles (train)")
    plt.ylabel("Nombre d'images")
    plt.show()

def visualize_genre_repartition(df, number_of_genre):
    df["genre_name"].value_counts().head(10)
    plt.figure(figsize=(12,4))
    df["genre_name"].value_counts().plot.bar()
    plt.title("Distribution des genres")
    plt.ylabel("Nombre d'images")
    plt.show()

def visualize_artist_repartition(df):
    df["artist_name"].value_counts().head(10)
    plt.figure(figsize=(12,4))
    df["artist_name"].value_counts().plot.bar()
    plt.title("Distribution des artistes")
    plt.ylabel("Nombre d'images")
    plt.show()

def count_nb_artist_per_style(df):
    artists_per_style = (
    df
    .groupby("style_name")["artist_name"]
    .nunique()
    .sort_values()
    )

    artists_per_style.plot.barh(figsize=(6,8))
    plt.title("Nombre d'artistes par style")
    plt.show()
