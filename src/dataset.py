from matplotlib import pyplot as plt
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import seaborn as sns

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
    
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

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

def load_image(row):
    img_path = DATA_DIR / row["split"] / str(row["style"]) / row["filename"]
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

def visualize_data(df, n=9):
    sample = df.sample(n)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        ax.imshow(load_image(row))
        ax.set_title(row["style_name"], fontsize=9)
        ax.axis("off")
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

def count_nb_genre_per_style(df):
    artists_per_style = (
    df
    .groupby("genre_name")["artist_name"]
    .nunique()
    .sort_values()
    )

    artists_per_style.plot.barh(figsize=(6,8))
    plt.title("Nombre de genress par style")
    plt.show()

def repartition_genre_per_style(df):
    ct = pd.crosstab(
        df["style_name"],
        df["genre_name"],
        normalize="index"
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        ct,
        cmap="viridis",
        cbar_kws={"label": "Proportion"}
    )

    plt.title("Répartition des genres par style artistique")
    plt.xlabel("Genre")
    plt.ylabel("Style")
    plt.tight_layout()
    plt.show()

def repartition_artist_per_style(df):
    ct = pd.crosstab(
        df["style_name"],
        df["artist_name"],
        normalize="index"
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        ct,
        cmap="viridis",
        cbar_kws={"label": "Proportion"}
    )

    plt.title("Répartition des artistes par style artistique")
    plt.xlabel("Artiste")
    plt.ylabel("Style")
    plt.tight_layout()
    plt.show()

from scipy.spatial.distance import pdist, squareform


def similarities_between_style_based_on_genre(df):
    
    ct = pd.crosstab(
        df["style_name"],
        df["genre_name"],
        normalize="index"
    )

    dist_matrix = squareform(
        pdist(ct.values, metric="euclidean")
    )

    dist_df = pd.DataFrame(
        dist_matrix,
        index=ct.index,
        columns=ct.index
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_df, cmap="viridis")
    plt.title("Distance entre styles basée sur la distribution des genres")
    plt.show()

def styles_cluster_based_on_genre(df):
    ct = pd.crosstab(
        df["style_name"],
        df["genre_name"],
        normalize="index"
    )
    sns.clustermap(
        ct,
        cmap="viridis",
        figsize=(12, 8),
        metric="euclidean",
        method="ward"
    )
    plt.suptitle("Clustering des styles basé sur la distribution des genres", y=1.02)
    plt.show()

def unknown_genre_per_style(df):
    unknown_rate = (
        df["genre_name"].str.lower().str.contains("unknown")
        .groupby(df["style_name"])
        .mean()
    )

    unknown_rate.sort_values(ascending=False).plot(kind="bar", figsize=(12,4))
    plt.title("Proportion de genres inconnus par style")
    plt.show()

def unknown_artist_per_style(df):
    unknown_rate = (
        df["artist_name"].str.lower().str.contains("unknown")
        .groupby(df["style_name"])
        .mean()
    )

    unknown_rate.sort_values(ascending=False).plot(kind="bar", figsize=(12,4))
    plt.title("Proportion d'artiste inconnus par style")
    plt.show()


def observe_common_artist(df_train, df_test, df_val):
    # Listes d'artistes par split
    train_artists = set(df_train["artist_name"])
    val_artists = set(df_val["artist_name"])
    test_artists = set(df_test["artist_name"])

    # Artistes présents dans train ET test
    overlap_train_test = train_artists & test_artists
    overlap_train_val  = train_artists & val_artists
    overlap_val_test   = val_artists & test_artists

    print("Artistes communs train-test:", len(overlap_train_test))
    print("Artistes communs train-val:", len(overlap_train_val))
    print("Artistes communs val-test:", len(overlap_val_test))

    overlaps = [len(overlap_train_test), len(overlap_train_val), len(overlap_val_test)]
    labels = ["train-test", "train-val", "val-test"]

    plt.bar(labels, overlaps)
    plt.ylabel("Nb artistes communs")
    plt.title("Fuite potentielle d'artistes entre splits")
    plt.show()

import random
def split_artist(df, df_train, df_val, df_test):

    random.seed(42)
    MAX_PER_STYLE = 400
    TRAIN_ARTIST_FRACTION = 0.6  # fraction d'artistes par style pour le train

    train_rows, val_rows, test_rows = [], [], []

    for style, group in df.groupby("style_name"):
        artists = list(group["artist_name"].unique())
        random.shuffle(artists)

        n_train_artists = max(1, int(len(artists) * TRAIN_ARTIST_FRACTION))
        train_artists = artists[:n_train_artists]
        remaining_artists = artists[n_train_artists:]

        # Sélection des images pour train, sans dépasser MAX_PER_STYLE
        train_group = group[group["artist_name"].isin(train_artists)]
        if len(train_group) > MAX_PER_STYLE:
            train_group = train_group.sample(MAX_PER_STYLE, random_state=42)
        
        train_rows.append(train_group)

        # Split restant en val/test (50/50)
        remaining_group = group[group["artist_name"].isin(remaining_artists)]
        n_val = len(remaining_group) // 2
        val_rows.append(remaining_group.iloc[:n_val])
        test_rows.append(remaining_group.iloc[n_val:])

    # Concaténation finale
    df_train = pd.concat(train_rows).reset_index(drop=True)
    df_val   = pd.concat(val_rows).reset_index(drop=True)
    df_test  = pd.concat(test_rows).reset_index(drop=True)

    # Vérification
    print("Artistes communs train-test:", len(set(df_train["artist_name"]) & set(df_test["artist_name"])))
    print("Artistes communs train-val :", len(set(df_train["artist_name"]) & set(df_val["artist_name"])))

    # Nombre d'images par split
    print("Train :", len(df_train), "Val :", len(df_val), "Test :", len(df_test))
    return df_train, df_val, df_test