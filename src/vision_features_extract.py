import numpy as np
import cv2
from PIL import Image
from pathlib import Path

############## EXTRACTION DES FEATURES VISUELLES ##############

def compute_sobel_gradients(img: Image.Image):
    """
    Compute Sobel gradients, magnitude and orientation for one image.
    
    Returns:
        magnitude: np.ndarray
        orientation: np.ndarray (radians, in [-pi, pi])
    """
    # Convert to grayscale
    img_gray = np.array(img.convert("L"), dtype=np.float32)

    # Sobel gradients
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)

    return magnitude, orientation


def gradient_histograms(
    magnitude,
    orientation,
    mag_bins=20,
    ori_bins=18
):
    """
    Compute normalized histograms of gradient magnitude and orientation.
    """
    # Magnitude histogram
    mag_hist, _ = np.histogram(
        magnitude.flatten(),
        bins=mag_bins,
        range=(0, np.max(magnitude) + 1e-6),
        density=True
    )

    # Orientation histogram
    ori_hist, _ = np.histogram(
        orientation.flatten(),
        bins=ori_bins,
        range=(-np.pi, np.pi),
        density=True
    )

    return mag_hist, ori_hist


############# AGREGATION PAR STYLE #############

from collections import defaultdict
from tqdm import tqdm


def aggregate_histograms_by_style(
    df,
    load_image_fn,
    DATA_DIR,
    mag_bins=20,
    ori_bins=18,
    max_images_per_style=200
):
    """
    Compute mean gradient histograms per style.

    Returns:
        dict[style] = {
            "mag_hist_mean": np.ndarray,
            "ori_hist_mean": np.ndarray,
            "n_images": int
        }
    """
    mag_hists = defaultdict(list)
    ori_hists = defaultdict(list)

    counters = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        style = row["style_name"]

        if counters[style] >= max_images_per_style:
            continue

        try:
            img = load_image_fn(row, DATA_DIR)
            magnitude, orientation = compute_sobel_gradients(img)
            mag_hist, ori_hist = gradient_histograms(
                magnitude,
                orientation,
                mag_bins=mag_bins,
                ori_bins=ori_bins
            )

            mag_hists[style].append(mag_hist)
            ori_hists[style].append(ori_hist)
            counters[style] += 1

        except Exception:
            continue

    results = {}
    for style in mag_hists:
        results[style] = {
            "mag_hist_mean": np.mean(mag_hists[style], axis=0),
            "ori_hist_mean": np.mean(ori_hists[style], axis=0),
            "n_images": len(mag_hists[style])
        }

    return results

############## COMPARAISON DES STYLES ##################


def compute_scores(selected_styles, style_gradients):
    score = {}
    for style in selected_styles:
        mag_hist_mean = style_gradients[style]["mag_hist_mean"]
        weights = np.arange(len(mag_hist_mean))
        sharpness_score = np.sum(weights * mag_hist_mean)
        score[style] = sharpness_score
    scores = np.array(list(score.values()))
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    sorted_items = sorted(
        score.items(),
        key=lambda item: item[1],
        reverse=True
    )    
    sorted_styles = [item[0] for item in sorted_items]
    sorted_scores = [item[1] for item in sorted_items]
    sorted_scores = np.array(sorted_scores)
    sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
    plt.figure(figsize=(10,6))
    plt.barh(sorted_styles, sorted_scores)
    plt.xlabel("Score de netteté des contours")
    plt.title("Comparaison de la netteté des contours par style")
    plt.show()


############## VISUALISATION DES HISTOGRAMMES #############

import matplotlib.pyplot as plt

def plot_single_sobel(img: Image.Image):
    magnitude, orientation = compute_sobel_gradients(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(magnitude, cmap="gray")
    axes[1].set_title("Magnitude du gradient")
    axes[1].axis("off")

    axes[2].imshow(orientation, cmap="hsv")
    axes[2].set_title("Orientation du gradient")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_mean_gradient_histograms(style_gradients, style_list):
    """
    Compare mean gradient histograms for selected styles.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    for style in style_list:
        axes[0].plot(style_gradients[style]["mag_hist_mean"], label=style)
        axes[1].plot(style_gradients[style]["ori_hist_mean"], label=style)

    axes[0].set_title("Histogramme moyen des magnitudes")
    axes[1].set_title("Histogramme moyen des orientations")
    axes[1].set_xlabel("Bins")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import random

def plot_sobel_examples(
    df,
    load_image_fn,
    DATA_DIR,
    styles,
    n_images_per_style=3,
):
    """
    Visualize original image + Sobel magnitude for several styles.

    Args:
        df: DataFrame containing at least ["style_name"]
        load_image_fn: function(row) -> PIL.Image
        styles: list of style names to visualize
        n_images_per_style: number of images per style
    """
    rows = len(styles)
    cols = n_images_per_style

    fig, axes = plt.subplots(
        rows, cols * 2, figsize=(4 * cols, 3 * rows)
    )

    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, style in enumerate(styles):
        df_style = df[df["style_name"] == style]
        samples = df_style.sample(
            min(n_images_per_style, len(df_style)),
            random_state=42
        )

        for j, (_, row) in enumerate(samples.iterrows()):
            img = load_image_fn(row,DATA_DIR)
            magnitude, _ = compute_sobel_gradients(img)

            # Image originale
            axes[i, 2*j].imshow(img)
            axes[i, 2*j].set_title(f"{style}\nOriginal")
            axes[i, 2*j].axis("off")

            # Magnitude Sobel
            axes[i, 2*j + 1].imshow(magnitude, cmap="gray")
            axes[i, 2*j + 1].set_title("Sobel magnitude")
            axes[i, 2*j + 1].axis("off")

    plt.tight_layout()
    plt.show()


################## HOG, HSV FEATURES EXTRACTION ####################
from skimage.feature import hog


def extract_hog(img: Image.Image,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                orientations=9):
    """
    Extract HOG descriptor from a PIL image.
    """
    img_gray = np.array(img.convert("L"))

    hog_features = hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False
    )

    return hog_features

def extract_hsv_histogram(img: Image.Image, bins=(16, 16, 16)):
    """
    HSV color histogram (normalized).
    """
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=bins,
        ranges=[0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lab_histogram(img: Image.Image, bins=(16, 16, 16)):
    """
    Lab color histogram (normalized).
    """
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    hist = cv2.calcHist(
        [lab],
        channels=[0, 1, 2],
        mask=None,
        histSize=bins,
        ranges=[0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(img: Image.Image):
    """
    Global feature vector for one image.
    """
    img = img.resize((224, 224))
    hog_feat = extract_hog(img)
    hsv_feat = extract_hsv_histogram(img)
    lab_feat = extract_lab_histogram(img)

    features = np.concatenate([
        hog_feat,
        hsv_feat,
        lab_feat
    ])

    return features


def extract_features_dataset(df, load_image_fn, DATA_DIR):
    """
    Extract features for all images in the dataframe.
    """
    X = []
    y = []
    filenames = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = load_image_fn(row,DATA_DIR)
            feat = extract_features(img)

            X.append(feat)
            y.append(row["style_name"])
            filenames.append(row["filename"])

        except Exception:
            continue

    X = np.stack(X)
    y = np.array(y)

    return X, y, filenames




############ Analyse des distances entre styles ############

from scipy.spatial.distance import pdist, squareform
import pandas as pd

def visualize_style_distances(X, y):
    "quels styles sont visuellement proches avec HOG + couleurs ?"
    df_feat = pd.DataFrame(X)
    df_feat["style"] = y
    mean_by_style = df_feat.groupby("style").mean()

    dist_matrix = squareform(pdist(mean_by_style.values, metric="euclidean"))
    dist_df = pd.DataFrame(
        dist_matrix,
        index=mean_by_style.index,
        columns=mean_by_style.index
    )
    return dist_df

def heatmap_style_distances(dist_df):

    plt.figure(figsize=(10, 8))
    plt.imshow(dist_df, cmap="viridis")
    plt.colorbar(label="Distance euclidienne")
    plt.xticks(ticks=np.arange(len(dist_df)), labels=dist_df.index, rotation=90)
    plt.yticks(ticks=np.arange(len(dist_df)), labels=dist_df.index)
    plt.title("Matrice de distance entre styles (HOG + couleurs)")
    plt.tight_layout()
    plt.show()
################## PALETTES DE COULEURS ####################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm


def compute_style_palette(df, style_name, load_image_fn, DATA_DIR,
                          n_colors=5, resize=(150,150)):
    """
    Compute dominant color palette for a given style.
    """
    print(f"Computing palette for style: {style_name}")
    df_style = df[df["style_name"] == style_name]

    pixels = []

    for _, row in tqdm(df_style.iterrows(), total=len(df_style)):
        try:
            img = load_image_fn(row,DATA_DIR)
            img = np.array(img)
            img = cv2.resize(img, resize)
            print(img.shape, "after resize")
            # Convert to Lab
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            print(img_lab.shape, "after color convert")
            img_lab = img_lab.reshape(-1, 3)
            print(img_lab.shape, "after reshape")
            pixels.append(img_lab)
            print(len(pixels), "images processed for palette")

        except Exception:
            continue

    pixels = np.vstack(pixels)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors_lab = kmeans.cluster_centers_

    # Convert back to RGB for display
    colors_lab_uint8 = np.uint8(colors_lab.reshape(1, -1, 3))
    colors_rgb = cv2.cvtColor(colors_lab_uint8, cv2.COLOR_LAB2RGB)[0]

    return colors_rgb


def display_palette(colors, style_name):

    n_colors = len(colors)
    palette = np.zeros((50, 50*n_colors, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        palette[:, i*50:(i+1)*50, :] = color

    plt.figure(figsize=(8,2))
    plt.imshow(palette)
    plt.title(f"Palette dominante - {style_name}")
    plt.axis("off")
    plt.show()


def compute_and_display_all_style_palettes(
    df,
    load_image_fn,
    DATA_DIR,
    n_colors=5,
    resize=(150,150),
    max_images_per_style=200
):
    """
    Compute and display dominant color palettes for all styles.
    """

    styles = sorted(df["style_name"].unique())
    style_palettes = {}

    n_styles = len(styles)
    fig, axes = plt.subplots(n_styles, 1, figsize=(8, 2*n_styles))

    if n_styles == 1:
        axes = [axes]

    for idx, style in enumerate(styles):

        df_style = df[df["style_name"] == style].head(max_images_per_style)

        pixels = []
        for _, row in tqdm(df_style.iterrows(), total=len(df_style)):
            try:
                img = load_image_fn(row, DATA_DIR)
                print(img.shape)
                img = cv2.resize(img, resize)

                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_lab = img_lab.reshape(-1, 3)

                # échantillonnage pour réduire mémoire
                if img_lab.shape[0] > 2000:
                    idx_sample = np.random.choice(
                        img_lab.shape[0], 2000, replace=False
                    )
                    img_lab = img_lab[idx_sample]

                pixels.append(img_lab)

            except:
                continue

        pixels = np.vstack(pixels)

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors_lab = kmeans.cluster_centers_

        # Convertir vers RGB
        colors_lab_uint8 = np.uint8(colors_lab.reshape(1, -1, 3))
        colors_rgb = cv2.cvtColor(colors_lab_uint8, cv2.COLOR_LAB2RGB)[0]

        style_palettes[style] = colors_rgb

        # Création bande palette
        palette_img = np.zeros((50, 50*n_colors, 3), dtype=np.uint8)
        for i, color in enumerate(colors_rgb):
            palette_img[:, i*50:(i+1)*50, :] = color

        axes[idx].imshow(palette_img)
        axes[idx].set_title(style)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

    return style_palettes


def compute_style_hsv_stats(df, style_name, load_image_fn, DATA_DIR):

    df_style = df[df["style_name"] == style_name]

    h_vals, s_vals, v_vals = [], [], []

    for _, row in df_style.iterrows():
        try:
            img = load_image_fn(row, DATA_DIR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h_vals.append(hsv[:,:,0].mean())
            s_vals.append(hsv[:,:,1].mean())
            v_vals.append(hsv[:,:,2].mean())

        except:
            continue

    return {
        "mean_hue": np.mean(h_vals),
        "mean_saturation": np.mean(s_vals),
        "mean_value": np.mean(v_vals)
    }

from scipy.spatial.distance import pdist, squareform


def compute_hsv_stats_all_styles(df, load_image_fn, DATA_DIR):
    """
    Compute mean HSV statistics for each style.
    Returns dict: {style: [mean_H, mean_S, mean_V]}
    """

    styles = df["style_name"].unique()
    style_vectors = {}

    for style in styles:
        df_style = df[df["style_name"] == style]

        h_vals, s_vals, v_vals = [], [], []

        for _, row in tqdm(df_style.iterrows(), total=len(df_style), leave=False):
            try:
                img = load_image_fn(row, DATA_DIR)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                h_vals.append(hsv[:,:,0].mean())
                s_vals.append(hsv[:,:,1].mean())
                v_vals.append(hsv[:,:,2].mean())

            except:
                continue

        style_vectors[style] = [
            np.mean(h_vals),
            np.mean(s_vals),
            np.mean(v_vals)
        ]

    return style_vectors

def plot_style_distance_heatmap(style_vectors, title="Distance HSV entre styles"):

    styles = list(style_vectors.keys())
    vectors = np.array(list(style_vectors.values()))

    # Distance euclidienne
    dist_matrix = squareform(pdist(vectors, metric="euclidean"))

    plt.figure(figsize=(10,8))
    sns.heatmap(
        dist_matrix,
        xticklabels=styles,
        yticklabels=styles,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return dist_matrix


######### PCA ################333
from sklearn.decomposition import PCA   
import joblib
from sklearn.preprocessing import StandardScaler
import os

def apply_block_pca_for_model(
    X_train,
    X_val=None,
    X_test=None,
    hog_dim=6084,
    hsv_dim=4096,
    lab_dim=4096,
    hog_pca_dim=150,
    hsv_pca_dim=75,
    lab_pca_dim=75
):


    X_train_hog = X_train[:, :hog_dim]
    X_train_hsv = X_train[:, hog_dim:hog_dim+hsv_dim]
    X_train_lab = X_train[:, hog_dim+hsv_dim:]

    if X_val is not None:
        X_val_hog = X_val[:, :hog_dim]
        X_val_hsv = X_val[:, hog_dim:hog_dim+hsv_dim]
        X_val_lab = X_val[:, hog_dim+hsv_dim:]

    if X_test is not None:
        X_test_hog = X_test[:, :hog_dim]
        X_test_hsv = X_test[:, hog_dim:hog_dim+hsv_dim]
        X_test_lab = X_test[:, hog_dim+hsv_dim:]

    scaler_hog = StandardScaler().fit(X_train_hog)
    scaler_hsv = StandardScaler().fit(X_train_hsv)
    scaler_lab = StandardScaler().fit(X_train_lab)

    X_train_hog = scaler_hog.transform(X_train_hog)
    X_train_hsv = scaler_hsv.transform(X_train_hsv)
    X_train_lab = scaler_lab.transform(X_train_lab)

    if X_val is not None:
        X_val_hog = scaler_hog.transform(X_val_hog)
        X_val_hsv = scaler_hsv.transform(X_val_hsv)
        X_val_lab = scaler_lab.transform(X_val_lab)

    if X_test is not None:
        X_test_hog = scaler_hog.transform(X_test_hog)
        X_test_hsv = scaler_hsv.transform(X_test_hsv)
        X_test_lab = scaler_lab.transform(X_test_lab)


    pca_hog = PCA(n_components=hog_pca_dim).fit(X_train_hog)
    pca_hsv = PCA(n_components=hsv_pca_dim).fit(X_train_hsv)
    pca_lab = PCA(n_components=lab_pca_dim).fit(X_train_lab)

    X_train_hog_pca = pca_hog.transform(X_train_hog)
    X_train_hsv_pca = pca_hsv.transform(X_train_hsv)
    X_train_lab_pca = pca_lab.transform(X_train_lab)

    if X_val is not None:
        X_val_hog_pca = pca_hog.transform(X_val_hog)
        X_val_hsv_pca = pca_hsv.transform(X_val_hsv)
        X_val_lab_pca = pca_lab.transform(X_val_lab)

    if X_test is not None:
        X_test_hog_pca = pca_hog.transform(X_test_hog)
        X_test_hsv_pca = pca_hsv.transform(X_test_hsv)
        X_test_lab_pca = pca_lab.transform(X_test_lab)


    X_train_final = np.concatenate(
        [X_train_hog_pca, X_train_hsv_pca, X_train_lab_pca],
        axis=1
    )

    if X_val is not None:
        X_val_final = np.concatenate(
            [X_val_hog_pca, X_val_hsv_pca, X_val_lab_pca],
            axis=1
        )
    else:
        X_val_final = None

    if X_test is not None:
        X_test_final = np.concatenate(
            [X_test_hog_pca, X_test_hsv_pca, X_test_lab_pca],
            axis=1
        )
    else:
        X_test_final = None


    os.makedirs("/content/drive/MyDrive/models", exist_ok=True)

    joblib.dump(scaler_hog, "/content/drive/MyDrive/models/scaler_hog.pkl")
    joblib.dump(scaler_hsv, "/content/drive/MyDrive/models/scaler_hsv.pkl")
    joblib.dump(scaler_lab, "/content/drive/MyDrive/models/scaler_lab.pkl")

    joblib.dump(pca_hog, "/content/drive/MyDrive/models/pca_hog.pkl")
    joblib.dump(pca_hsv, "/content/drive/MyDrive/models/pca_hsv.pkl")
    joblib.dump(pca_lab, "/content/drive/MyDrive/models/pca_lab.pkl")

    print("Variance HOG :", np.sum(pca_hog.explained_variance_ratio_))
    print("Variance HSV :", np.sum(pca_hsv.explained_variance_ratio_))
    print("Variance LAB :", np.sum(pca_lab.explained_variance_ratio_))

    return {
        "X_train": X_train_final,
        "X_val": X_val_final,
        "X_test": X_test_final
    }

def build_pca_dataframe(X_pca, df_split, split_name):
    
    hog_cols = [f"hog_pc_{i}" for i in range(150)]
    hsv_cols = [f"hsv_pc_{i}" for i in range(75)]
    lab_cols = [f"lab_pc_{i}" for i in range(75)]

    feature_columns = hog_cols + hsv_cols + lab_cols
    df_features = pd.DataFrame(X_pca, columns=feature_columns)

    df_meta = pd.DataFrame({
        "filename": df_split["filename"].values,
        "split": split_name,
        "style_name": df_split["style_name"].values
    })
    
    df_final = pd.concat([df_meta.reset_index(drop=True),
                          df_features.reset_index(drop=True)], axis=1)
    
    return df_final
