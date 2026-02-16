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

from sklearn.decomposition import PCA

def pca_style_features(X, y, n_components=2):

    df_feat = pd.DataFrame(X)
    df_feat["style"] = y
    mean_by_style = df_feat.groupby("style").mean()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(mean_by_style.values)

    plt.figure(figsize=(10, 8))
    for i, style in enumerate(mean_by_style.index):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], label=style)
        plt.text(X_pca[i, 0], X_pca[i, 1], style)

    plt.title("PCA des features par style (HOG + couleurs)")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



############ PCA pour entrainement du SVM ############
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
def apply_pca_for_model(X_train, X_val=None, X_test=None, n_components=300):
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    if X_val is not None:
        X_val_pca = pca.transform(X_val_scaled)
    if X_test is not None:
        X_test_pca = pca.transform(X_test_scaled)
    os.makedirs("/content/drive/MyDrive/models", exist_ok=True)

    joblib.dump(scaler, "/content/drive/MyDrive/models/scaler.pkl")
    joblib.dump(pca, "/content/drive/MyDrive/models/pca.pkl")
    print("Variance expliquée totale :", np.sum(pca.explained_variance_ratio_))

    return {
        "X_train": X_train_pca,
        "X_val": X_val_pca if X_val is not None else None,
        "X_test": X_test_pca if X_test is not None else None,
        "scaler": scaler,
        "pca": pca
    }

def build_pca_dataframe(X_pca, df_split, split_name):
    
    n_components = X_pca.shape[1]
    feature_columns = [f"pc_{i}" for i in range(n_components)]
    
    df_features = pd.DataFrame(X_pca, columns=feature_columns)
    
    df_meta = pd.DataFrame({
        "filename": df_split["filename"].values,
        "split": split_name,
        "style": df_split["style"].values,
        "style_name": df_split["style_name"].values
    })
    
    df_final = pd.concat([df_meta.reset_index(drop=True),
                          df_features.reset_index(drop=True)], axis=1)
    
    return df_final
