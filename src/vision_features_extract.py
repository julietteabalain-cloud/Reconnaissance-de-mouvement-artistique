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
            img = load_image_fn(row)
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
            img = load_image_fn(row)
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
