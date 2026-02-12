""" Fonctions utiles pour l'évaluation des modèles """

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


@torch.no_grad()
def compute_confusion_matrix(model, loader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    return cm


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# accuracy par style
@torch.no_grad()
def accuracy_per_class(model, loader, device, class_names):
    model.eval()

    num_classes = len(class_names)

    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for label, pred in zip(labels, preds):
            total_per_class[label.item()] += 1
            if label == pred:
                correct_per_class[label.item()] += 1

    accuracy_per_class = correct_per_class / total_per_class

    return accuracy_per_class

def visualize_accuracy_per_style(results): 
    styles = [x[0] for x in results]
    accs = [x[1] for x in results]

    plt.figure(figsize=(10, 8))
    plt.barh(styles, accs)
    plt.xlabel("Accuracy")
    plt.title("Accuracy per Style")
    plt.gca().invert_yaxis()
    plt.show()
