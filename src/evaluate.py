""" Fonctions utiles pour l'évaluation des modèles """

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

########## Evaluation classique : loss et accuracy sur le validation set

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            encoded_to_style = dict(zip(loader.dataset.df["style_encoded"], loader.dataset.df["style_name"]))
            pred_names = [encoded_to_style[pred.item()] for pred in preds]
            label_names = [encoded_to_style[label.item()] for label in labels]

            all_preds.extend(pred_names)
            all_labels.extend(label_names)
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, cm, report


#######" Fonctions pour la matrice de confusion et l'accuracy par style"

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
        encoded_to_style = dict(zip(loader.dataset.df["style_encoded"], loader.dataset.df["style_name"]))
        pred_names = [encoded_to_style[pred.item()] for pred in preds]
        label_names = [encoded_to_style[label.item()] for label in labels]
        all_preds.extend(pred_names)
        all_labels.extend(label_names)

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
