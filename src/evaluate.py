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
        encoded_to_style = dict(zip(loader.dataset.df["style_encoded"], loader.dataset.df["style_name"]))
        pred_names = [encoded_to_style[pred.item()] for pred in preds]
        label_names = [encoded_to_style[label.item()] for label in labels]
        for label, pred in zip(label_names, pred_names):
            total_per_class[class_names.index(label)] += 1
            if label == pred:
                correct_per_class[class_names.index(label)] += 1
        
    accuracy_per_class = correct_per_class / total_per_class

    return accuracy_per_class


@torch.no_grad()
def accuracy_per_class_fast(model, loader, device, class_names):
    """Version optimisée de accuracy_per_class — utilise un dict pour O(1)"""
    model.eval()

    correct_per_class = np.zeros(len(class_names))
    total_per_class   = np.zeros(len(class_names))

    encoded_to_style = dict(zip(
        loader.dataset.df["style_encoded"],
        loader.dataset.df["style_name"]
    ))
    style_to_idx = {style: idx for idx, style in enumerate(class_names)}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            style_name = encoded_to_style[label]
            idx = style_to_idx[style_name]
            total_per_class[idx] += 1
            if label == pred:
                correct_per_class[idx] += 1

    return correct_per_class / total_per_class


def visualize_accuracy_per_style(results): 
    styles = [x[0] for x in results]
    accs = [x[1] for x in results]
    plt.figure(figsize=(10, 8))
    plt.barh(styles, accs)
    plt.xlabel("Accuracy")
    plt.title("Accuracy per Style")
    plt.gca().invert_yaxis()
    plt.show()


def plot_style_accuracy_comparison(results_dict, class_names, save_path=None):
    """
    Barplot horizontal groupé — accuracy par style pour N configurations.
    
    results_dict = {
        'Original'        : [acc_style0, acc_style1, ...],
        'Haute fréquence' : [acc_style0, acc_style1, ...],
        'Basse fréquence' : [acc_style0, acc_style1, ...],
    }
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_styles = len(class_names)
    n_models = len(results_dict)
    height   = 0.8 / n_models 

    colors = ['steelblue', 'tomato', 'mediumseagreen', 'mediumpurple']
    labels = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(10, n_styles * 0.45 + 2))

    y = np.arange(n_styles)
    for i, (label, vals, color) in enumerate(zip(labels, values, colors)):
        offset = (i - n_models / 2 + 0.5) * height
        bars = ax.barh(y + offset, vals, height=height * 0.9,
                      label=label, color=color, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Accuracy (test)')
    ax.set_title('Accuracy par style artistique\nselon la représentation fréquentielle')
    ax.axvline(x=0.0, color='black', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


################## Pour architecture multibranches ############################

def evaluate_model_multibranch(model, loader, device):
    """Version multibranch de evaluate_model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_orig, x_hf, x_lf, labels in loader:
            x_orig = x_orig.to(device)
            x_hf   = x_hf.to(device)
            x_lf   = x_lf.to(device)
            labels = labels.to(device)

            outputs = model(x_orig, x_hf, x_lf)
            _, preds = torch.max(outputs, 1)

            encoded_to_style = dict(zip(
                loader.dataset.dataset_orig.df["style_encoded"],
                loader.dataset.dataset_orig.df["style_name"]
            ))
            pred_names  = [encoded_to_style[p.item()] for p in preds]
            label_names = [encoded_to_style[l.item()] for l in labels]

            all_preds.extend(pred_names)
            all_labels.extend(label_names)

    acc    = accuracy_score(all_labels, all_preds)
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, cm, report


@torch.no_grad()
def accuracy_per_class_multibranch(model, loader, device, class_names):
    """Version multibranch de accuracy_per_class"""
    model.eval()

    correct_per_class = np.zeros(len(class_names))
    total_per_class   = np.zeros(len(class_names))

    for x_orig, x_hf, x_lf, labels in loader:
        x_orig = x_orig.to(device)
        x_hf   = x_hf.to(device)
        x_lf   = x_lf.to(device)
        labels = labels.to(device)

        outputs = model(x_orig, x_hf, x_lf)
        _, preds = torch.max(outputs, 1)

        encoded_to_style = dict(zip(
            loader.dataset.dataset_orig.df["style_encoded"],
            loader.dataset.dataset_orig.df["style_name"]
        ))
        pred_names  = [encoded_to_style[p.item()] for p in preds]
        label_names = [encoded_to_style[l.item()] for l in labels]

        for label, pred in zip(label_names, pred_names):
            total_per_class[class_names.index(label)] += 1
            if label == pred:
                correct_per_class[class_names.index(label)] += 1

    return correct_per_class / total_per_class


################ Approche Hybride ######################################
# Pour la visualisation t-SNE 

def extract_embeddings(model, loader, device):
    """Extrait les embeddings avant la dernière couche FC — ResNet18"""
    model.eval()
    all_embeddings = []
    all_labels = []
    embeddings = []

    def hook_fn(module, input, output):
        embeddings.append(input[0].detach().cpu())

    # Hook sur la couche FC (dernière couche)
    hook = model.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            _ = model(images)
            all_labels.extend(labels.numpy())
            if i % 10 == 0:
                print(f"  Batch {i}/{len(loader)}")

    hook.remove()

    all_embeddings = torch.cat(embeddings, dim=0).numpy()
    all_labels = np.array(all_labels)

    return all_embeddings, all_labels

def extract_embeddings_multibranch(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    embeddings = []

    def hook_fn(module, input, output):
        embeddings.append(input[0].detach().cpu())

    hook = model.classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x_orig, x_hf, x_lf, labels in loader:
            x_orig = x_orig.to(device)
            x_hf   = x_hf.to(device)
            x_lf   = x_lf.to(device)
            _ = model(x_orig, x_hf, x_lf)
            all_labels.extend(labels.numpy())

    hook.remove()

    all_embeddings = torch.cat(embeddings, dim=0).numpy()
    all_labels = np.array(all_labels)

    return all_embeddings, all_labels

def plot_tsne(embeddings, labels, class_names, title, save_path=None):

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    # PCA d'abord :
    print(f"  PCA...")
    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    # t-SNE ensuite :
    print(f"  t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                n_iter=1000, n_jobs=-1)  # n_jobs=-1 = tous les CPU
    reduced = tsne.fit_transform(embeddings_pca)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    plt.figure(figsize=(14, 8))
    for i, style in enumerate(class_names):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                   color=colors[i], alpha=0.5, s=15)
        # Annoter le centroïde
        if mask.sum() > 0:
            cx = reduced[mask, 0].mean()
            cy = reduced[mask, 1].mean()
            plt.annotate(style[:10], (cx, cy), fontsize=6,
                        ha='center', fontweight='bold')
    
    plt.title(f"t-SNE — {title}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
