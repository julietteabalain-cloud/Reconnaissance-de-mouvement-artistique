""" entrainement générique """

import torch
from tqdm import tqdm
import copy


def train_one_epoch(model, loader, criterion, optimizer, device):
    """ Entraine le modèle pour une époque """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """ Valide une époch """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    num_epochs,
                    early_stopping=None,
                    scheduler=None
                ):
    """ Entraine le modèle complet : toute les epoch """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float('inf') 
    best_model_state = None

    if early_stopping is not None:
        best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        #Ajout scheduler 
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

        
        if early_stopping is not None:
            early_stopping.step(val_loss)
            if early_stopping.stop:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Meilleur modèle restauré (val_loss={best_val_loss:.4f})")


    return history


########################## POUR LE FINETUNING ######################################

def unfreeze_last_layers(model, architecture: str, num_blocks: int = 3):
    """
    Dégèle les derniers blocs d'un modèle pré-entraîné.
    Gèle tout d'abord, puis dégèle les N derniers blocs + classifier.
    
    Args:
        model       : le modèle PyTorch
        architecture : archi utilisée
        num_blocks  : nombre de blocs à dégeler depuis la fin
    
    Returns:
        model avec les paramètres appropriés dégelés
    """
    # Geler tout
    for param in model.parameters():
        param.requires_grad = False

    # Identifier les couches features selon l'architecture
    arch = architecture.lower()

    if arch in ("resnet18", "resnet34"):
        feature_blocks = [
            model.layer1,
            model.layer2, 
            model.layer3,
            model.layer4,
        ]
        classifier_params = model.fc.parameters()

    elif arch == "mobilenet_v2":
        feature_blocks = list(model.features.children())
        classifier_params = model.classifier.parameters()

    elif arch in ("mobilenet_v3_small", "mobilenet_v3_large"):
        feature_blocks = list(model.features.children())
        classifier_params = model.classifier.parameters()

    elif arch in ("efficientnet_b0", "efficientnet_b1"):
        feature_blocks = list(model.features.children())
        classifier_params = model.classifier.parameters()

    else:
        raise ValueError(
            f"Architecture '{architecture}' non supportée. "
            f"Ajoutez-la dans unfreeze_last_layers()."
        )

    # Dégeler les N derniers blocs
    blocks_to_unfreeze = feature_blocks[-num_blocks:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True

    for param in classifier_params:
        param.requires_grad = True

    # Log de ce qui est dégelé
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    return model


def unfreeze_all(model):
    """Dégèle tous les paramètres du modèle pour finetuning complet."""
    for param in model.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Tous les paramètres dégelés : {total:,}")
    return model


########################## APPROCHE HYBRIDE #########################################
# Pour l'entrainement de notre modèle avec la fusion des 3 branches (normal, haute-fréquence, basse-fréquence)

def train_one_epoch_multibranch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x_orig, x_hf, x_lf, labels in loader:
        x_orig  = x_orig.to(device)
        x_hf    = x_hf.to(device)
        x_lf    = x_lf.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        outputs = model(x_orig, x_hf, x_lf)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch_multibranch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x_orig, x_hf, x_lf, labels in loader:
            x_orig  = x_orig.to(device)
            x_hf    = x_hf.to(device)
            x_lf    = x_lf.to(device)
            labels  = labels.to(device)

            outputs = model(x_orig, x_hf, x_lf)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_model_multibranch(model, train_loader, val_loader, criterion,
                             optimizer, device, num_epochs,
                             early_stopping=None, scheduler=None):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss   = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch_multibranch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch_multibranch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if early_stopping is not None:
            early_stopping.step(val_loss)
            if early_stopping.stop:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Meilleur modèle restauré (val_loss={best_val_loss:.4f})")

    return history