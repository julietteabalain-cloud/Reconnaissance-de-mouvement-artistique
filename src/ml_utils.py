import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


def split_data(df_all):
    """
    Sépare le dataframe fusionné en trois ensembles : Train, Val, Test.
    """
    # Masques basés sur la colonne split définie dans le projet
    train_df = df_all[df_all['split'] == 'train']
    val_df   = df_all[df_all['split'] == 'val']
    test_df  = df_all[df_all['split'] == 'test']

    # Colonnes à exclure des features pour le SVM
    drop_cols = ['filename', 'split', 'style_name']
    
    X_train, y_train = train_df.drop(columns=drop_cols), train_df['style_name']
    X_val, y_val     = val_df.drop(columns=drop_cols), val_df['style_name']
    X_test, y_test   = test_df.drop(columns=drop_cols), test_df['style_name']
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preparer_donnees_combinees(df_vision, path_freq):
    """
    Prépare les données pour un GridSearchCV avec PredefinedSplit.
    
    Fusionne, splitte, standardise et combine les sets Train et Val pour 
    permettre une optimisation rigoureuse des hyperparamètres.

    Args:
        df_vision (pd.DataFrame): Features de vision (PCA).
        path_freq (str): Chemin vers le CSV des fréquences.

    Returns:
        tuple: (X_combined_s, y_combined, test_fold, X_test_s, y_test)
            - X_combined_s: Train + Val standardisés.
            - y_combined: Labels Train + Val.
            - test_fold: Indices indiquant à Python où s'arrête le Train et où commence le Val.
            - X_test_s: Set de test final standardisé.
            - y_test: Labels de test.
    """
    # 1. Fusion et Split classique
    df_freq = pd.read_csv(path_freq)
    df_all = pd.merge(df_vision, df_freq, on=["filename", "split", "style_name"])
    
    from src.ml_utils import split_data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_all)
    
    # 2. Standardisation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # 3. Création du set combiné (Train + Val)
    X_combined = np.vstack((X_train_s, X_val_s))
    y_combined = np.concatenate((y_train, y_val))
    
    # 4. Création du PredefinedSplit index
    # -1 pour les données de Train, 0 pour les données de Val
    test_fold = np.concatenate([
        -1 * np.ones(X_train_s.shape[0]), 
         0 * np.ones(X_val_s.shape[0])
    ])
    
    return X_combined, y_combined, test_fold, X_test_s, y_test


def plot_art_confusion_matrix(y_true, y_pred):
    """    
    Affiche une matrice de confusion pour analyser les erreurs entre styles.
    """
    labels = sorted(list(set(y_true)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(18, 14)) 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Nombre de tableaux'})
    
    plt.title('Matrice de Confusion : Styles Artistiques', fontsize=16)
    plt.ylabel('Style Réel', fontsize=12)
    plt.xlabel('Style Prédit', fontsize=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()