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


def analyser_importance_familles(model, feature_names):
    """
    Calcule l'importance cumulée et moyenne par famille de descripteurs.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): 
            Le modèle de classification entraîné disposant de l'attribut `feature_importances_`.
        feature_names (list of str): 
            La liste ordonnée des noms des caractéristiques correspondant aux colonnes 
            utilisées lors de l'entraînement du modèle.

    Returns:
        pd.DataFrame: Un tableau récapitulatif indexé par famille de descripteurs contenant :
            - 'sum' : L'importance totale de la famille (poids brut dans la décision).
            - 'mean' : L'importance moyenne par caractéristique (efficacité relative).
            - 'count' : Le nombre de caractéristiques appartenant à cette famille.
            Le DataFrame est trié par importance cumulée ('sum') décroissante.
    """

    importances = model.feature_importances_
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    
    def categoriser(name):
        name_lower = name.lower()
        if 'gabor' in name_lower: return 'Texture (Gabor)'
        if 'hog' in name_lower: return 'Structure (HOG)'
        if 'hsv' in name_lower: return 'Couleur (HSV)'
        if 'lab' in name_lower: return 'Couleur (LAB)'
        if any(x in name_lower for x in ['hf_bf', 'energy', 'spectral', 'pass_']): 
            return 'Fréquentiel (Fourier)'
        return 'Vision (PCA)'

    df_imp['famille'] = df_imp['feature'].apply(categoriser)
    
    # On agrège par somme et par moyenne, et on compte le nombre de features
    stats = df_imp.groupby('famille')['importance'].agg(['sum', 'mean', 'count'])
    stats = stats.sort_values('sum', ascending=False)
    
    return stats


def extraire_signatures_styles(X_scaled, y, feature_names):
    """
    Calcule la signature visuelle moyenne de chaque style artistique.

    Args:
        X_scaled (np.array): Features standardisées (Train + Val).
        y (np.array): Labels des styles correspondants.
        feature_names (list): Liste des noms des caractéristiques.

    Returns:
        pd.DataFrame: Moyennes des caractéristiques par style.
    """
    import pandas as pd
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df['style'] = y
    # Calcul de la moyenne par style
    signatures = df.groupby('style').mean()
    return signatures


def generer_tableau_synthese_evolue(signatures_df):
    """
    Génère un tableau de synthèse avec signatures dynamiques.
    Note : Pour les colonnes PCA (_pc), l'amplitude (distance à 0) est 
    privilégiée car elle indique la force du signal stylistique.

    Args:
        signatures_df (pd.DataFrame): DataFrame indexé par style contenant les moyennes 
            standardisées (Z-scores) de chaque caractéristique (feature).

    Returns:
        pd.DataFrame: Tableau de synthèse indexé par style avec les colonnes :
            - 'Trait' : Caractérisation de la touche picturale (Gabor/Highpass).
            - 'Contour' : Caractérisation de la netteté des formes (HOG).
            - 'Palette' : Diversité et richesse chromatique (LAB/HSV).
            - 'Structure' : Type de composition (Centrée vs Diffuse).
            - 'Signature' : La caractéristique la plus discriminante du style.
    """
    import pandas as pd
    import numpy as np

    scores = pd.DataFrame(index=signatures_df.index)
    scores['t_val'] = signatures_df[[c for c in signatures_df.columns if 'gabor_entropy' in c or 'highpass' in c]].mean(axis=1)
    scores['c_val'] = signatures_df[[c for c in signatures_df.columns if 'hog' in c.lower()]].mean(axis=1)
    scores['p_val'] = signatures_df[[c for c in signatures_df.columns if 'hsv' in c.lower() or 'lab' in c.lower()]].abs().mean(axis=1)
    scores['s_val'] = signatures_df.get('energy_center', 0) - signatures_df.get('energy_periphery', 0)

    synthese = []
    for style in signatures_df.index:
        s = scores.loc[style]
        
        # Attribution des étiquettes
        trait = f"Vibrant ({s.t_val:+.2f})" if s.t_val > scores.t_val.quantile(0.75) else f"Lissé ({s.t_val:+.2f})" if s.t_val < scores.t_val.quantile(0.25) else f"Équilibré ({s.t_val:+.2f})"
        contour = f"Précis ({s.c_val:+.2f})" if s.c_val > scores.c_val.quantile(0.7) else f"Fondu ({s.c_val:+.2f})"
        palette = f"Riche ({s.p_val:.2f})" if s.p_val > scores.p_val.median() else f"Sobre ({s.p_val:.2f})"
        structure = f"Centrée ({s.s_val:+.2f})" if s.s_val > 0.1 else f"Diffuse ({s.s_val:+.2f})"

        # Signature : On identifie la feature la plus discriminante
        row_raw = signatures_df.loc[style]
        dom_feat = row_raw.abs().idxmax()
        dom_val = row_raw[dom_feat]
        
        # Précision sur la PCA
        prefix = dom_feat.split('_')[0].upper()
        if 'pc' in dom_feat:
            signature = f"{prefix} (Impact {abs(dom_val):.2f})"
        else:
            signature = f"{prefix} ({dom_val:+.2f})"

        synthese.append({
            'Style': style.replace('_', ' '),
            'Trait': trait,
            'Contour': contour,
            'Palette': palette,
            'Structure': structure,
            'Signature': signature
        })

    return pd.DataFrame(synthese).set_index('Style')