# Classification de Styles Artistiques

Reconnaissance automatique de 23 mouvements artistiques par vision par ordinateur et deep learning, à partir d'un sous-ensemble de WikiArt (9 942 images).

## Structure du projet

Les notebooks préfixés `RV` correspondent au rapport de reconnaissance visuelle (approches classiques), ceux préfixés `DL` au rapport de deep learning. Les notebooks `RV-DL` contribuent aux deux.

```
├── notebooks/
│   ├── 1_RV-DL_exploration.ipynb                    # Exploration et analyse du dataset
│   ├── 2_RV_Analyse_Vision_Classique.ipynb          # Descripteurs HOG et couleur
│   ├── 2_RV_Analyse_Frequentielle.ipynb             # Analyse fréquentielle (Fourier, Gabor)
│   ├── 3_RV_Classification_ML.ipynb                 # Classification SVM + Random Forest
│   ├── 4_RV-DL_BaselineResNet.ipynb                 # CNN from scratch + ResNet18 original, haute et basse fréquence
│   ├── 5_DL_Archi_ResNet.ipynb                      # Comparaison architectures ResNet
│   ├── 5_DL_Archi_EfficientNet.ipynb                # Comparaison architectures EfficientNet
│   ├── 5_DL_Archi_Finetuning_MobileNet.ipynb        # Finetuning progressif MobileNetV3-Large
│   └── 6_RV-DL_Approche_Hybride.ipynb               # Architecture multi-branches + barplot + t-SNE
│
├── src/
│   ├── dataset.py                       # Chargement du dataset (partie RV)
│   ├── dataset_dl.py                    # Chargement du dataset (partie DL)
│   ├── load_dataset.py                  # Script de téléchargement du sous-ensemble WikiArt (usage unique, Google Colab)
│   ├── preprocessing.py                 # Nettoyage et préparation des données
│   ├── frequency_analysis.py            # Transformations fréquentielles (Fourier, Gabor)
│   ├── frequency_visualizer.py          # Visualisations fréquentielles
│   ├── vision_features_extract.py       # Extraction HOG et couleur
│   ├── ml_utils.py                      # SVM, Random Forest, métriques et signatures stylistiques
│   ├── models.py                        # Architectures deep learning
│   ├── train.py                         # Boucle d'entraînement
│   ├── evaluate.py                      # Évaluation, t-SNE, barplot
│   └── utils.py                         # Fonctions utilitaires communes
│
└── results/
    └── CSV/
        ├── dataset_vision_features.csv      # Descripteurs HOG et couleur
        └── final_frequency_features.csv     # Descripteurs fréquentiels
```

## Pipeline

1. **Exploration** (`1_RV-DL_exploration`) — analyse du dataset, distributions et matrices de distance par style
2. **Features vision** (`2_RV_Analyse_Vision_Classique`) — extraction HOG + couleur → `dataset_vision_features.csv`
3. **Analyse fréquentielle** (`2_RV_Analyse_Frequentielle`) — Fourier, Gabor, filtres passe-haut/bas → `final_frequency_features.csv`
4. **Classification ML** (`3_RV_Classification_ML`) — SVM (31.4%) et Random Forest (30.8%) comme baseline classique
5. **Baseline DL** (`4_RV-DL_BaselineResNet`) — ResNet18 entraîné sur images originales, haute fréquence et basse fréquence, CNN from scratch
6. **Comparaison architectures** (`5_DL_Archi_*`) — ResNet, EfficientNet, MobileNet en transfer learning (backbone gelé)
7. **Finetuning** (`5_DL_Archi_Finetuning_MobileNet`) — dégel progressif de MobileNetV3-Large → **50% test** (meilleur modèle)
8. **Approche hybride** (`6_RV-DL_Approche_Hybride`) — architecture multi-branches fusionnant les trois représentations fréquentielles, barplot par style et visualisations t-SNE

## Dataset

Le dataset n’est pas versionné sur Git.

1. Télécharger `WikiArt_Subset.zip` (lien fourni séparément)
2. Décompresser dans :
   deepl-projet/data/
3. Vérifier la structure :
   - data/train
   - data/val
   - data/test
   - *_metadata.csv


mklink /D data "G:\Mon Drive\DeepLearning\WikiArt_Subset"


