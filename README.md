# Classification de Styles Artistiques

Reconnaissance automatique de 23 mouvements artistiques par vision par ordinateur 
et deep learning, à partir d'un sous dataset de WikiArt.

## Structure du projet

├── notebooks/
│   ├── 1_RV-DL_exploration.ipynb         # Exploration et analyse du dataset
│   ├── 2_RV_Analyse_Frequentielle.ipynb  # Analyse fréquentielle multi-échelle
│   ├── 2_RV_Analyse_Vision_Classique.ipynb # Descripteurs HOG et couleur
│   ├── 3_RV_Classification_ML.ipynb      # Classification SVM + Random Forest
│   ├── 4_RV-DL_BaselineResNet.ipynb      # Baseline ResNet18
│   ├── 5_DL_Archi_ResNet.ipynb           # Expérimentations ResNet
│   ├── 5_DL_Archi_EfficientNet.ipynb     # Expérimentations EfficientNet
│   ├── 5_DL_Archi_MobileNet.ipynb        # Expérimentations MobileNet
│   ├── 6_DL_Tuning.ipynb                 # Optimisation des hyperparamètres
│   ├── 7_RV-DL_Approche_Hybride.ipynb    # Fusion vision classique + DL
│   └── 8_RV-DL_Robustesse_Evaluation.ipynb # Évaluation finale et robustesse
│
├── src/
│   ├── dataset.py                    # Chargement du dataset (partie RV)
│   ├── dataset_dl.py                 # Chargement du dataset (partie DL)
│   ├── load_dataset.py               # Utilitaires de chargement
│   ├── preprocessing.py              # Nettoyage et préparation des données
│   ├── frequency_analysis.py         # Analyse fréquentielle (Fourier, Gabor)
│   ├── frequency_visualizer.py       # Visualisations fréquentielles
│   ├── vision_features_extract.py    # Extraction HOG et couleur
│   ├── ml_utils.py                   # SVM, Random Forest, métriques et signatures stylistiques
│   ├── models.py                     # Architectures deep learning
│   ├── train.py                      # Boucle d'entraînement
│   ├── evaluate.py                   # Évaluation des modèles
│   └── utils.py                      # Fonctions utilitaires communes
│
└── results/
    └── CSV/                         
         ├── dataset_vision_features.csv   # Caractéristiques issues des techniques de vision
         └── final_frequency_features.csv  # Caractéristiques issues de l'analyse fréquentielle     

## Pipeline

1. **Exploration** (`1_RV-DL_exploration`) → analyse du dataset, 
   matrices de distance par genre
2. **Features vision** (`2_RV_Analyse_Vision_Classique`) → extraction HOG + couleur 
   via `src/vision_features_extract.py` → `dataset_vision_features.csv`
3. **Analyse fréquentielle** (`2_RV_Analyse_Frequentielle`) → Fourier, Gabor, filtres 
   via `src/frequency_analysis.py` → `final_frequency_features*.csv`
4. **Classification ML** (`3_RV_Classification_ML`) → SVM + Random Forest 
   via `src/ml_utils.py`
5. **Baseline DL** (`4_RV-DL_BaselineResNet`) → ResNet18 + ablation fréquentielle
   via `src/train.py` et `src/evaluate.py`
6. **Architectures DL** (`5_DL_Archi_*`) → comparaison ResNet / EfficientNet / MobileNet
   via `src/models.py`
7. **Tuning** (`6_DL_Tuning`) → optimisation des hyperparamètres
8. **Approche hybride** (`7_RV-DL_Approche_Hybride`) → fusion reconnaissance visuelle + DL
9. **Évaluation finale** (`8_RV-DL_Robustesse_Evaluation`) → robustesse et analyse des confusions


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


