import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FrequencyVisualizer:
    """
    Classe utilitaire pour l'affichage des analyses de styles artistiques.
    Regroupe les configurations Matplotlib/Seaborn pour l'interprétation visuelle.
    """
    
    @staticmethod
    def plot_comparative_analysis(styles, df_analysis, analyzer, dataset_module, DATA_ROOT, mode="fourier"):
        """
        Génère une grille comparative (Fourier, Multi-échelle ou Gabor).
        
        Args:
            styles (list): Liste des noms de styles à comparer.
            df_analysis (pd.DataFrame): DataFrame contenant les métadonnées des images.
            analyzer (FrequencyAnalyzer): Instance de la classe d'analyse pour le calcul à la volée.
            dataset_module: Module personnalisé pour le chargement des images.
            DATA_ROOT (str): Chemin racine vers les images.
            mode (str): Type d'analyse ('fourier', 'multiscale', 'gabor').
        """
        n_styles = len(styles)
        
        if mode == "fourier":
            # Grille 2 colonnes par style (Image + Spectre)
            fig, axes = plt.subplots((n_styles + 1) // 2, 4, figsize=(16, 4 * ((n_styles + 1) // 2)))
            title = "Analyse de Fourier : Contrasté (en haut) vs Ambigu (en bas)"
        
        elif mode == "multiscale":
            # Grille 3 colonnes (Original + Basse Freq + Haute Freq)
            fig, axes = plt.subplots(n_styles, 3, figsize=(15, 4 * n_styles))
            title = "Analyse Multi-échelle : Extraction des textures"
            
        elif mode == "gabor":
            # Grille 2 colonnes pour comparer les réponses de filtres
            fig, axes = plt.subplots((n_styles + 1) // 2, 2, figsize=(14, 5 * ((n_styles + 1) // 2)))
            title = "Analyse de Texture (Gabor) : Comparaison du 'Grain' Artistique"

        axes_flat = axes.flatten()

        for i, style in enumerate(styles):
            # Pipeline de chargement commun 
            row = df_analysis[df_analysis['style_name'] == style].iloc[0]
            img_pil = dataset_module.load_image(row, DATA_ROOT)
            img_np = analyzer.preprocess_image(img_pil)

            # Logique d'affichage spécifique
            if mode == "fourier":
                f_shift, log_spec, metrics = analyzer.get_fourier_data(img_np)
                base = i * 2
                axes_flat[base].imshow(img_np, cmap='gray')
                axes_flat[base].set_title(f"{style}\nRatio HF/BF: {metrics['hf_bf_ratio']:.4f}", fontsize=9)
                axes_flat[base+1].imshow(log_spec, cmap='magma')
                axes_flat[base+1].set_title(f"Spectre - Entropie: {metrics['spectral_entropy']:.2f}", fontsize=9)

            elif mode == "multiscale":
                f_shift, _, _ = analyzer.get_fourier_data(img_np)
                img_lp, img_hp, m_filt = analyzer.apply_multiscale_filters(f_shift)
                base = i * 3
                axes_flat[base].imshow(img_np, cmap='gray')
                axes_flat[base].set_title(f"Original ({style})")
                axes_flat[base+1].imshow(img_lp, cmap='gray')
                axes_flat[base+1].set_title("Passe-Bas (Structure)")
                axes_flat[base+2].imshow(img_hp, cmap='gray')
                axes_flat[base+2].set_title(f"Passe-Haut (Détails)\nVar HP: {m_filt['highpass_variance']:.1f}")

            elif mode == "gabor":
                responses, _ = analyzer.get_gabor_bank_features(img_np)
                target_idx = 4 # Orientation 45°, Échelle Fine
                im = axes_flat[i].imshow(responses[target_idx], cmap='magma', vmin=0)
                axes_flat[i].set_title(f"Style : {style}\n(Filtre Gabor 45°)", fontsize=11)
                plt.colorbar(im, ax=axes_flat[i], fraction=0.046, pad=0.04)

        for ax in axes_flat:
            ax.axis('off')

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show() 
        plt.close(fig) 


    @staticmethod
    def plot_gabor_radar(df, styles_to_comp):
        """
        Génère un graphique radar montrant la directionnalité des textures.
        
        Args:
            df (pd.DataFrame): DataFrame des features (df_features).
            styles_to_comp (list): Liste des styles à superposer sur le radar.
        """
        # Configuration des axes
        labels = ['Horiz. (0°)', 'Diag. (45°)', 'Vert. (90°)', 'Diag. (135°)']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1] # Fermer le cercle pour le tracé

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Tracé pour chaque style
        for style in styles_to_comp:
            subset = df[df['style_name'] == style]
            if subset.empty:
                continue
                
            # Moyenne des variances pour chaque angle
            values = [
                subset[[c for c in df.columns if 'var_th0' in c]].mean().mean(),
                subset[[c for c in df.columns if 'var_th45' in c]].mean().mean(),
                subset[[c for c in df.columns if 'var_th90' in c]].mean().mean(),
                subset[[c for c in df.columns if 'var_th135' in c]].mean().mean()
            ]
            values += values[:1] # Fermer le tracé
            
            ax.plot(angles, values, linewidth=2, label=style)
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title("Radar Chart : Directionnalité des traits par style", fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


    @staticmethod
    def plot_pca(df_pca, explained_variance):
        """
        Affiche la projection PCA des styles artistiques.
        
        Args:
            df_pca (pd.DataFrame)
            explained_variance (np.ndarray): Ratio de variance pour chaque composante.
        """
        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            data=df_pca, x='PC1', y='PC2', hue='style_name', 
            palette='tab20', s=100, alpha=0.8, edgecolor='black'
        )
        
        total_var = explained_variance.sum()
        plt.title(f"PCA : Structure latente des styles artistiques\n"
                f"(Variance expliquée : {total_var:.1f}%)", fontsize=15)
        plt.xlabel(f"PC1 ({explained_variance[0]:.1f}%) - Axe de Complexité")
        plt.ylabel(f"PC2 ({explained_variance[1]:.1f}%) - Axe de Texture")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Styles")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()