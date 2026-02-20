import numpy as np
import cv2
from PIL import Image
from scipy.stats import entropy

class FrequencyAnalyzer:
    """
    Analyseur de caractéristiques fréquentielles et texturales pour les styles artistiques.
    
    Cette classe centralise les algorithmes de traitement du signal (Fourier, Filtres passe-haut/passe-bas, Gabor) 
    pour transformer une image en un vecteur de descripteurs statistiques utilisables 
    par un modèle de Machine Learning.
    """

    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def preprocess_image(self, img_pil):
        """
        Prépare l'image pour l'analyse (Conversion gris et redimensionnement).
        Entrée : Image PIL (RGB)
        Sortie : Array Numpy (Gris, target_size)
        """
        img_gray = img_pil.convert('L')
        img_resized = img_gray.resize(self.target_size, resample=Image.LANCZOS)
        return np.array(img_resized)

    # PARTIE A : FOURIER 2D
    
    def get_fourier_data(self, img_np):
        """
        Calcule la FFT 2D et extrait les métriques spectrales globales.
        Entrée : Array Numpy (Gris)
        Sortie : 
            - f_shift : Spectre complexe centré
            - log_spectrum : Spectre de puissance (échelle log pour visualisation)
            - metrics : Dictionnaire (hf_bf_ratio, energy_center, energy_periphery, spectral_entropy)
        """
        # FFT 2D et décalage du centre
        f_transform = np.fft.fft2(np.float32(img_np) / 255.0)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log-spectrum pour visualisation
        log_spectrum = 20 * np.log(magnitude + 1)
        
        # Masques fréquentiels (Rayon de 10% pour les basses fréquences)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask_center = dist <= (min(h, w) * 0.1)
        
        # Calcul des énergies
        e_center = np.sum(magnitude[mask_center])
        e_periphery = np.sum(magnitude[~mask_center])
        
        # Entropie spectrale
        pk = magnitude.flatten() / (np.sum(magnitude) + 1e-6)
        spec_entropy = entropy(pk)
        
        metrics = {
            "hf_bf_ratio": e_periphery / (e_center + 1e-6),
            "energy_center": e_center,
            "energy_periphery": e_periphery,
            "spectral_entropy": spec_entropy
        }
        return f_shift, log_spectrum, metrics

    # PARTIE B : FILTRES PASSE-HAUT / PASSE-BAS

    def apply_multiscale_filters(self, f_shift, radius_pct=0.1):
        """
        Sépare l'image en structures globales (LP) et détails fins (HP).
        Entrée : f_shift (Spectre complexe centré)
        Sortie :
            - img_lp : Image filtrée passe-bas
            - img_hp : Image filtrée passe-haut
            - metrics : Dictionnaire (energies et variances pour LP et HP)
        """
        h, w = f_shift.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - (w//2))**2 + (y - (h//2))**2)
        radius = min(h, w) * radius_pct
        
        # Application des filtres dans le domaine fréquentiel
        f_lp = f_shift * (dist <= radius)
        f_hp = f_shift * (dist > radius)
        
        # Retour au domaine spatial
        img_lp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_lp)))
        img_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_hp)))
        
        metrics = {
            "lowpass_energy": np.sum(img_lp**2),
            "highpass_energy": np.sum(img_hp**2),
            "lowpass_variance": np.var(img_lp),
            "highpass_variance": np.var(img_hp)
        }
        return img_lp, img_hp, metrics

    # PARTIE C : BANC DE FILTRES DE GABOR

    def get_gabor_bank_features(self, img_np, orientations=4, scales=3):
        """
        Analyse les micro-textures (coups de pinceau) via un banc de filtres.
        Entrée : Array Numpy (Gris)
        Sortie :
            - gabor_responses : Liste des images résultantes (pour visualisation 4x3)
            - metrics : Dictionnaire de 36 features (mean, var, entropy par filtre)
        """
        gabor_responses = []
        metrics = {}
        
        # Paramètres du banc : 4 orientations (0, 45, 90, 135) et 3 fréquences (échelles)
        thetas = np.linspace(0, np.pi, orientations, endpoint=False)
        frequencies = [0.1, 0.25, 0.4] 
        
        for i, theta in enumerate(thetas):
            for j, freq in enumerate(frequencies):
                # Création du noyau de Gabor
                kernel = cv2.getGaborKernel((21, 21), sigma=3.0, theta=theta, 
                                            lambd=1.0/freq, gamma=0.5, psi=0, ktype=cv2.CV_32F)
                
                # Filtrage
                filtered = cv2.filter2D(img_np, cv2.CV_32F, kernel)
                gabor_responses.append(filtered)
                
                # Extraction des statistiques
                suffix = f"th{int(np.degrees(theta))}_f{j}"
                metrics[f"gabor_mean_{suffix}"] = np.mean(filtered)
                metrics[f"gabor_var_{suffix}"] = np.var(filtered)
                
                # Entropie de la réponse (mesure de régularité texturelle)
                hist, _ = np.histogram(filtered, bins=32, density=True)
                metrics[f"gabor_entropy_{suffix}"] = entropy(hist + 1e-6)
                
        return gabor_responses, metrics
    

    def extract_features_from_image(self, img_np):
        """
        Regroupe toutes les extractions mathématiques pour une image.
        Args:
            img_np (np.ndarray): Image en niveaux de gris pré-traitée (Array Numpy).

        Returns:
            dict: Un dictionnaire fusionné contenant l'ensemble des descripteurs :
                - Métriques de Fourier
                - Métriques multi-échelles
                - Métriques de Gabor 
        """
        # 1. Fourier
        f_shift, _, f_metrics = self.get_fourier_data(img_np)
        
        # 2. Multi-échelles (en utilisant f_shift déjà calculé)
        _, _, m_metrics = self.apply_multiscale_filters(f_shift)
        
        # 3. Gabor
        _, g_metrics = self.get_gabor_bank_features(img_np)
        
        # Fusion des 3 dictionnaires
        return {**f_metrics, **m_metrics, **g_metrics}


########## Transformation des images ###########

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import cv2
import numpy as np

class LowFrequencyTransform:
    def __init__(self, kernel_size=9, sigma=2):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        # img est un PIL Image
        img_np = np.array(img)

        low = cv2.GaussianBlur(
            img_np,
            (self.kernel_size, self.kernel_size),
            self.sigma
        )

        return F.to_pil_image(low)



class HighFrequencyTransform:
    def __init__(self, kernel_size=9, sigma=2):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img)

        low = cv2.GaussianBlur(
            img_np,
            (self.kernel_size, self.kernel_size),
            self.sigma
        )

        high = img_np.astype(np.float32) - low.astype(np.float32)

        # Recentre pour rester dans 0-255
        high = high + 127
        high = np.clip(high, 0, 255).astype(np.uint8)

        return F.to_pil_image(high)
