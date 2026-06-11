"""
MODULE_A: AR Tabanlı Hacim ve Kütle Tahmini

ARCore/ARKit'ten alınan boyut ölçümleri kullanılarak
karpuzun hacmi (elipsoid modeli) ve kütlesi hesaplanır.

Formüller:
  - Hacim: V = (4/3) * π * a * b * c  (elipsoid)
  - Kütle: m = V * ρ  (ρ ≈ 0.98 g/cm³)
  
Bu değerler Elasticity Index (EI) hesabında kullanılır.
"""

import numpy as np
from typing import Tuple, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import WATERMELON_DENSITY, PI


class ARVolumeEstimator:
    """
    AR ile ölçülen boyutlardan hacim ve kütle tahmini.
    
    Karpuz elipsoid olarak modellenir.
    ARCore/ARKit width, height, depth değerlerini kullanır.
    """

    def __init__(self, density: float = WATERMELON_DENSITY):
        """
        Args:
            density: Karpuz yoğunluğu (g/cm³), varsayılan 0.98
        """
        self.density = density

    def calculate_ellipsoid_volume(
        self,
        width_cm: float,
        height_cm: float,
        depth_cm: float
    ) -> float:
        """
        Elipsoid hacmi hesaplar.
        
        V = (4/3) * π * a * b * c
        
        Args:
            width_cm: Genişlik (cm) - 2a
            height_cm: Yükseklik (cm) - 2b
            depth_cm: Derinlik (cm) - 2c
            
        Returns:
            Hacim (cm³)
        """
        a = width_cm / 2.0
        b = height_cm / 2.0
        c = depth_cm / 2.0

        volume = (4.0 / 3.0) * PI * a * b * c
        return volume

    def estimate_mass(
        self,
        width_cm: float,
        height_cm: float,
        depth_cm: float
    ) -> float:
        """
        Karpuzun kütlesini tahmin eder.
        
        m = V * ρ
        
        Args:
            width_cm: Genişlik (cm)
            height_cm: Yükseklik (cm)
            depth_cm: Derinlik (cm)
            
        Returns:
            Tahmini kütle (gram)
        """
        volume = self.calculate_ellipsoid_volume(width_cm, height_cm, depth_cm)
        mass = volume * self.density
        return mass

    def estimate_from_2d(
        self,
        width_cm: float,
        length_cm: float,
        aspect_ratio: float = 0.85
    ) -> Dict[str, float]:
        """
        2D ölçümlerden (width, length) 3D hacim tahmini.
        
        Derinlik bilgisi yoksa aspect_ratio ile tahmin edilir.
        Karpuzlar genellikle hafif yassı elipsoid şeklindedir.
        
        Args:
            width_cm: Genişlik/çap (cm)
            length_cm: Uzunluk (cm)
            aspect_ratio: Derinlik/genişlik oranı (varsayılan 0.85)
            
        Returns:
            Hacim ve kütle tahminleri
        """
        depth_cm = width_cm * aspect_ratio

        volume = self.calculate_ellipsoid_volume(width_cm, length_cm, depth_cm)
        mass = volume * self.density
        mass_kg = mass / 1000.0

        return {
            "width_cm": width_cm,
            "length_cm": length_cm,
            "depth_cm_estimated": depth_cm,
            "volume_cm3": volume,
            "mass_g": mass,
            "mass_kg": mass_kg
        }

    def mass_to_kg(self, mass_g: float) -> float:
        """Gramı kilogramadönüştürür."""
        return mass_g / 1000.0

    @staticmethod
    def validate_dimensions(
        width_cm: float,
        length_cm: float,
        depth_cm: float = None
    ) -> bool:
        """
        Boyut değerlerinin makul karpuz boyutlarında olduğunu kontrol eder.
        
        Tipik karpuz boyutları: 15-50 cm
        
        Returns:
            True eğer boyutlar makul ise
        """
        min_dim = 10.0  # cm
        max_dim = 60.0  # cm

        dims = [width_cm, length_cm]
        if depth_cm is not None:
            dims.append(depth_cm)

        return all(min_dim <= d <= max_dim for d in dims)

