"""
MODULE_E: Elasticity Index (EI) Hesaplayıcı

Karpuzun mekanik sertliğini gösteren Elasticity Index formülü:

    EI = f2² × m^(2/3)

Burada:
  - f2: Dominant rezonans frekansı (Hz)
  - m: Karpuzun kütlesi (kg)

Fiziksel Doğrulama:
  - f2 < 150Hz ve magnitude > 25dB → olgun karpuz
  - Düşük EI → yumuşak (olgun veya aşırı olgun)
  - Yüksek EI → sert (olgunlaşmamış)

Hollow Heart Entegrasyonu:
  - HollowHeartDetector ile çoklu akustik gösterge analizi
  - classify_by_ei() artık ses sinyalini de alabilir
  - Hollow heart skoru sınıflandırma kararına dahil edilir
"""

import numpy as np
from typing import Dict, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    F2_RIPE_THRESHOLD,
    MAGNITUDE_RIPE_THRESHOLD,
    CLASS_LABELS,
    HH_DETECTION_THRESHOLD,
    HH_HIGH_CONFIDENCE_THRESHOLD,
    AUDIO_SAMPLE_RATE
)
from backend.module_e.hollow_heart_detector import HollowHeartDetector


class ElasticityIndexCalculator:
    """
    Elasticity Index (EI) hesaplayıcı ve fiziksel doğrulayıcı.

    HollowHeartDetector entegrasyonu ile hollow heart tespiti güçlendirildi.
    """

    def __init__(
        self,
        f2_threshold: float = F2_RIPE_THRESHOLD,
        magnitude_threshold: float = MAGNITUDE_RIPE_THRESHOLD,
        sample_rate: int = AUDIO_SAMPLE_RATE
    ):
        self.f2_threshold = f2_threshold
        self.magnitude_threshold = magnitude_threshold
        self.hh_detector = HollowHeartDetector(sample_rate=sample_rate)

    def calculate_ei(self, f2: float, mass_kg: float) -> float:
        """
        Elasticity Index hesaplar.
        
        EI = f2² × m^(2/3)
        
        Args:
            f2: Dominant rezonans frekansı (Hz)
            mass_kg: Kütle (kg)
            
        Returns:
            Elasticity Index değeri
        """
        if mass_kg <= 0:
            raise ValueError(f"Kütle pozitif olmalı, alınan: {mass_kg}")
        if f2 <= 0:
            return 0.0

        ei = (f2 ** 2) * (mass_kg ** (2.0 / 3.0))
        return ei

    def validate_physical_constraints(
        self,
        f2: float,
        magnitude_db: float
    ) -> Dict[str, object]:
        """
        Fiziksel kısıtlamaları doğrular.
        
        Bilimsel referans:
        - f2 < 150Hz → olgun karpuz göstergesi
        - magnitude > 25dB → yeterli sinyal kalitesi
        
        Args:
            f2: Dominant frekans (Hz)
            magnitude_db: Genlik (dB)
            
        Returns:
            Doğrulama sonuçları
        """
        f2_valid = f2 < self.f2_threshold
        magnitude_valid = magnitude_db > self.magnitude_threshold

        return {
            "f2_value": f2,
            "f2_threshold": self.f2_threshold,
            "f2_indicates_ripe": f2_valid,
            "magnitude_db": magnitude_db,
            "magnitude_threshold": self.magnitude_threshold,
            "magnitude_sufficient": magnitude_valid,
            "physical_validation_passed": f2_valid and magnitude_valid
        }

    def classify_by_ei(
        self,
        ei: float,
        f2: float,
        zcr: float,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        magnitude_db: Optional[float] = None
    ) -> Tuple[int, str, float]:
        """
        EI, f2, ZCR ve (opsiyonel) akustik sinyal ile sınıflandırma.

        Karar kuralları (kural tabanlı + HollowHeartDetector):
        - Yüksek f2 (>150Hz) + Yüksek EI → Olgunlaşmamış (0)
        - Düşük f2 (<150Hz) + Orta EI → Olgun (1)
        - HH dedektör skoru > eşik → İçi Geçmiş (2)
        - Eski kural: Çok düşük f2 + Düşük ZCR → İçi Geçmiş (2)

        Ses sinyali sağlandığında HollowHeartDetector 5 bağımsız
        göstergeyi (dual-peak, sönüm, spektral, cepstral, HNR)
        analiz eder ve sonuç çok daha güvenilir olur.

        Args:
            ei: Elasticity Index
            f2: Dominant frekans (Hz)
            zcr: Zero Crossing Rate
            audio: Ses sinyali (opsiyonel) - sağlanırsa HH dedektör kullanılır
            sr: Örnekleme hızı (opsiyonel)
            magnitude_db: f2 genliği dB (opsiyonel)

        Returns:
            (class_id, class_label, confidence): Sınıf, etiket ve güven skoru
        """
        # --- 1) HollowHeartDetector ile güçlü tespit (ses varsa) ---
        hh_result = None
        if audio is not None and len(audio) >= 128:
            hh_result = self.hh_detector.detect(
                audio, sr=sr, f2=f2,
                f2_magnitude_db=magnitude_db, ei=ei,
                verbose=False
            )
            if hh_result["is_hollow"]:
                hh_confidence = hh_result["confidence"]
                # Yüksek güvenli hollow heart
                if hh_result["is_high_confidence"]:
                    return 2, CLASS_LABELS[2], min(1.0, hh_confidence)
                # Orta güvenli: ek kural ile doğrula
                if hh_result["active_indicators"] >= 3:
                    return 2, CLASS_LABELS[2], min(0.95, hh_confidence)
                # Tek-çift gösterge: kural tabanlı teyitle birleştir
                if f2 < self.f2_threshold * 0.75:
                    return 2, CLASS_LABELS[2], min(0.85, hh_confidence)

        # --- 2) Klasik kural tabanlı hollow heart (geriye uyumlu) ---
        if f2 < self.f2_threshold * 0.6 and zcr < 0.05:
            confidence = 0.85
            # HH dedektör sonucu varsa güveni ayarla
            if hh_result is not None:
                hh_score = hh_result["hh_score"]
                # HH skoru düşükse güveni düşür (false positive azalt)
                if hh_score < 0.3:
                    confidence = 0.6
                elif hh_score > 0.4:
                    confidence = min(0.95, 0.85 + hh_score * 0.1)
            return 2, CLASS_LABELS[2], confidence

        # --- 3) Olgun ---
        if f2 < self.f2_threshold:
            confidence = min(1.0, (self.f2_threshold - f2) / self.f2_threshold + 0.5)
            # HH skoru sınırda ise güveni düşür
            if hh_result is not None and hh_result["hh_score"] > 0.4:
                confidence = max(0.5, confidence - 0.1)
            return 1, CLASS_LABELS[1], confidence

        # --- 4) Olgunlaşmamış ---
        confidence = min(1.0, (f2 - self.f2_threshold) / self.f2_threshold + 0.5)
        return 0, CLASS_LABELS[0], confidence

    def detect_hollow_heart(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        f2: Optional[float] = None,
        f2_magnitude_db: Optional[float] = None,
        ei: Optional[float] = None,
        verbose: bool = True
    ) -> Dict[str, object]:
        """
        Hollow heart (iç boşluk) tespiti yapar.

        HollowHeartDetector'a yetkilendirilir. 5 bağımsız akustik
        göstergeyi analiz eder.

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı
            f2: Dominant frekans (Hz) - opsiyonel
            f2_magnitude_db: f2 genliği (dB) - opsiyonel
            ei: Elasticity Index - opsiyonel
            verbose: Detaylı çıktı

        Returns:
            HollowHeartDetector sonuçları
        """
        return self.hh_detector.detect(
            audio, sr=sr, f2=f2,
            f2_magnitude_db=f2_magnitude_db, ei=ei,
            verbose=verbose
        )

    def get_hollow_heart_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Hollow heart özellik vektörü üretir (fusion modeli için).

        Returns:
            8 boyutlu HH özellik vektörü
        """
        return self.hh_detector.extract_hollow_heart_features(audio, sr)

    def compute_stiffness_index(
        self,
        f2: float,
        magnitude_db: float,
        mass_kg: float
    ) -> Dict[str, float]:
        """
        Akustik sertlik indeksi hesaplar (Late Fusion için).
        
        Returns:
            Sertlik metrikleri
        """
        ei = self.calculate_ei(f2, mass_kg)

        # Normalize edilmiş sertlik skoru (0-1 arası, 0=yumuşak, 1=sert)
        # f2 normalleştirme: 50-300Hz aralığını 0-1'e map
        f2_norm = np.clip((f2 - 50) / (300 - 50), 0, 1)

        # EI normalleştirme (log ölçek)
        ei_norm = np.clip(np.log10(ei + 1) / 6, 0, 1)

        stiffness_score = 0.6 * f2_norm + 0.4 * ei_norm

        return {
            "elasticity_index": ei,
            "f2_normalized": f2_norm,
            "ei_normalized": ei_norm,
            "stiffness_score": stiffness_score,
            "magnitude_db": magnitude_db
        }

