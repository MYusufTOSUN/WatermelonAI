"""
MODULE_E: Late Fusion Stratejisi

Görsel olgunluk skorlarını akustik sertlik indeksleri ile
geç birleştirme (Late Fusion) stratejisi ile kombine eder.

Ağırlıklar:
  - Görsel (Visual): 0.35
  - Akustik (Acoustic): 0.45
  - Haptik (Haptic): 0.20

Sınıflandırma:
  0: Olgunlaşmamış (Immature)
  1: Olgun (Mature/Ripe)
  2: İçi Geçmiş (Over-mature/Hollow Heart)

Hollow Heart Entegrasyonu:
  - HollowHeartDetector sonuçları Late Fusion'a ek girdi olarak eklendi
  - Hollow heart skoru > eşik ise class 2 olasılığı artırılır
  - fuse_probabilities() ve fuse_scores() metodlarına hh_score parametresi eklendi
"""

import numpy as np
from typing import Dict, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    FUSION_VISUAL_WEIGHT,
    FUSION_ACOUSTIC_WEIGHT,
    FUSION_HAPTIC_WEIGHT,
    CLASS_LABELS,
    HH_DETECTION_THRESHOLD,
    HH_HIGH_CONFIDENCE_THRESHOLD
)


class LateFusionEngine:
    """
    Late Fusion birleştirme motoru.
    
    Her modaliteden gelen bağımsız tahminleri ağırlıklı
    olarak birleştirerek nihai karar verir.
    """

    def __init__(
        self,
        visual_weight: float = FUSION_VISUAL_WEIGHT,
        acoustic_weight: float = FUSION_ACOUSTIC_WEIGHT,
        haptic_weight: float = FUSION_HAPTIC_WEIGHT
    ):
        # Ağırlıkları normalize et
        total = visual_weight + acoustic_weight + haptic_weight
        self.visual_weight = visual_weight / total
        self.acoustic_weight = acoustic_weight / total
        self.haptic_weight = haptic_weight / total

    def fuse_probabilities(
        self,
        visual_probs: np.ndarray,
        acoustic_probs: np.ndarray,
        haptic_probs: Optional[np.ndarray] = None,
        hh_score: Optional[float] = None
    ) -> np.ndarray:
        """
        Olasılık vektörlerini ağırlıklı birleştirir.

        Hollow heart skoru sağlandığında class 2 (İçi Geçmiş)
        olasılığı boost edilir.

        Args:
            visual_probs: Görsel model olasılıkları (3,)
            acoustic_probs: Akustik model olasılıkları (3,)
            haptic_probs: Haptik model olasılıkları (3,) - opsiyonel
            hh_score: HollowHeartDetector skoru (0-1) - opsiyonel

        Returns:
            Birleştirilmiş olasılık vektörü (3,)
        """
        if haptic_probs is None:
            # Haptik veri yoksa diğer iki modaliteyi yeniden ağırlıklandır
            total = self.visual_weight + self.acoustic_weight
            fused = (
                (self.visual_weight / total) * visual_probs +
                (self.acoustic_weight / total) * acoustic_probs
            )
        else:
            fused = (
                self.visual_weight * visual_probs +
                self.acoustic_weight * acoustic_probs +
                self.haptic_weight * haptic_probs
            )

        # Hollow heart boost: HH skoru eşik üzerindeyse class 2 olasılığını artır
        if hh_score is not None and hh_score > HH_DETECTION_THRESHOLD:
            boost_strength = (hh_score - HH_DETECTION_THRESHOLD) / (1.0 - HH_DETECTION_THRESHOLD + 1e-10)
            # Yüksek güvenli HH: daha güçlü boost
            if hh_score >= HH_HIGH_CONFIDENCE_THRESHOLD:
                boost = 0.3 * boost_strength
            else:
                boost = 0.15 * boost_strength
            # Class 2 olasılığını artır, diğerlerini azalt
            fused[2] += boost
            reduction = boost / 2.0
            fused[0] = max(0.01, fused[0] - reduction)
            fused[1] = max(0.01, fused[1] - reduction)

        # Normalize et (toplam = 1.0)
        fused = fused / (np.sum(fused) + 1e-10)
        return fused

    def fuse_scores(
        self,
        visual_score: float,
        acoustic_stiffness: float,
        haptic_score: Optional[float] = None,
        hh_score: Optional[float] = None
    ) -> Dict[str, object]:
        """
        Skor tabanlı birleştirme.

        Skorlar [0, 1] arasında:
        - Düşük skor → olgunlaşmamış
        - Orta skor → olgun
        - Yüksek skor → içi geçmiş

        Hollow heart skoru sağlandığında class 2 kararı
        güçlendirilir.

        Args:
            visual_score: Görsel olgunluk skoru (0-1)
            acoustic_stiffness: Akustik sertlik skoru (0-1, ters: 0=yumuşak=olgun)
            haptic_score: Haptik yanıt skoru (0-1) - opsiyonel
            hh_score: HollowHeartDetector skoru (0-1) - opsiyonel

        Returns:
            Birleştirme sonuçları
        """
        # Akustik sertliği olgunluk skoruna çevir (ters ilişki)
        acoustic_ripeness = 1.0 - acoustic_stiffness

        if haptic_score is None:
            total = self.visual_weight + self.acoustic_weight
            fused_score = (
                (self.visual_weight / total) * visual_score +
                (self.acoustic_weight / total) * acoustic_ripeness
            )
        else:
            fused_score = (
                self.visual_weight * visual_score +
                self.acoustic_weight * acoustic_ripeness +
                self.haptic_weight * haptic_score
            )

        # Hollow heart boost: HH skoru fused_score'u yükseltir (class 2'ye iter)
        hh_override = False
        if hh_score is not None and hh_score > HH_DETECTION_THRESHOLD:
            boost = (hh_score - HH_DETECTION_THRESHOLD) * 0.5
            fused_score = min(1.0, fused_score + boost)
            # Yüksek güvenli HH: doğrudan class 2 ata
            if hh_score >= HH_HIGH_CONFIDENCE_THRESHOLD:
                fused_score = max(fused_score, 0.80)
                hh_override = True

        # Sınıflandırma
        if fused_score < 0.35:
            class_id = 0  # Olgunlaşmamış
        elif fused_score < 0.75:
            class_id = 1  # Olgun
        else:
            class_id = 2  # İçi geçmiş

        # Güven skoru: Sınıf merkezine yakınlık
        class_centers = {0: 0.15, 1: 0.55, 2: 0.88}
        distance_to_center = abs(fused_score - class_centers[class_id])
        confidence = max(0.5, 1.0 - distance_to_center * 2)

        # HH override güveni artır
        if hh_override and class_id == 2:
            confidence = max(confidence, 0.85)

        result = {
            "fused_score": fused_score,
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "confidence": confidence,
            "component_scores": {
                "visual": visual_score,
                "acoustic_ripeness": acoustic_ripeness,
                "acoustic_stiffness": acoustic_stiffness,
                "haptic": haptic_score,
                "hollow_heart": hh_score,
            },
            "weights": {
                "visual": self.visual_weight,
                "acoustic": self.acoustic_weight,
                "haptic": self.haptic_weight
            },
            "hh_override": hh_override,
        }

        return result

    def compute_dynamic_weights(
        self,
        signal_quality: Optional['SignalQuality'] = None,
        xigua_weight_base: float = 0.20,
        min_weight: float = 0.05,
    ) -> Dict[str, float]:
        """
        Sinyal kalitesine göre dinamik ağırlıklar hesapla.

        w_i(t) = max(min_weight, w_i_base * q_i) / Σ(w_j_base * q_j)

        Args:
            signal_quality: SignalQuality nesnesi (visual, acoustic, haptic [0-1])
            xigua_weight_base: Xigua audio+image erken füzyon baz ağırlığı
            min_weight: Minimum ağırlık (herhangi bir modalite tamamen göz ardı edilmez)

        Returns:
            {'visual': w_v, 'acoustic': w_a, 'haptic': w_h, 'xigua': w_x}
        """
        if signal_quality is None:
            # Statik ağırlıklar
            total = self.visual_weight + self.acoustic_weight + self.haptic_weight + xigua_weight_base
            return {
                'visual': self.visual_weight / total,
                'acoustic': self.acoustic_weight / total,
                'haptic': self.haptic_weight / total,
                'xigua': xigua_weight_base / total,
            }

        # Dinamik ağırlıklar
        w_v = max(min_weight, self.visual_weight * signal_quality.visual_quality)
        w_a = max(min_weight, self.acoustic_weight * signal_quality.acoustic_quality)
        w_h = max(min_weight, self.haptic_weight * signal_quality.haptic_quality)
        xigua_q = (signal_quality.visual_quality + signal_quality.acoustic_quality) / 2.0
        w_x = max(min_weight, xigua_weight_base * xigua_q)

        total = w_v + w_a + w_h + w_x
        return {
            'visual': w_v / total,
            'acoustic': w_a / total,
            'haptic': w_h / total,
            'xigua': w_x / total,
        }

    def fuse_with_xigua(
        self,
        visual_probs: np.ndarray,
        acoustic_probs: np.ndarray,
        haptic_probs: Optional[np.ndarray] = None,
        hh_score: Optional[float] = None,
        xigua_brix: float = 10.5,
        signal_quality: Optional['SignalQuality'] = None,
    ) -> Dict[str, object]:
        """
        Xigua audio+image erken füzyon ve sinyal kalitesi destekli Late Fusion.

        Xigua Brix skoru olgunluk olasılıklarına çevrilir ve
        sinyal kalitesine göre dinamik ağırlıklarla birleştirilir.

        Args:
            visual_probs: Görsel model olasılıkları (3,)
            acoustic_probs: Akustik model olasılıkları (3,)
            haptic_probs: Haptik model olasılıkları (3,) - opsiyonel
            hh_score: HollowHeartDetector skoru (0-1) - opsiyonel
            xigua_brix: Xigua fusion Brix skoru
            signal_quality: Sinyal kalitesi nesnesi

        Returns:
            Birleştirilmiş sonuç
        """
        # Dinamik ağırlıklar
        weights = self.compute_dynamic_weights(signal_quality)

        # Xigua Brix → olasılıklar
        xigua_probs = self._brix_to_probabilities(xigua_brix)

        # Ağırlıklı birleştirme
        fused = (
            weights['visual'] * visual_probs +
            weights['acoustic'] * acoustic_probs +
            weights['xigua'] * xigua_probs
        )
        if haptic_probs is not None:
            fused += weights['haptic'] * haptic_probs
        else:
            # Haptik yoksa diğerlerini yeniden ölçekle
            remaining = weights['visual'] + weights['acoustic'] + weights['xigua']
            fused = fused / (remaining + 1e-10)

        # Hollow heart boost
        if hh_score is not None and hh_score > HH_DETECTION_THRESHOLD:
            boost_strength = (hh_score - HH_DETECTION_THRESHOLD) / (1.0 - HH_DETECTION_THRESHOLD + 1e-10)
            if hh_score >= HH_HIGH_CONFIDENCE_THRESHOLD:
                boost = 0.3 * boost_strength
            else:
                boost = 0.15 * boost_strength
            fused[2] += boost
            reduction = boost / 2.0
            fused[0] = max(0.01, fused[0] - reduction)
            fused[1] = max(0.01, fused[1] - reduction)

        fused = fused / (np.sum(fused) + 1e-10)

        class_id = int(np.argmax(fused))

        return {
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "confidence": float(fused[class_id]),
            "probabilities": {
                CLASS_LABELS[i]: float(fused[i])
                for i in range(len(fused))
            },
            "dynamic_weights": weights,
            "xigua_brix": xigua_brix,
            "xigua_probs": xigua_probs.tolist(),
            "signal_quality": {
                "visual": signal_quality.visual_quality if signal_quality else None,
                "acoustic": signal_quality.acoustic_quality if signal_quality else None,
                "haptic": signal_quality.haptic_quality if signal_quality else None,
            },
        }

    @staticmethod
    def _brix_to_probabilities(brix: float) -> np.ndarray:
        """Xigua Brix skorunu 3-sınıf olasılığa çevirir.

        Brix < 8  → Immature (%80)
        Brix 8–11 → Ripe (%80)
        Brix > 11 → Overripe (%60)
        """
        if brix < 8.0:
            t = min(1.0, max(0.0, brix / 8.0))
            return np.array([0.8 - 0.3 * t, 0.15 + 0.3 * t, 0.05])
        elif brix <= 11.0:
            return np.array([0.05, 0.85, 0.10])
        else:
            t = min(1.0, max(0.0, (brix - 11.0) / 3.0))
            return np.array([0.05, 0.6 - 0.3 * t, 0.35 + 0.3 * t])

    def fuse_multi_sample(
        self,
        visual_probs_list: list,
        acoustic_probs_list: list,
        haptic_probs_list: list = None
    ) -> Dict[str, object]:
        """
        Birden fazla ölçüm sonucunu birleştirir (temporal fusion).
        
        Aynı karpuz üzerinden alınan birden fazla ölçümü
        zamansal olarak birleştirir.
        
        Args:
            visual_probs_list: Görsel olasılık listesi
            acoustic_probs_list: Akustik olasılık listesi
            haptic_probs_list: Haptik olasılık listesi
            
        Returns:
            Birleştirilmiş sonuç
        """
        # Her ölçüm için modal fusion uygula
        fused_probs_list = []

        for i in range(len(visual_probs_list)):
            v = np.array(visual_probs_list[i])
            a = np.array(acoustic_probs_list[i])
            h = np.array(haptic_probs_list[i]) if haptic_probs_list else None

            fused = self.fuse_probabilities(v, a, h)
            fused_probs_list.append(fused)

        # Temporal ortalama
        avg_probs = np.mean(fused_probs_list, axis=0)
        avg_probs = avg_probs / (np.sum(avg_probs) + 1e-10)

        class_id = int(np.argmax(avg_probs))

        return {
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "confidence": float(avg_probs[class_id]),
            "probabilities": {
                CLASS_LABELS[i]: float(avg_probs[i])
                for i in range(len(avg_probs))
            },
            "n_measurements": len(visual_probs_list)
        }

