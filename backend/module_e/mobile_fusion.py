"""
On-device Late Fusion Engine

ROLE: Senior Mobile AI Deployment Engineer

Bu modül, mobil tarafta uygulanacak fusion mantığının Python referans
uygulamasını verir. Aynı mantığın Kotlin / Swift karşılığını aşağıda
yorumlarda örnekliyoruz.

Fusion Logic:
  - P_visual: MRD-YOLO / Vision modelinden gelen sınıf olasılıkları
  - EI: Elasticity Index  (EI = f2^2 * m^(2/3))

Final skor:
  Score_final = w1 * P_ripe + w2 * Normalize(EI)

Hollow Heart / Defective Trigger:
  - Eğer f2 < 134 Hz ve Vision modeli görünüş olarak "mature" (Ripe/Overripe)
    sınıfına yüksek olasılık veriyorsa, kalite "Overripe/Defective" olarak işaretlenir.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FusionConfig:
    w1_visual: float = 0.6
    w2_ei: float = 0.4
    f2_hollow_heart_threshold: float = 134.0  # Hz
    ei_min: float = 0.0
    ei_max: float = 1e7  # EI normalizasyonu için kaba aralık (gerekirse datasetten kalibre et)


class MobileFusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    def compute_ei(self, f2: float, mass_kg: float) -> float:
        """
        EI = f2^2 * m^(2/3)
        - f2: Hz
        - mass_kg: kg
        """
        if f2 <= 0 or mass_kg <= 0:
            return 0.0
        return float((f2 ** 2) * (mass_kg ** (2.0 / 3.0)))

    def _normalize_ei(self, ei: float) -> float:
        c = self.config
        ei_clamped = max(c.ei_min, min(c.ei_max, ei))
        if c.ei_max <= c.ei_min:
            return 0.0
        return (ei_clamped - c.ei_min) / (c.ei_max - c.ei_min)

    def fuse(
        self,
        vision_probs: Dict[str, float],
        f2_hz: float,
        mass_kg: float,
    ) -> Dict[str, float]:
        """
        Args:
          vision_probs: {"Immature": p0, "Ripe": p1, "Overripe": p2}
          f2_hz: dominant frequency from 1600 Hz reconstructed vibration
          mass_kg: fruit mass (from visual volume + scale)
        """
        cfg = self.config

        p_ripe = float(vision_probs.get("Ripe", 0.0))
        p_overripe = float(vision_probs.get("Overripe", 0.0))
        p_immature = float(vision_probs.get("Immature", 0.0))

        ei = self.compute_ei(f2_hz, mass_kg)
        ei_norm = self._normalize_ei(ei)

        score_final = cfg.w1_visual * p_ripe + cfg.w2_ei * ei_norm

        # Hollow Heart / Defective Trigger:
        # - f2 < threshold ve görünüş olarak "mature" (Ripe/Overripe) ise
        is_mature_looking = (p_ripe + p_overripe) > 0.5
        is_hollow_heart = is_mature_looking and (f2_hz < cfg.f2_hollow_heart_threshold)

        # Basit karar:
        # - Eğer hollow heart tetiklenmişse sınıfı Overripe/Defective'e çek
        if is_hollow_heart:
            final_class = "Overripe/Defective"
        else:
            # En yüksek olasılıklı sınıf
            label_order: List[Tuple[str, float]] = [
                ("Immature", p_immature),
                ("Ripe", p_ripe),
                ("Overripe", p_overripe),
            ]
            label_order.sort(key=lambda x: x[1], reverse=True)
            final_class = label_order[0][0]

        return {
            "final_score": float(score_final),
            "final_class": final_class,
            "p_immature": p_immature,
            "p_ripe": p_ripe,
            "p_overripe": p_overripe,
            "ei": float(ei),
            "ei_normalized": float(ei_norm),
            "f2_hz": float(f2_hz),
            "mass_kg": float(mass_kg),
            "is_hollow_heart": bool(is_hollow_heart),
        }


# ---------------------------------------------------------------------------
# Kotlin örneği (Android – Task Library veya Interpreter ile)
# ---------------------------------------------------------------------------
#
# data class FusionConfig(
#     val w1Visual: Float = 0.6f,
#     val w2Ei: Float = 0.4f,
#     val f2Threshold: Float = 134f,
#     val eiMin: Float = 0f,
#     val eiMax: Float = 1e7f
# )
#
# class MobileFusionEngineKt(private val cfg: FusionConfig = FusionConfig()) {
#     fun computeEi(f2Hz: Float, massKg: Float): Float {
#         if (f2Hz <= 0f || massKg <= 0f) return 0f
#         return (f2Hz * f2Hz * Math.pow(massKg.toDouble(), 2.0 / 3.0)).toFloat()
#     }
#
#     private fun normalizeEi(ei: Float): Float {
#         val clamped = ei.coerceIn(cfg.eiMin, cfg.eiMax)
#         if (cfg.eiMax <= cfg.eiMin) return 0f
#         return (clamped - cfg.eiMin) / (cfg.eiMax - cfg.eiMin)
#     }
#
#     fun fuse(
#         pImmature: Float,
#         pRipe: Float,
#         pOverripe: Float,
#         f2Hz: Float,
#         massKg: Float
#     ): Map<String, Any> {
#         val ei = computeEi(f2Hz, massKg)
#         val eiNorm = normalizeEi(ei)
#         val scoreFinal = cfg.w1Visual * pRipe + cfg.w2Ei * eiNorm
#
#         val isMatureLooking = (pRipe + pOverripe) > 0.5f
#         val isHollowHeart = isMatureLooking && (f2Hz < cfg.f2Threshold)
#
#         val finalClass = if (isHollowHeart) {
#             "Overripe/Defective"
#         } else {
#             listOf(
#                 "Immature" to pImmature,
#                 "Ripe" to pRipe,
#                 "Overripe" to pOverripe
#             ).maxBy { it.second }.first
#         }
#
#         return mapOf(
#             "final_score" to scoreFinal,
#             "final_class" to finalClass,
#             "ei" to ei,
#             "ei_normalized" to eiNorm,
#             "f2_hz" to f2Hz,
#             "mass_kg" to massKg,
#             "is_hollow_heart" to isHollowHeart
#         )
#     }
# }
#
# ---------------------------------------------------------------------------
# Swift örneği (iOS – TensorFlowLite Swift)
# ---------------------------------------------------------------------------
#
# struct FusionConfigSwift {
#     let w1Visual: Float = 0.6
#     let w2Ei: Float = 0.4
#     let f2Threshold: Float = 134.0
#     let eiMin: Float = 0.0
#     let eiMax: Float = 1e7
# }
#
# final class MobileFusionEngineSwift {
#     private let cfg = FusionConfigSwift()
#
#     private func computeEi(f2Hz: Float, massKg: Float) -> Float {
#         if f2Hz <= 0 || massKg <= 0 { return 0 }
#         return Float(pow(Double(f2Hz), 2.0) * pow(Double(massKg), 2.0/3.0))
#     }
#
#     private func normalizeEi(_ ei: Float) -> Float {
#         let clamped = max(cfg.eiMin, min(cfg.eiMax, ei))
#         guard cfg.eiMax > cfg.eiMin else { return 0 }
#         return (clamped - cfg.eiMin) / (cfg.eiMax - cfg.eiMin)
#     }
#
#     func fuse(
#         pImmature: Float,
#         pRipe: Float,
#         pOverripe: Float,
#         f2Hz: Float,
#         massKg: Float
#     ) -> [String: Any] {
#         let ei = computeEi(f2Hz: f2Hz, massKg: massKg)
#         let eiNorm = normalizeEi(ei)
#         let scoreFinal = cfg.w1Visual * pRipe + cfg.w2Ei * eiNorm
#
#         let isMatureLooking = (pRipe + pOverripe) > 0.5
#         let isHollowHeart = isMatureLooking && (f2Hz < cfg.f2Threshold)
#
#         let finalClass: String
#         if isHollowHeart {
#             finalClass = "Overripe/Defective"
#         } else {
#             let candidates: [(String, Float)] = [
#                 ("Immature", pImmature),
#                 ("Ripe", pRipe),
#                 ("Overripe", pOverripe),
#             ]
#             finalClass = candidates.max(by: { $0.1 < $1.1 })?.0 ?? "Ripe"
#         }
#
#         return [
#             "final_score": scoreFinal,
#             "final_class": finalClass,
#             "ei": ei,
#             "ei_normalized": eiNorm,
#             "f2_hz": f2Hz,
#             "mass_kg": massKg,
#             "is_hollow_heart": isHollowHeart,
#         ]
#     }
# }


