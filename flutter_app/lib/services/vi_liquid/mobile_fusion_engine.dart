import 'dart:math' as math;

import '../../models/vi_liquid_result.dart';

/// Dart port of backend/module_e/mobile_fusion.py — Vi-Liquid late
/// fusion combining ML fusion DNN probabilities with the physical
/// Elasticity Index (EI) derived from the SRR-reconstructed signal.
///
/// Formula (Koc & Akbalik 2025, Yamamoto 1980):
///   EI = f2^2 * mass^(2/3)
///   score = w1_visual * P_ripe + w2_ei * normalize(EI)
///
/// Hollow Heart trigger:
///   f2 < 134 Hz AND visually mature (P_ripe + P_overripe > 0.5)
///   → final class = "Overripe/Defective"
class MobileFusionEngine {
  static const double w1Visual = 0.6;
  static const double w2Ei = 0.4;
  static const double f2HollowThreshold = 134.0;
  static const double eiMin = 0.0;
  static const double eiMax = 1e7;

  /// Acoustic hollow-heart score (0-1) required to CONFIRM the f2 trigger.
  /// The SRR-derived f2 alone is unreliable on phones (cubic interpolation
  /// concentrates energy near the 50 Hz band floor), which caused false
  /// "İçi Geçmiş" verdicts on good watermelons. Requiring the independent
  /// microphone-side HH indicator to agree cuts those false alarms.
  static const double hhConfirmThreshold = 0.45;

  ViLiquidResult fuse({
    required double pImmature,
    required double pRipe,
    required double pOverripe,
    required double f2Hz,
    required double massKg,
    double acousticHhScore = 0.0,
  }) {
    final ei = computeEi(f2Hz, massKg);
    final eiNorm = _normalizeEi(ei);
    final scoreFinal = w1Visual * pRipe + w2Ei * eiNorm;

    final isMatureLooking = (pRipe + pOverripe) > 0.5;
    // Dual-confirmation Hollow Heart trigger:
    //   (1) physical rule: SRR f2 below Yamamoto threshold, AND
    //   (2) the acoustic 8-D HH detector independently flags risk.
    final isHollowHeart = isMatureLooking &&
        (f2Hz > 0 && f2Hz < f2HollowThreshold) &&
        acousticHhScore >= hhConfirmThreshold;

    // KARAR HER ZAMAN DNN'DEN GELIR. Vi-Liquid fiziksel kurali (SRR f2 + EI)
    // yalnizca BILGI AMACLIDIR ve sonucu EZMEZ.
    //
    // Gerekce: telefon donaniminda SRR f2 (cubic interpolation) enerjiyi
    // 50 Hz tabanina sikistirdigi icin guvenilmez; basitlestirilmis HH skoru
    // da sessiz/gurultulu kayitta yuksek cikabiliyor. Bu ikisi birlikte iyi
    // bir karpuzu yanlislikla "Ici Gecmis" isaretliyordu (saha hatasi).
    // DNN zaten 3-sinifli ve gercek HH-etiketli veriyle egitildi; ici gecmis
    // tespitini kendi yapar. Vi-Liquid metrikleri (f2, EI, ici-bos riski)
    // sonuc ekraninda destekleyici olcum olarak gosterilir.
    final candidates = [
      ("Olgunlaşmamış", pImmature, 0),
      ("Olgun", pRipe, 1),
      ("İçi Geçmiş", pOverripe, 2),
    ];
    candidates.sort((a, b) => b.$2.compareTo(a.$2));
    final finalClass = candidates.first.$1;
    final finalClassId = candidates.first.$3;

    return ViLiquidResult(
      f2Hz: f2Hz,
      ei: ei,
      eiNormalized: eiNorm,
      massKg: massKg,
      finalScore: scoreFinal,
      finalClassId: finalClassId,
      finalClassLabel: finalClass,
      // Bilgi amacli bayrak: titresim cevabi dusuk mu? (karari ETKILEMEZ,
      // sadece "titresim cevabi zayif" uyarisi olarak gosterilir)
      isHollowHeart: isHollowHeart,
      // Olasiliklar dogrudan DNN ciktisidir — override yok.
      combinedPImmature: pImmature,
      combinedPRipe: pRipe,
      combinedPOverripe: pOverripe,
    );
  }

  /// EI = f2^2 * mass^(2/3). Returns 0 for invalid inputs.
  static double computeEi(double f2Hz, double massKg) {
    if (f2Hz <= 0 || massKg <= 0) return 0.0;
    return f2Hz * f2Hz * math.pow(massKg, 2.0 / 3.0);
  }

  double _normalizeEi(double ei) {
    final clamped = ei.clamp(eiMin, eiMax);
    if (eiMax <= eiMin) return 0.0;
    return (clamped - eiMin) / (eiMax - eiMin);
  }
}
