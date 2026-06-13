import 'package:flutter/material.dart';

/// Vi-Liquid (active haptic + SRR + physical Elasticity Index) result.
/// Produced by MobileFusionEngine.fuse() given fusion DNN probs + f2 + mass.
class ViLiquidResult {
  final double f2Hz; // SRR-extracted dominant resonance
  final double ei; // f2^2 * mass^(2/3)
  final double eiNormalized; // 0..1
  final double massKg;
  final double finalScore; // w1*P_ripe + w2*EI_norm
  final int finalClassId; // 0=immature, 1=ripe, 2=overripe/defective
  final String finalClassLabel;
  final bool isHollowHeart;
  final double combinedPImmature;
  final double combinedPRipe;
  final double combinedPOverripe;

  ViLiquidResult({
    required this.f2Hz,
    required this.ei,
    required this.eiNormalized,
    required this.massKg,
    required this.finalScore,
    required this.finalClassId,
    required this.finalClassLabel,
    required this.isHollowHeart,
    required this.combinedPImmature,
    required this.combinedPRipe,
    required this.combinedPOverripe,
  });

  /// Binary verdict — DNN kararina dayanir. isHollowHeart KARARI ETKILEMEZ
  /// (yalnizca bilgi amacli "titresim cevabi zayif" uyarisidir).
  bool get isEdible => finalClassId == 1;
  String get verdict => isEdible ? "Yenir" : "Yenmez";
  double get verdictConfidence =>
      isEdible ? combinedPRipe : (combinedPImmature + combinedPOverripe);

  Color get verdictColor => isEdible
      ? const Color(0xFF1B7F3A)
      : const Color(0xFFE53935);

  /// Quick interpretation string
  String get diagnosticLine =>
      "f2=${f2Hz.toStringAsFixed(0)} Hz · "
      "EI=${ei.toStringAsExponential(2)} · "
      "Kütle=${massKg.toStringAsFixed(1)} kg";

  /// ─── Halk dili sürümleri ──────────────────────────────────────────

  String get simpleVerdict => isEdible ? "AL" : "ALMA";

  String get simpleHeadline =>
      isEdible ? "Bu karpuz iyi gibi" : "Bu karpuz iyi değil";

  String get simpleLabel {
    switch (finalClassId) {
      case 0:
        return "Henüz ham";
      case 1:
        return "Tam kıvamında";
      case 2:
        return "Geçmiş olabilir";
      default:
        return "Belirsiz";
    }
  }

  /// Sade neden, halk diline yakın
  String get simpleReason {
    switch (finalClassId) {
      case 0:
        return "Daha sert ve ham hissediliyor, bekleyebilir.";
      case 1:
        return "Sesi olgun bir karpuza uyuyor.";
      case 2:
        return "Yumuşaklık ve cevap zayıf, içi geçmiş olabilir.";
      default:
        return "";
    }
  }
}
