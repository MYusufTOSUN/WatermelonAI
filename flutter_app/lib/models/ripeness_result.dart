import 'package:flutter/material.dart';

/// Karpuz olgunluk analiz sonucu
class RipenessResult {
  /// Detay sınıf (0=Olgunlaşmamış, 1=Olgun, 2=İçi Geçmiş)
  final int classId;

  /// 3 sınıf olasılıkları (toplam = 1.0)
  final double pImmature;
  final double pRipe;
  final double pOverripe;

  final DateTime timestamp;

  RipenessResult({
    required this.classId,
    required this.pImmature,
    required this.pRipe,
    required this.pOverripe,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// En yüksek detay sınıf olasılığı
  double get confidence {
    switch (classId) {
      case 0:
        return pImmature;
      case 1:
        return pRipe;
      case 2:
        return pOverripe;
      default:
        return 0.0;
    }
  }

  /// Yenir olasılığı (yalnızca "olgun")
  double get pEdible => pRipe;

  /// Yenmez olasılığı (olgunlaşmamış + içi geçmiş)
  double get pNotEdible => pImmature + pOverripe;

  /// Binary karar: yenir mi?
  bool get isEdible => pEdible > pNotEdible;

  /// Binary etiket
  String get edibleVerdict => isEdible ? "Yenir" : "Yenmez";

  /// Binary güven skoru (0.5 - 1.0)
  double get edibleConfidence => isEdible ? pEdible : pNotEdible;

  /// Detay 3-sınıf etiketi
  String get detailLabel {
    switch (classId) {
      case 0:
        return "Olgunlaşmamış";
      case 1:
        return "Olgun";
      case 2:
        return "İçi Geçmiş";
      default:
        return "Bilinmiyor";
    }
  }

  /// Detay sınıf kısa açıklama
  String get detailDescription {
    switch (classId) {
      case 0:
        return "Karpuz henüz olgunlaşmamış, bekletilebilir.";
      case 1:
        return "Karpuz olgun, tüketime uygun.";
      case 2:
        return "Karpuzun içi geçmiş olabilir, dikkatli ol.";
      default:
        return "";
    }
  }

  /// Binary karar rengi
  Color get verdictColor => isEdible
      ? const Color(0xFF2E7D32)
      : const Color(0xFFC62828);

  /// Detay sınıf rengi
  Color get detailColor {
    switch (classId) {
      case 0:
        return const Color(0xFF7CB342); // açık yeşil
      case 1:
        return const Color(0xFF2E7D32); // koyu yeşil
      case 2:
        return const Color(0xFFEF6C00); // turuncu
      default:
        return Colors.grey;
    }
  }

  /// Detay sınıf ikonu
  IconData get detailIcon {
    switch (classId) {
      case 0:
        return Icons.eco_rounded;
      case 1:
        return Icons.verified_rounded;
      case 2:
        return Icons.warning_rounded;
      default:
        return Icons.help_rounded;
    }
  }

  /// ─── Halk dili sürümleri ──────────────────────────────────────────

  /// "AL" / "ALMA"
  String get simpleVerdict => isEdible ? "AL" : "ALMA";

  /// "Bu karpuz iyi gibi" / "Bu karpuz iyi değil"
  String get simpleHeadline =>
      isEdible ? "Bu karpuz iyi gibi" : "Bu karpuz iyi değil";

  /// Halk dilinde sınıf etiketi (Tam kıvamında / Henüz ham / Geçmiş olabilir)
  String get simpleLabel {
    switch (classId) {
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

  /// Sade neden açıklaması, halk diline yakın
  String get simpleReason {
    switch (classId) {
      case 0:
        return "Daha bekleyebilir, henüz olgun değil.";
      case 1:
        return "Sesi ve hisli ölçümler iyi karpuza uyuyor.";
      case 2:
        return "İçi boş ya da fazla yumuşak olabilir.";
      default:
        return "";
    }
  }

  Map<String, dynamic> toJson() => {
        "classId": classId,
        "label": detailLabel,
        "pImmature": pImmature,
        "pRipe": pRipe,
        "pOverripe": pOverripe,
        "isEdible": isEdible,
        "timestamp": timestamp.toIso8601String(),
      };
}
