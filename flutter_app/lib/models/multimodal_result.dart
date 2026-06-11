import 'package:flutter/material.dart';

import 'ripeness_result.dart';
import 'vi_liquid_result.dart';

/// Aggregated result of running the multimodal pipeline.
/// Holds the ML fusion prediction (acoustic+visual+HH→DNN), the
/// visual-only baseline (MobileNetV3), and the Vi-Liquid active haptic
/// late-fusion result (physical f2+EI rule).
class MultimodalResult {
  final RipenessResult fusion;
  final RipenessResult? visualOnly;
  final ViLiquidResult? viLiquid;
  final double f2Hz; // acoustic-derived f2 (from microphone)
  final double f2Db;
  final double hhScore;
  final double contactQuality;
  final double signalRms;
  final DateTime timestamp;

  MultimodalResult({
    required this.fusion,
    this.visualOnly,
    this.viLiquid,
    required this.f2Hz,
    required this.f2Db,
    required this.hhScore,
    required this.contactQuality,
    required this.signalRms,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Helper: textual diagnosis line based on f2 and contact quality.
  String get diagnosticLine {
    final hh = (hhScore * 100).toStringAsFixed(0);
    return "f2=${f2Hz.toStringAsFixed(0)} Hz · "
        "HH=%$hh · "
        "Temas=${(contactQuality * 100).toStringAsFixed(0)}/100";
  }

  Color get verdictColor => fusion.verdictColor;
}
