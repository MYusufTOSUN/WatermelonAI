import 'dart:math' as math;

import 'package:image/image.dart' as img;

/// Photo-based watermelon mass estimation.
///
/// Backend ref: backend/module_a/volume_estimator.py (Koc 2007 disk method).
/// Full disk-method requires pixel→cm calibration (reference object) which
/// isn't practical on phone. This simplified version:
///   1. Coarse HSV segmentation (watermelon is mostly green/dark)
///   2. Find largest connected blob → bounding ellipse approximation
///   3. Use ratio (blob_area / image_area) + aspect ratio to heuristically
///      map to typical watermelon mass range (1.5–8 kg).
///
/// Returns a [MassEstimate] with the estimated mass + segmentation
/// area ratio + confidence (low/medium/high based on segmentation quality).
class MassEstimator {
  /// Density used by backend (Koc 2007): 0.98 g/cm³ ≈ water.
  static const double densityGramsPerCm3 = 0.98;

  /// Typical watermelon mass range (kg) used for heuristic mapping.
  static const double minMassKg = 1.5;
  static const double maxMassKg = 8.0;

  MassEstimate estimate(img.Image src) {
    final w = src.width;
    final h = src.height;
    final totalPixels = w * h;
    if (totalPixels == 0) {
      return MassEstimate.fallback();
    }

    // Build coarse mask: dark green + green-dominant pixels
    final mask = List<int>.filled(totalPixels, 0);
    int blobCount = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = src.getPixel(x, y);
        final r = p.r.toDouble();
        final g = p.g.toDouble();
        final b = p.b.toDouble();
        // Green-dominant + medium-to-dark
        final isGreenDom = g > r * 0.95 && g > b * 0.95;
        final notTooBright = (r + g + b) / 3.0 < 200.0;
        final notTooDark = (r + g + b) / 3.0 > 25.0;
        if (isGreenDom && notTooBright && notTooDark) {
          mask[y * w + x] = 1;
          blobCount++;
        }
      }
    }

    if (blobCount < totalPixels * 0.02) {
      // Too little green → can't segment, fallback
      return MassEstimate(
        massKg: 4.0,
        areaRatio: blobCount / totalPixels,
        confidence: MassConfidence.low,
        method: "fallback-default",
      );
    }

    // Bounding box of green pixels
    int minX = w, maxX = 0, minY = h, maxY = 0;
    int blobInside = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (mask[y * w + x] == 1) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          blobInside++;
        }
      }
    }
    final bboxW = (maxX - minX + 1).clamp(1, w);
    final bboxH = (maxY - minY + 1).clamp(1, h);
    final bboxArea = bboxW * bboxH;
    // "Compactness" — how filled the bounding box is by green
    final compactness = bboxArea > 0 ? blobInside / bboxArea : 0.0;
    // Aspect ratio (watermelon elongated ≈ 1.2-1.6)
    final aspect = bboxW > bboxH ? bboxW / bboxH : bboxH / bboxW;
    final areaRatio = blobInside / totalPixels;

    // Heuristic mass: bigger blob + compact + watermelon-aspect → larger karpuz
    // Map areaRatio in [0.05, 0.65] linearly to [minMassKg, maxMassKg]
    final clampedArea = areaRatio.clamp(0.05, 0.65);
    final lerpT = (clampedArea - 0.05) / (0.65 - 0.05);
    var massKg = minMassKg + (maxMassKg - minMassKg) * lerpT;

    // Slight aspect adjustment: very round (aspect~1) → smaller, elongated → bigger
    if (aspect > 1.6) massKg *= 1.05;
    if (aspect < 1.1) massKg *= 0.95;

    // Confidence based on compactness + segmentation quality
    final confidence = compactness > 0.55 && areaRatio > 0.10
        ? MassConfidence.high
        : compactness > 0.35
            ? MassConfidence.medium
            : MassConfidence.low;

    return MassEstimate(
      massKg: massKg.clamp(minMassKg, maxMassKg).toDouble(),
      areaRatio: areaRatio,
      bboxAspect: aspect,
      compactness: compactness,
      confidence: confidence,
      method: "hsv-blob-heuristic",
    );
  }
}

enum MassConfidence { low, medium, high }

class MassEstimate {
  final double massKg;
  final double areaRatio;
  final double? bboxAspect;
  final double? compactness;
  final MassConfidence confidence;
  final String method;

  MassEstimate({
    required this.massKg,
    required this.areaRatio,
    this.bboxAspect,
    this.compactness,
    required this.confidence,
    required this.method,
  });

  factory MassEstimate.fallback() => MassEstimate(
        massKg: 4.0,
        areaRatio: 0.0,
        confidence: MassConfidence.low,
        method: "fallback",
      );

  String get confidenceLabel {
    switch (confidence) {
      case MassConfidence.low:
        return "düşük güven";
      case MassConfidence.medium:
        return "orta güven";
      case MassConfidence.high:
        return "yüksek güven";
    }
  }
}
