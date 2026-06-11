import 'dart:math' as math;
import 'dart:typed_data';

/// 7-D haptic feature vector matching backend/module_b/pressure_analyzer.py:
///   extract_pressure_features().
///
/// Layout:
///   [0] contact_pressure (0-1) — normalized gravity deviation / good_threshold
///   [1] gravity_deviation (m/s^2)
///   [2] vibration_rms — high-pass filtered Z RMS
///   [3] tilt_angle_norm (0-1) — arccos(mean_z / 9.81) / 90 deg
///   [4] magnitude_mean — mean of sqrt(x^2+y^2+z^2)
///   [5] stability_score (0-1) — 1 - mag_std/stability_threshold
///   [6] contact_quality_score (0-1) — heuristic mapping
class HapticFeatureExtractor {
  static const int featureDim = 7;
  static const double gravityRef = 9.81;
  static const double lightThreshold = 0.3;
  static const double goodThreshold = 0.8;
  static const double strongThreshold = 2.5;
  static const double stabilityThreshold = 0.15;

  /// All input lists must be the same length, units m/s^2.
  Float64List extract(
    List<double> accelX,
    List<double> accelY,
    List<double> accelZ,
  ) {
    final out = Float64List(featureDim);
    final n = math.min(accelX.length, math.min(accelY.length, accelZ.length));
    if (n < 4) return out;

    // Magnitude
    final magnitude = Float64List(n);
    double meanZ = 0.0;
    for (int i = 0; i < n; i++) {
      magnitude[i] = math.sqrt(
        accelX[i] * accelX[i] +
            accelY[i] * accelY[i] +
            accelZ[i] * accelZ[i],
      );
      meanZ += accelZ[i];
    }
    meanZ /= n;

    // Gravity deviation
    double gravDevSum = 0.0;
    for (int i = 0; i < n; i++) {
      gravDevSum += (accelZ[i] - gravityRef).abs();
    }
    final gravityDeviation = gravDevSum / n;

    // Magnitude mean/std
    double magSum = 0.0;
    for (final v in magnitude) {
      magSum += v;
    }
    final magMean = magSum / n;
    double magVar = 0.0;
    for (final v in magnitude) {
      final d = v - magMean;
      magVar += d * d;
    }
    final magStd = math.sqrt(magVar / n);

    // Vibration RMS — simple high-pass (subtract mean Z)
    double vibSum = 0.0;
    for (int i = 0; i < n; i++) {
      final hp = accelZ[i] - meanZ;
      vibSum += hp * hp;
    }
    final vibrationRms = math.sqrt(vibSum / n);

    // Tilt angle
    final cosVal = (meanZ / gravityRef).clamp(-1.0, 1.0);
    final tiltDeg = math.acos(cosVal) * 180.0 / math.pi;
    final tiltNorm = (tiltDeg / 90.0).clamp(0.0, 1.0);

    // Contact pressure (0-1) — normalize gravity deviation to good_threshold
    final contactPressure =
        (gravityDeviation / goodThreshold).clamp(0.0, 1.0);

    // Stability score
    final stabilityScore =
        (1.0 - magStd / stabilityThreshold).clamp(0.0, 1.0);

    // Contact quality
    String quality;
    if (gravityDeviation < lightThreshold) {
      quality = "no_contact";
    } else if (gravityDeviation < goodThreshold) {
      quality = "light_contact";
    } else if (gravityDeviation < strongThreshold) {
      quality = "good_contact";
    } else {
      quality = "too_strong";
    }
    final qualityMap = {
      "no_contact": 0.0,
      "light_contact": 0.3,
      "good_contact": 0.8,
      "too_strong": 0.5,
    };
    final qualityScore = qualityMap[quality] ?? 0.0;

    out[0] = contactPressure;
    out[1] = gravityDeviation;
    out[2] = vibrationRms;
    out[3] = tiltNorm;
    out[4] = magMean;
    out[5] = stabilityScore;
    out[6] = qualityScore;
    return out;
  }
}
