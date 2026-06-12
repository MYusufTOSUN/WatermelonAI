import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'acoustic_features.dart';
import 'haptic_features.dart';
import 'hh_features.dart';
import 'visual_features.dart';
import '../vi_liquid/mass_estimator.dart';

/// Top-level isolate-safe worker functions for `compute()`.
///
/// Dart's `compute(fn, message)` requires the function to be a top-level
/// (or static) callable. Wrapping each heavy feature extractor here lets
/// the UI thread stay responsive while DSP runs in a background isolate.

/// Acoustic 120-D extraction
Float64List acousticExtractIsolate(Float64List audio) {
  return AcousticFeatureExtractor().extract(audio);
}

/// Hollow-heart 8-D extraction
Float64List hhExtractIsolate(Float64List audio) {
  return HhFeatureExtractor().extract(audio);
}

/// Combined acoustic + HH from a single audio buffer — saves an extra
/// isolate roundtrip when both vectors are needed.
///
/// [hh] is the DNN input (training-mean substitution); [localHhScore] is
/// the LIVE 0-1 hollow-likelihood from the recording, used by the
/// Vi-Liquid trigger gate and the UI risk tile.
class AcousticHhResult {
  final Float64List acoustic;
  final Float64List hh;
  final double localHhScore;
  AcousticHhResult(this.acoustic, this.hh, this.localHhScore);
}

AcousticHhResult acousticAndHhIsolate(Float64List audio) {
  final extractor = HhFeatureExtractor();
  final a = AcousticFeatureExtractor().extract(audio);
  final h = extractor.extract(audio);
  final live = extractor.liveScore(audio);
  return AcousticHhResult(a, h, live);
}

/// Visual 11-D extraction (decodes + resizes inside isolate)
Float64List visualExtractIsolate(Uint8List imageBytes) {
  final decoded = img.decodeImage(imageBytes);
  if (decoded == null) {
    return Float64List(11);
  }
  final resized = img.copyResize(decoded, width: 256, height: 256);
  return VisualFeatureExtractor().extract(resized);
}

/// Mass estimation from image (isolate-safe)
MassEstimate massEstimateIsolate(Uint8List imageBytes) {
  final decoded = img.decodeImage(imageBytes);
  if (decoded == null) return MassEstimate.fallback();
  final resized = img.copyResize(decoded, width: 320, height: 320);
  return MassEstimator().estimate(resized);
}

/// Haptic 7-D extraction
class HapticInput {
  final List<double> x;
  final List<double> y;
  final List<double> z;
  HapticInput(this.x, this.y, this.z);
}

Float64List hapticExtractIsolate(HapticInput input) {
  return HapticFeatureExtractor().extract(input.x, input.y, input.z);
}
