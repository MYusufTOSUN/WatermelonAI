import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/ripeness_result.dart';

/// Runs the multimodal fusion TFLite model with 4 inputs.
///
/// Backend: data/models/fusion_model_fp16.tflite
///   acoustic_input: (1, 120) float32
///   visual_input:   (1, 11)  float32
///   haptic_input:   (1, 7)   float32
///   hh_input:       (1, 8)   float32
///   → output:       (1, 3)   float32 [immature, ripe, overripe]
class FusionModelService {
  static const String _modelAsset = "assets/models/fusion_model_fp16.tflite";

  Interpreter? _interpreter;
  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    if (_interpreter != null) return;
    _interpreter = await Interpreter.fromAsset(_modelAsset);
    _interpreter!.allocateTensors();
  }

  RipenessResult run({
    required Float64List acoustic,
    required Float64List visual,
    required Float64List haptic,
    required Float64List hh,
  }) {
    if (_interpreter == null) {
      throw StateError("Fusion model not loaded — call loadModel() first");
    }

    final byDim = <int, Object>{
      120: _toFloat32x1xN(acoustic, 120),
      11: _toFloat32x1xN(visual, 11),
      7: _toFloat32x1xN(haptic, 7),
      8: _toFloat32x1xN(hh, 8),
    };

    final inputTensors = _interpreter!.getInputTensors();
    final ordered = <Object>[];
    for (final t in inputTensors) {
      final dim = t.shape.length == 2 ? t.shape[1] : -1;
      final input = byDim[dim];
      if (input == null) {
        throw StateError("Unexpected fusion input dim: $dim");
      }
      ordered.add(input);
    }

    final output =
        List<List<double>>.generate(1, (_) => List<double>.filled(3, 0.0));
    _interpreter!.runForMultipleInputs(ordered, <int, Object>{0: output});

    final probs = _softmaxIfNeeded(output[0]);

    int classId = 0;
    double best = probs[0];
    if (probs[1] > best) {
      best = probs[1];
      classId = 1;
    }
    if (probs[2] > best) classId = 2;

    return RipenessResult(
      classId: classId,
      pImmature: probs[0],
      pRipe: probs[1],
      pOverripe: probs[2],
    );
  }

  List<List<double>> _toFloat32x1xN(Float64List src, int n) {
    final padded = List<double>.filled(n, 0.0);
    for (int i = 0; i < n && i < src.length; i++) {
      final v = src[i];
      padded[i] = v.isFinite ? v : 0.0;
    }
    return [padded];
  }

  List<double> _softmaxIfNeeded(List<double> raw) {
    final total = raw.fold<double>(0.0, (a, b) => a + b);
    if (total > 0.95 &&
        total < 1.05 &&
        raw.every((v) => v >= 0 && v <= 1)) {
      return raw;
    }
    final maxV = raw.reduce(math.max);
    final exps = raw.map((v) => math.exp(v - maxV)).toList();
    final sum = exps.fold<double>(0.0, (a, b) => a + b);
    if (sum <= 0) return const [1 / 3, 1 / 3, 1 / 3];
    return exps.map((v) => v / sum).toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
