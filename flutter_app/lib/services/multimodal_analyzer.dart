import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../models/multimodal_result.dart';
import 'audio_recorder_service.dart';
import 'feature_extractors/isolate_workers.dart';
import 'fusion_model_service.dart';
import 'sensor_recorder_service.dart';
import 'vi_liquid/mobile_fusion_engine.dart';
import 'vi_liquid/srr_processor.dart';
import 'visual_classifier_service.dart';

/// Orchestrates the on-device multimodal pipeline + Vi-Liquid active
/// haptic late fusion.
///
/// Heavy DSP runs in background isolates so the UI thread stays alive
/// during the 2-5 second analysis step.
class MultimodalAnalyzer {
  final VisualClassifierService _visualClassifier = VisualClassifierService();
  final FusionModelService _fusion = FusionModelService();
  final SrrProcessor _srr = SrrProcessor();
  final MobileFusionEngine _mobileFusion = MobileFusionEngine();

  bool _loaded = false;
  bool get isLoaded => _loaded;

  Future<void> loadModels() async {
    if (_loaded) return;
    await _visualClassifier.loadModel();
    await _fusion.loadModel();
    _loaded = true;
  }

  Future<MultimodalResult> analyze({
    required File photo,
    required String audioWavPath,
    required SensorRecording sensors,
    required double massKg,
  }) async {
    if (!_loaded) await loadModels();

    // Visual: bytes → isolate → 11-D
    final photoBytes = await photo.readAsBytes();
    final visualVecFuture = compute(visualExtractIsolate, photoBytes);

    // Visual-only baseline (MobileNetV3)
    final visualOnlyFuture = _visualClassifier.classifyImage(photo);

    // Audio: load WAV, trim silence, run isolate for 120+8
    final audio = await AudioRecorderService.loadWavAsFloat64(audioWavPath);
    final trimmed = AudioRecorderService.trimSilence(audio);
    final acousticHhFuture = compute(acousticAndHhIsolate, trimmed);

    // Haptic (passive): extraction in isolate (still passed to fusion DNN
    // for completeness — backend trained with zeros so impact is minimal)
    final hapticVecFuture = compute(
      hapticExtractIsolate,
      HapticInput(sensors.x, sensors.y, sensors.z),
    );

    // Vi-Liquid SRR on z-axis IMU (active vibration response)
    final viLiquidFuture = compute(_runSrrInIsolate, sensors.z);

    final visualVec = await visualVecFuture;
    final acousticHh = await acousticHhFuture;
    final hapticVec = await hapticVecFuture;
    final visualOnly = await visualOnlyFuture;
    final srrResult = await viLiquidFuture;

    // Fusion DNN (TFLite, main isolate)
    final fusion = _fusion.run(
      acoustic: acousticHh.acoustic,
      visual: visualVec,
      haptic: hapticVec,
      hh: acousticHh.hh,
    );

    // Vi-Liquid late fusion: combine DNN probs with physical EI rule.
    // acousticHhScore gates the Hollow Heart trigger (dual confirmation).
    final viLiquid = _mobileFusion.fuse(
      pImmature: fusion.pImmature,
      pRipe: fusion.pRipe,
      pOverripe: fusion.pOverripe,
      f2Hz: srrResult.f2Hz,
      massKg: massKg,
      acousticHhScore: acousticHh.hh[0],
    );

    final acousticF2Hz = acousticHh.acoustic[111];
    final acousticF2Db = acousticHh.acoustic[112];
    final hhScore = acousticHh.hh[0];
    final contactQuality = hapticVec[6];
    final signalRms = _rms(trimmed);

    return MultimodalResult(
      fusion: fusion,
      visualOnly: visualOnly,
      viLiquid: viLiquid,
      f2Hz: acousticF2Hz,
      f2Db: acousticF2Db,
      hhScore: hhScore,
      contactQuality: contactQuality,
      signalRms: signalRms,
    );
  }

  double _rms(Float64List a) {
    if (a.isEmpty) return 0.0;
    double s = 0.0;
    for (final v in a) {
      s += v * v;
    }
    return math.sqrt(s / a.length);
  }

  void dispose() {
    _visualClassifier.dispose();
    _fusion.dispose();
    _loaded = false;
  }
}

/// Top-level wrapper so compute() can run SrrProcessor in isolate.
SrrResult _runSrrInIsolate(List<double> z) {
  return SrrProcessor().reconstruct(z);
}
