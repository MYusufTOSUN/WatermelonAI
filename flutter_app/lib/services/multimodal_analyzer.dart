import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../models/multimodal_result.dart';
import 'audio_recorder_service.dart';
import 'dsp/dsp_utils.dart';
import 'feature_extractors/isolate_workers.dart';
import 'fusion_model_service.dart';
import 'sensor_recorder_service.dart';
import 'vi_liquid/mobile_fusion_engine.dart';
import 'vi_liquid/srr_processor.dart';

/// Orchestrates the on-device multimodal pipeline + Vi-Liquid active
/// haptic late fusion.
///
/// Heavy DSP runs in background isolates so the UI thread stays alive
/// during the 2-5 second analysis step.
///
/// NOTE: The standalone MobileNetV3 "visual-only" baseline was removed
/// from the live path — it scored only ~30% on held-out watermelons
/// (subject leakage) and was no longer displayed anywhere, so running it
/// per analysis just wasted ~1 second. The 11-D handcrafted visual
/// features still feed the fusion DNN; the MobileNetV3 model artifact is
/// kept in the repo for the report.
class MultimodalAnalyzer {
  final FusionModelService _fusion = FusionModelService();
  final MobileFusionEngine _mobileFusion = MobileFusionEngine();

  bool _loaded = false;
  bool get isLoaded => _loaded;

  Future<void> loadModels() async {
    if (_loaded) return;
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

    // Visual: bytes → isolate → 11-D handcrafted features
    final photoBytes = await photo.readAsBytes();
    final visualVecFuture = compute(visualExtractIsolate, photoBytes);

    // Audio: load WAV, run isolate for 120+8.
    // NO trimming — the backend training features were computed on the FULL
    // recording (silence included), and trimming shifted frame statistics
    // (delta-MFCC mean r dropped 0.998→0.792 in tool/parity_check.dart).
    // Full 3 s costs only ~170 ms since the cepstrum fix, so parity wins.
    final audio = await AudioRecorderService.loadWavAsFloat64(audioWavPath);
    final acousticHhFuture = compute(acousticAndHhIsolate, audio);

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
    final srrResult = await viLiquidFuture;

    // Fusion DNN (TFLite, main isolate)
    final fusion = _fusion.run(
      acoustic: acousticHh.acoustic,
      visual: visualVec,
      haptic: hapticVec,
      hh: acousticHh.hh,
    );

    // Vi-Liquid late fusion: combine DNN probs with physical EI rule.
    // The LIVE local HH score (computed from the actual knock, not the
    // training-mean DNN input) gates the Hollow Heart trigger.
    final viLiquid = _mobileFusion.fuse(
      pImmature: fusion.pImmature,
      pRipe: fusion.pRipe,
      pOverripe: fusion.pOverripe,
      f2Hz: srrResult.f2Hz,
      massKg: massKg,
      acousticHhScore: acousticHh.localHhScore,
    );

    final acousticF2Hz = acousticHh.acoustic[111];
    final acousticF2Db = acousticHh.acoustic[112];
    final hhScore = acousticHh.localHhScore;
    final contactQuality = hapticVec[6];
    final signalRms = _rms(audio);
    final quality = DspUtils.knockQuality(audio);

    return MultimodalResult(
      fusion: fusion,
      viLiquid: viLiquid,
      f2Hz: acousticF2Hz,
      f2Db: acousticF2Db,
      hhScore: hhScore,
      contactQuality: contactQuality,
      signalRms: signalRms,
      recordingQuality: quality.quality,
      lowBandRatio: quality.lowBandRatio,
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
    _fusion.dispose();
    _loaded = false;
  }
}

/// Top-level wrapper so compute() can run SrrProcessor in isolate.
SrrResult _runSrrInIsolate(List<double> z) {
  return SrrProcessor().reconstruct(z);
}
