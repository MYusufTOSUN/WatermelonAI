import 'dart:async';

import 'package:sensors_plus/sensors_plus.dart';

/// Records 3-axis accelerometer (gravity-included) for a fixed duration.
///
/// Returns lists of x, y, z samples (m/s^2) plus the effective sample rate.
class SensorRecorderService {
  final List<double> _x = [];
  final List<double> _y = [];
  final List<double> _z = [];
  StreamSubscription<AccelerometerEvent>? _sub;
  DateTime? _startTime;
  DateTime? _stopTime;

  Future<void> start() async {
    _x.clear();
    _y.clear();
    _z.clear();
    _startTime = DateTime.now();
    _stopTime = null;
    _sub = accelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 10),
    ).listen((e) {
      _x.add(e.x);
      _y.add(e.y);
      _z.add(e.z);
    });
  }

  Future<SensorRecording> stop() async {
    await _sub?.cancel();
    _sub = null;
    _stopTime = DateTime.now();
    // Guard against stop() being called without a prior start() (would
    // otherwise throw on the null _startTime). Fall back to nominal 100 Hz.
    final durationMs = _startTime != null
        ? _stopTime!.difference(_startTime!).inMilliseconds
        : 0;
    final n = _x.length;
    final sampleRate =
        durationMs > 0 ? (n * 1000.0) / durationMs : 100.0;
    return SensorRecording(
      x: List.of(_x),
      y: List.of(_y),
      z: List.of(_z),
      sampleRate: sampleRate,
    );
  }

  Future<void> cancel() async {
    await _sub?.cancel();
    _sub = null;
  }

  bool get isRecording => _sub != null;
  int get samplesCollected => _x.length;
}

class SensorRecording {
  final List<double> x;
  final List<double> y;
  final List<double> z;
  final double sampleRate;
  SensorRecording({
    required this.x,
    required this.y,
    required this.z,
    required this.sampleRate,
  });
}
