import 'package:vibration/vibration.dart';

/// Wraps the `vibration` package to provide a continuous LRA vibration
/// pulse during haptic capture (Vi-Liquid methodology).
///
/// Android note: the device LRA motor frequency is not user-selectable —
/// we trigger the strongest available continuous vibration for the
/// requested duration. The reflected response in the IMU is what matters
/// for f2 extraction, not the input vibration frequency.
class VibrationService {
  bool? _hasVibrator;
  bool? _hasAmplitudeControl;

  Future<void> init() async {
    _hasVibrator = await Vibration.hasVibrator() ?? false;
    _hasAmplitudeControl =
        await Vibration.hasAmplitudeControl() ?? false;
  }

  bool get isAvailable => _hasVibrator ?? false;
  bool get hasAmplitudeControl => _hasAmplitudeControl ?? false;

  /// Start a continuous vibration for [durationMs] milliseconds.
  /// Best-effort — silently no-op on devices without vibrator.
  Future<void> startContinuous({int durationMs = 3000}) async {
    if (_hasVibrator == null) await init();
    if (!isAvailable) return;
    if (hasAmplitudeControl) {
      await Vibration.vibrate(duration: durationMs, amplitude: 255);
    } else {
      await Vibration.vibrate(duration: durationMs);
    }
  }

  Future<void> cancel() async {
    if (isAvailable) {
      await Vibration.cancel();
    }
  }
}
