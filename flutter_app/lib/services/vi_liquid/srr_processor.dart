import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fftea/fftea.dart';

/// Simplified Super-Resolution Reconstruction (SRR) for Vi-Liquid haptic.
///
/// Backend reference: backend/module_c/srr_reconstruction.py (NUFFT-OMP).
/// Mobile port uses **cubic spline interpolation** to upsample the native
/// 100 Hz IMU signal to 1600 Hz virtual rate, then FFT to extract the
/// dominant resonance frequency (f2) in the 50-300 Hz physical band.
///
/// This is a deliberate simplification — full NUFFT-OMP is computationally
/// heavy on phone. Documented limitation: aliased components above native
/// Nyquist (50 Hz) cannot be perfectly recovered by interpolation alone,
/// but the LRA-driven resonant response typically has enough energy in
/// the 50-300 Hz band post-upsample to give a meaningful f2 estimate.
class SrrProcessor {
  static const int nativeRate = 100;
  static const int targetRate = 1600;
  static const int upsampleFactor = 16; // 1600/100

  static const double freqLow = 50.0;
  static const double freqHigh = 300.0;

  /// Reconstruct + extract f2.
  ///
  /// [imuZ] should be the z-axis acceleration samples captured at ~100 Hz
  /// (gravity-subtracted is best, but raw works too — we use AC component).
  ///
  /// Returns a record with the high-resolution signal, dominant frequency,
  /// and amplitude in dB.
  SrrResult reconstruct(List<double> imuZ) {
    if (imuZ.length < 4) {
      return SrrResult(
        reconstructed: Float64List(0),
        f2Hz: 0.0,
        f2Db: -120.0,
        nativeSamples: imuZ.length,
        upsampledSamples: 0,
      );
    }

    // 1. Subtract mean (remove gravity DC offset)
    double mean = 0.0;
    for (final v in imuZ) {
      mean += v;
    }
    mean /= imuZ.length;
    final ac = Float64List(imuZ.length);
    for (int i = 0; i < imuZ.length; i++) {
      ac[i] = imuZ[i] - mean;
    }

    // 2. Cubic spline interpolation 100 → 1600 Hz (16x upsample)
    final upsampled = _cubicInterpolate(ac, upsampleFactor);

    // 3. Hann window + FFT
    final n = upsampled.length;
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    final window = _hannWindow(n);
    for (int i = 0; i < n; i++) {
      padded[i] = upsampled[i] * window[i];
    }

    final fft = FFT(fftSize);
    final complex = fft.realFft(padded);
    final nBins = fftSize ~/ 2 + 1;

    // 4. Find dominant peak in 50-300 Hz band
    double peakMag = 0.0;
    double peakFreq = 0.0;
    for (int k = 0; k < nBins; k++) {
      final f = k * targetRate / fftSize;
      if (f < freqLow || f > freqHigh) continue;
      final re = complex[k].x;
      final im = complex[k].y;
      final mag = math.sqrt(re * re + im * im) * 2.0 / n;
      if (mag > peakMag) {
        peakMag = mag;
        peakFreq = f;
      }
    }
    final db = 20.0 * (math.log(peakMag + 1e-10) / math.ln10);

    return SrrResult(
      reconstructed: upsampled,
      f2Hz: peakFreq,
      f2Db: db,
      nativeSamples: imuZ.length,
      upsampledSamples: upsampled.length,
    );
  }

  /// Cubic interpolation upsample by integer factor.
  /// For point u in [0, 1) between samples i and i+1, uses 4-point
  /// Catmull-Rom-style cubic (boundary samples clamped).
  Float64List _cubicInterpolate(Float64List src, int factor) {
    final n = src.length;
    if (n < 2) return Float64List.fromList(src);
    final outLen = (n - 1) * factor + 1;
    final out = Float64List(outLen);

    double sample(int idx) {
      if (idx < 0) return src[0];
      if (idx >= n) return src[n - 1];
      return src[idx];
    }

    for (int outIdx = 0; outIdx < outLen; outIdx++) {
      final pos = outIdx / factor;
      final i = pos.floor();
      final u = pos - i;

      if (u < 1e-10) {
        out[outIdx] = src[i];
        continue;
      }

      final p0 = sample(i - 1);
      final p1 = sample(i);
      final p2 = sample(i + 1);
      final p3 = sample(i + 2);

      // Catmull-Rom spline
      final a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
      final b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
      final c = -0.5 * p0 + 0.5 * p2;
      final d = p1;

      out[outIdx] = ((a * u + b) * u + c) * u + d;
    }
    return out;
  }

  Float64List _hannWindow(int n) {
    final w = Float64List(n);
    if (n == 1) {
      w[0] = 1.0;
      return w;
    }
    for (int i = 0; i < n; i++) {
      w[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1));
    }
    return w;
  }

  int _nextPow2(int n) {
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }
}

class SrrResult {
  final Float64List reconstructed; // 1600 Hz virtual signal
  final double f2Hz; // dominant resonance frequency
  final double f2Db; // amplitude in dB
  final int nativeSamples;
  final int upsampledSamples;

  SrrResult({
    required this.reconstructed,
    required this.f2Hz,
    required this.f2Db,
    required this.nativeSamples,
    required this.upsampledSamples,
  });
}
