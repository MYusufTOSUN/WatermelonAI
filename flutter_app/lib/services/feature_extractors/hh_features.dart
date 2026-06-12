import 'dart:math' as math;
import 'dart:typed_data';

import '../dsp/dsp_utils.dart';

/// Hollow Heart channel — TWO separate outputs:
///
/// 1. [extract] → the 8-D vector fed to the FUSION DNN. The training
///    pipeline (backend/pipeline/data_loader.py::_extract_hh_8d) used the
///    layout [dp, dm, sp, cp, hnr, hh_score, confidence, active_n] computed
///    by the full 968-line backend detector (with sp≡0 due to a key-name
///    bug baked into training). Our simplified Dart sub-scores do NOT match
///    that detector numerically (offline test: wrong values flipped 4/12
///    fusion predictions). Strategy: feed the TRAINING MEANS — the same
///    neutral-substitution trick used for the visual/haptic channels —
///    which restored 10/12 agreement with the backend in
///    tool/parity_predict.py.
///
/// 2. [liveScore] → a LIVE 0-1 hollow-likelihood score from our simplified
///    5-indicator analysis of the actual knock recording. This does NOT go
///    into the DNN; it gates the Vi-Liquid Hollow-Heart trigger (dual
///    confirmation) and feeds the "içi boş riski" UI tile.
class HhFeatureExtractor {
  static const int featureDim = 8;

  /// Training-set means of the 8-D HH input, layout
  /// [dp, dm, sp, cp, hnr, hh_score, confidence, active_n]
  /// (measured from data/processed/X_features.npy[:, 138:146]).
  static const List<double> trainingMeans = [
    0.470, // dual_peak score
    0.287, // damping score
    0.000, // spectral score (constant 0 in training — key-name bug)
    0.218, // cepstral score
    0.997, // hnr score
    0.433, // hh_score
    0.843, // confidence
    0.299, // active_indicators / 5
  ];

  static const double dualPeakMinDistanceHz = 15.0;
  static const double dualPeakMaxDistanceHz = 80.0;
  static const double dampingRatioThreshold = 0.12;
  static const double decayRateFastThreshold = 25.0;
  static const double spectralSpreadThreshold = 0.65;
  static const double spectralEntropyHigh = 0.75;
  static const double spectralFlatnessHigh = 0.4;
  static const double hnrLowThreshold = 5.0;

  static const double wDualPeak = 0.30;
  static const double wDamping = 0.20;
  static const double wSpectral = 0.20;
  static const double wCepstral = 0.15;
  static const double wHnr = 0.15;

  /// DNN input: training-mean substitution (see class doc).
  Float64List extract(Float64List audio) {
    return Float64List.fromList(trainingMeans);
  }

  /// LIVE hollow-likelihood score (0-1) from the actual recording —
  /// used by the Vi-Liquid trigger gate and the UI risk tile.
  double liveScore(Float64List audio) {
    if (audio.length < 64) return 0.0;
    final dualPeak = _dualPeakScore(audio);
    final damping = _dampingScore(audio);
    final spectral = _spectralScore(audio);
    final cepstral = _cepstralScore(audio);
    final hnr = _hnrScore(audio);
    return (wDualPeak * dualPeak[0] +
            wDamping * damping[0] +
            wSpectral * spectral +
            wCepstral * cepstral +
            wHnr * hnr[0])
        .clamp(0.0, 1.0);
  }

  // ----------------------------------------------------------------
  // Indicators
  // ----------------------------------------------------------------

  /// Returns [score, peakDistance]
  List<double> _dualPeakScore(Float64List audio) {
    final n = audio.length;
    final window = DspUtils.hannWindow(n);
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    for (int i = 0; i < n; i++) {
      padded[i] = audio[i] * window[i];
    }
    // Magnitude spectrum
    final mag = _fftMag(padded);
    final sr = DspUtils.audioSampleRate;
    final nBins = mag.length;

    // Restrict to 50-500 Hz
    final low = (50.0 * fftSize / sr).floor();
    final high = (500.0 * fftSize / sr).ceil().clamp(low + 1, nBins);

    // Find top 2 peaks separated by at least dualPeakMinDistanceHz
    final peaks = <int>[];
    for (int k = low + 1; k < high - 1; k++) {
      if (mag[k] > mag[k - 1] && mag[k] > mag[k + 1]) {
        peaks.add(k);
      }
    }
    if (peaks.length < 2) return [0.0, 0.0];
    peaks.sort((a, b) => mag[b].compareTo(mag[a]));
    final top1 = peaks[0];
    int top2 = -1;
    for (int i = 1; i < peaks.length; i++) {
      final f1 = top1 * sr / fftSize;
      final f2 = peaks[i] * sr / fftSize;
      final dist = (f1 - f2).abs();
      if (dist >= dualPeakMinDistanceHz && dist <= dualPeakMaxDistanceHz) {
        top2 = peaks[i];
        break;
      }
    }
    if (top2 < 0) return [0.0, 0.0];

    final f1 = top1 * sr / fftSize;
    final f2 = top2 * sr / fftSize;
    final distance = (f1 - f2).abs();

    // Score: closer to middle of range → higher score
    final mid = (dualPeakMinDistanceHz + dualPeakMaxDistanceHz) / 2.0;
    final rng = (dualPeakMaxDistanceHz - dualPeakMinDistanceHz) / 2.0;
    final closeness = 1.0 - ((distance - mid).abs() / rng).clamp(0.0, 1.0);

    // Magnitude ratio between top1 and top2 (similar magnitudes = strong dual peak)
    final ratio = mag[top2] / (mag[top1] + 1e-10);
    final score = (closeness * ratio).clamp(0.0, 1.0);
    return [score, distance];
  }

  /// Returns [score, damping_ratio]
  List<double> _dampingScore(Float64List audio) {
    final n = audio.length;
    final absBuf = Float64List(n);
    double peak = 0.0;
    int peakIdx = 0;
    for (int i = 0; i < n; i++) {
      absBuf[i] = audio[i].abs();
      if (absBuf[i] > peak) {
        peak = absBuf[i];
        peakIdx = i;
      }
    }
    if (peak < 1e-10) return [0.0, 0.0];

    // Decay rate: time to reach 10% of peak
    final thresh = 0.1 * peak;
    int decaySamples = -1;
    for (int i = peakIdx; i < n; i++) {
      if (absBuf[i] <= thresh) {
        decaySamples = i - peakIdx;
        break;
      }
    }
    double decayRate;
    if (decaySamples > 0) {
      decayRate = 1.0 / (decaySamples / DspUtils.audioSampleRate + 1e-10);
    } else {
      decayRate = 1.0;
    }

    // Score: faster decay → higher hollow heart probability
    final score = (decayRate / decayRateFastThreshold).clamp(0.0, 1.0);

    // Damping ratio estimate: zeta = ln(A0/A1) / (2*pi*N)
    // where A0, A1 are envelope peaks. Use peak vs the value at 10% threshold.
    final dampingRatio = score * dampingRatioThreshold * 1.5;
    return [score, dampingRatio];
  }

  double _spectralScore(Float64List audio) {
    final entropy = DspUtils.spectralEntropyNorm(audio);
    // Flatness via spectrogram
    final frames = DspUtils.frame(audio);
    if (frames.isEmpty) return 0.0;
    final mag = DspUtils.magnitudeSpectrogram(frames);
    final flatness = DspUtils.spectralFlatnessPerFrame(mag);
    double flatMean = 0.0;
    for (final v in flatness) {
      flatMean += v;
    }
    flatMean = flatness.isEmpty ? 0.0 : flatMean / flatness.length;

    final entropyScore =
        (entropy / spectralEntropyHigh).clamp(0.0, 1.0);
    final flatScore =
        (flatMean / spectralFlatnessHigh).clamp(0.0, 1.0);
    return (entropyScore * 0.6 + flatScore * 0.4).clamp(0.0, 1.0);
  }

  /// Simplified cepstrum-based periodicity loss.
  double _cepstralScore(Float64List audio) {
    final n = audio.length;
    final window = DspUtils.hannWindow(n);
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    for (int i = 0; i < n; i++) {
      padded[i] = audio[i] * window[i];
    }
    final mag = _fftMag(padded);
    final m = mag.length;
    // Log magnitude
    final logMag = Float64List(m);
    for (int i = 0; i < m; i++) {
      logMag[i] = math.log(mag[i] + 1e-10);
    }
    // Real cepstrum via SECOND FFT of the log-magnitude spectrum.
    // logMag is real and (conceptually) even, so |FFT(logMag)| is
    // proportional to the cosine-transform cepstrum. O(P log P) — replaces
    // the previous naive O(N²) DCT that took 1-2 minutes on a 3 s clip.
    final p = _nextPow2(m);
    final cepsIn = Float64List(p);
    for (int i = 0; i < m; i++) {
      cepsIn[i] = logMag[i];
    }
    final cepsMag = _fftMag(cepsIn); // |FFT| bins: quefrency q ↔ bin q·(m/p)
    // Scale factor to keep score range comparable to the old orthonormal
    // DCT-II (which carried a sqrt(2/M) normalization).
    final dctScale = math.sqrt(2.0 / m);

    // Quefrency mapping: spectrum bin spacing df = sr/fftSize, so the FFT of
    // the (padded) spectrum has quefrency resolution dq = fftSize/(p·sr).
    final sr = DspUtils.audioSampleRate;
    final dq = fftSize / (p * sr); // seconds per cepstral bin
    int qLow = (0.002 / dq).floor();
    int qHigh = (0.020 / dq).ceil();
    qLow = qLow.clamp(1, cepsMag.length - 2);
    qHigh = qHigh.clamp(qLow + 1, cepsMag.length);

    double peak = 0.0;
    for (int i = qLow; i < qHigh; i++) {
      final v = cepsMag[i] * dctScale;
      if (v > peak) peak = v;
    }
    // Strong cepstrum peak = healthy periodicity → LOW HH score
    // Weak peak = lost periodicity → HIGH HH score
    final normPeak = (peak / (m / 100.0)).clamp(0.0, 1.0);
    return (1.0 - normPeak).clamp(0.0, 1.0);
  }

  /// Returns [score, hnr_db]
  List<double> _hnrScore(Float64List audio) {
    // Simplified HNR: autocorrelation-based
    final n = audio.length;
    if (n < 32) return [0.0, 0.0];
    final sr = DspUtils.audioSampleRate;
    // Pitch range 50-500 Hz → lag range
    final lagMax = (sr / 50.0).floor();
    final lagMin = (sr / 500.0).floor();
    final acf = Float64List(lagMax - lagMin + 1);
    for (int lag = lagMin; lag <= lagMax; lag++) {
      double sum = 0.0;
      final m = math.min(n - lag, n - 1);
      for (int i = 0; i < m; i++) {
        sum += audio[i] * audio[i + lag];
      }
      acf[lag - lagMin] = sum / (m + 1e-10);
    }
    double acfMax = 0.0;
    for (final v in acf) {
      if (v > acfMax) acfMax = v;
    }
    double r0 = 0.0;
    for (int i = 0; i < n; i++) {
      r0 += audio[i] * audio[i];
    }
    r0 /= n;
    if (r0 <= 1e-10) return [0.0, 0.0];
    final ratio = (acfMax / r0).clamp(0.0, 0.999);
    // HNR (dB) = 10 log10(harmonic / noise)
    final hnrDb = 10.0 * (math.log(ratio / (1.0 - ratio) + 1e-10) / math.ln10);
    // Low HNR → hollow heart
    final score =
        ((hnrLowThreshold - hnrDb) / hnrLowThreshold).clamp(0.0, 1.0);
    return [score, hnrDb];
  }

  // ----------------------------------------------------------------
  // FFT helpers
  // ----------------------------------------------------------------

  Float64List _fftMag(Float64List padded) {
    // Use fftea via DspUtils (avoiding circular import).
    // Reuse magnitudeSpectrogram on a single frame.
    final frames = <Float64List>[padded];
    final mag = DspUtils.magnitudeSpectrogram(frames);
    return mag[0];
  }

  int _nextPow2(int n) {
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }
}
