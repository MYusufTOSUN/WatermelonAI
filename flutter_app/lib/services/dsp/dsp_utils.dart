import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fftea/fftea.dart';

/// DSP utilities — Dart-side signal processing primitives.
///
/// Implements: Hann window, framing, FFT magnitude spectrum, mel filterbank,
/// DCT-II, librosa-compatible MFCC.
///
/// Backend reference: backend/module_d/feature_extractor.py
///   sample_rate=44100, n_fft=2048, hop_length=512, n_mfcc=13
class DspUtils {
  static const int audioSampleRate = 44100;
  static const int nFft = 2048;
  static const int hopLength = 512;
  static const int nMfcc = 13;
  // MUST stay 128: the fusion DNN was trained on backend features computed
  // with librosa's default n_mels=128. Using 64 shifts every MFCC value and
  // the model misclassifies (v13 regression — real watermelons read as
  // "Olgunlaşmamış"). Speed comes from sparse filterbank bounds instead.
  static const int nMels = 128;
  static const double freqBandLow = 50.0;
  static const double freqBandHigh = 500.0;

  /// Hann window of length n. Same as numpy.hanning.
  static Float64List hannWindow(int n) {
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

  /// Frame the signal into overlapping windows.
  /// Returns matrix [n_frames][n_fft] (Hann-windowed).
  static List<Float64List> frame(
    Float64List signal, {
    int frameLength = nFft,
    int hop = hopLength,
    bool center = true,
  }) {
    Float64List src = signal;
    if (center) {
      // Pad with reflection on both sides like librosa default
      final pad = frameLength ~/ 2;
      src = Float64List(signal.length + 2 * pad);
      for (int i = 0; i < pad; i++) {
        final mirror = pad - i;
        src[i] = mirror < signal.length ? signal[mirror] : 0.0;
      }
      for (int i = 0; i < signal.length; i++) {
        src[pad + i] = signal[i];
      }
      for (int i = 0; i < pad; i++) {
        final back = signal.length - 2 - i;
        src[pad + signal.length + i] =
            back >= 0 && back < signal.length ? signal[back] : 0.0;
      }
    }

    if (src.length < frameLength) {
      return [_zeroPadHann(src, frameLength)];
    }

    final nFrames = 1 + (src.length - frameLength) ~/ hop;
    final window = hannWindow(frameLength);
    final frames = <Float64List>[];
    for (int t = 0; t < nFrames; t++) {
      final start = t * hop;
      final f = Float64List(frameLength);
      for (int i = 0; i < frameLength; i++) {
        f[i] = src[start + i] * window[i];
      }
      frames.add(f);
    }
    return frames;
  }

  static Float64List _zeroPadHann(Float64List s, int n) {
    final window = hannWindow(n);
    final f = Float64List(n);
    final m = math.min(s.length, n);
    for (int i = 0; i < m; i++) {
      f[i] = s[i] * window[i];
    }
    return f;
  }

  /// Compute magnitude spectrogram. Returns [n_frames][n_fft/2+1] of |FFT|.
  static List<Float64List> magnitudeSpectrogram(List<Float64List> frames) {
    final fftSize = frames.isEmpty ? nFft : frames[0].length;
    final fft = FFT(fftSize);
    final nBins = fftSize ~/ 2 + 1;
    final result = <Float64List>[];
    for (final frame in frames) {
      // Convert to complex Float64x2List then FFT
      final complex = fft.realFft(frame);
      final mag = Float64List(nBins);
      for (int k = 0; k < nBins; k++) {
        final re = complex[k].x;
        final im = complex[k].y;
        mag[k] = math.sqrt(re * re + im * im);
      }
      result.add(mag);
    }
    return result;
  }

  /// Power spectrogram = |FFT|^2.
  static List<Float64List> powerSpectrogram(List<Float64List> frames) {
    final mag = magnitudeSpectrogram(frames);
    for (final m in mag) {
      for (int i = 0; i < m.length; i++) {
        m[i] = m[i] * m[i];
      }
    }
    return mag;
  }

  /// Convert pre-computed magnitude spectrogram to power (mag squared).
  /// Allocates a new list — does NOT mutate input.
  static List<Float64List> magToPower(List<Float64List> magSpec) {
    final out = List<Float64List>.generate(
      magSpec.length,
      (i) {
        final src = magSpec[i];
        final dst = Float64List(src.length);
        for (int k = 0; k < src.length; k++) {
          dst[k] = src[k] * src[k];
        }
        return dst;
      },
    );
    return out;
  }

  /// MFCC from a pre-computed POWER spectrogram (avoids redundant FFT).
  ///
  /// Mel filters are triangular and non-zero over only a narrow bin range;
  /// we precompute each filter's [start, end) support and skip the zeros.
  /// Numerically identical to the dense loop, ~10-30x fewer multiply-adds.
  static List<Float64List> mfccFromPower(
    List<Float64List> powerSpec, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    int nMelsLocal = nMels,
    int nMfccLocal = nMfcc,
  }) {
    final fb = melFilterbank(
      sr: sr,
      nFftLocal: nFftLocal,
      nMelsLocal: nMelsLocal,
    );
    // Sparse support bounds per filter
    final startBin = List<int>.filled(nMelsLocal, 0);
    final endBin = List<int>.filled(nMelsLocal, 0);
    for (int m = 0; m < nMelsLocal; m++) {
      final filter = fb[m];
      int s = 0;
      while (s < filter.length && filter[s] == 0.0) {
        s++;
      }
      int e = filter.length;
      while (e > s && filter[e - 1] == 0.0) {
        e--;
      }
      startBin[m] = s;
      endBin[m] = e;
    }
    final nFrames = powerSpec.length;
    final melSpec = List<Float64List>.generate(
      nMelsLocal,
      (_) => Float64List(nFrames),
    );
    for (int m = 0; m < nMelsLocal; m++) {
      final filter = fb[m];
      final s = startBin[m];
      final e = endBin[m];
      for (int t = 0; t < nFrames; t++) {
        double sum = 0.0;
        final spec = powerSpec[t];
        for (int k = s; k < e; k++) {
          sum += filter[k] * spec[k];
        }
        melSpec[m][t] = sum;
      }
    }

    // power_to_db(ref=max)
    double globalMax = 1e-10;
    for (final row in melSpec) {
      for (final v in row) {
        if (v > globalMax) globalMax = v;
      }
    }
    final db = List<Float64List>.generate(
      melSpec.length,
      (_) => Float64List(melSpec.isEmpty ? 0 : melSpec[0].length),
    );
    for (int m = 0; m < melSpec.length; m++) {
      for (int t = 0; t < melSpec[m].length; t++) {
        final ratio = melSpec[m][t] / globalMax;
        db[m][t] = 10.0 *
            (math.log(ratio.clamp(1e-10, double.infinity)) / math.ln10);
        if (db[m][t] < -80.0) db[m][t] = -80.0;
      }
    }

    return dct(db, nMfccLocal);
  }

  /// Hz to mel (librosa "htk"=False, default Slaney-style).
  static double hzToMel(double hz) {
    const double fMin = 0.0;
    const double fSp = 200.0 / 3.0;
    double mel = (hz - fMin) / fSp;
    const double minLogHz = 1000.0;
    const double minLogMel = (minLogHz - fMin) / fSp;
    final double logstep = math.log(6.4) / 27.0;
    if (hz >= minLogHz) {
      mel = minLogMel + math.log(hz / minLogHz) / logstep;
    }
    return mel;
  }

  static double melToHz(double mel) {
    const double fMin = 0.0;
    const double fSp = 200.0 / 3.0;
    double hz = fMin + fSp * mel;
    const double minLogHz = 1000.0;
    const double minLogMel = (minLogHz - fMin) / fSp;
    final double logstep = math.log(6.4) / 27.0;
    if (mel >= minLogMel) {
      hz = minLogHz * math.exp(logstep * (mel - minLogMel));
    }
    return hz;
  }

  /// Build mel filterbank (n_mels, n_fft/2+1) — librosa-compatible Slaney norm.
  static List<Float64List> melFilterbank({
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    int nMelsLocal = nMels,
    double fMin = 0.0,
    double? fMax,
  }) {
    final double fmax = fMax ?? sr / 2.0;
    final int nBins = nFftLocal ~/ 2 + 1;

    // FFT frequencies
    final fftFreqs = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      fftFreqs[k] = k * sr / nFftLocal;
    }

    // Mel points
    final melMin = hzToMel(fMin);
    final melMax = hzToMel(fmax);
    final melPoints = Float64List(nMelsLocal + 2);
    for (int i = 0; i < nMelsLocal + 2; i++) {
      melPoints[i] = melMin + (melMax - melMin) * i / (nMelsLocal + 1);
    }
    final hzPoints = Float64List(nMelsLocal + 2);
    for (int i = 0; i < hzPoints.length; i++) {
      hzPoints[i] = melToHz(melPoints[i]);
    }

    // Build triangular filters with Slaney norm
    final fb = List<Float64List>.generate(
      nMelsLocal,
      (_) => Float64List(nBins),
    );
    for (int m = 0; m < nMelsLocal; m++) {
      final fLeft = hzPoints[m];
      final fCenter = hzPoints[m + 1];
      final fRight = hzPoints[m + 2];
      final enorm = 2.0 / (fRight - fLeft);
      for (int k = 0; k < nBins; k++) {
        final f = fftFreqs[k];
        double weight = 0.0;
        if (f >= fLeft && f <= fCenter && fCenter > fLeft) {
          weight = (f - fLeft) / (fCenter - fLeft);
        } else if (f > fCenter && f <= fRight && fRight > fCenter) {
          weight = (fRight - f) / (fRight - fCenter);
        }
        fb[m][k] = weight * enorm;
      }
    }
    return fb;
  }

  /// Mel-spectrogram: melFilterbank @ powerSpectrogram → (n_mels, n_frames)
  static List<Float64List> melSpectrogram(
    Float64List audio, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    int hopLocal = hopLength,
    int nMelsLocal = nMels,
  }) {
    final frames = frame(audio, frameLength: nFftLocal, hop: hopLocal);
    final powerSpec = powerSpectrogram(frames);
    final fb = melFilterbank(
      sr: sr,
      nFftLocal: nFftLocal,
      nMelsLocal: nMelsLocal,
    );
    final nFrames = powerSpec.length;
    final melSpec = List<Float64List>.generate(
      nMelsLocal,
      (_) => Float64List(nFrames),
    );
    for (int m = 0; m < nMelsLocal; m++) {
      for (int t = 0; t < nFrames; t++) {
        double sum = 0.0;
        final filter = fb[m];
        final spec = powerSpec[t];
        for (int k = 0; k < filter.length; k++) {
          sum += filter[k] * spec[k];
        }
        melSpec[m][t] = sum;
      }
    }
    return melSpec;
  }

  /// DCT type-II ortho-normalized, librosa default for MFCC.
  /// melLogSpec: (n_mels, n_frames) → (n_mfcc, n_frames)
  static List<Float64List> dct(
    List<Float64List> melLogSpec,
    int outDim,
  ) {
    final n = melLogSpec.length;
    final t = melLogSpec.isEmpty ? 0 : melLogSpec[0].length;
    final out =
        List<Float64List>.generate(outDim, (_) => Float64List(t));

    // Precompute basis: N(m, k) = cos(pi/N * (n + 0.5) * k) for n in [0,N)
    for (int k = 0; k < outDim; k++) {
      final norm = (k == 0) ? math.sqrt(1.0 / n) : math.sqrt(2.0 / n);
      for (int frame = 0; frame < t; frame++) {
        double sum = 0.0;
        for (int nn = 0; nn < n; nn++) {
          sum += melLogSpec[nn][frame] *
              math.cos(math.pi / n * (nn + 0.5) * k);
        }
        out[k][frame] = sum * norm;
      }
    }
    return out;
  }

  /// MFCC pipeline: audio → mel → log → DCT.
  /// librosa.feature.mfcc default uses power_to_db on mel spectrogram.
  static List<Float64List> mfcc(
    Float64List audio, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    int hopLocal = hopLength,
    int nMelsLocal = nMels,
    int nMfccLocal = nMfcc,
  }) {
    final melSpec = melSpectrogram(
      audio,
      sr: sr,
      nFftLocal: nFftLocal,
      hopLocal: hopLocal,
      nMelsLocal: nMelsLocal,
    );

    // power_to_db(ref=max): 10 * log10(S / max(S))
    double globalMax = 1e-10;
    for (final row in melSpec) {
      for (final v in row) {
        if (v > globalMax) globalMax = v;
      }
    }
    final db = List<Float64List>.generate(
      melSpec.length,
      (_) => Float64List(melSpec.isEmpty ? 0 : melSpec[0].length),
    );
    for (int m = 0; m < melSpec.length; m++) {
      for (int t = 0; t < melSpec[m].length; t++) {
        final ratio = melSpec[m][t] / globalMax;
        db[m][t] = 10.0 * (math.log(ratio.clamp(1e-10, double.infinity)) /
            math.ln10);
        // librosa top_db clipping (default 80)
        if (db[m][t] < -80.0) db[m][t] = -80.0;
      }
    }

    return dct(db, nMfccLocal);
  }

  /// Delta of MFCC (first-order derivative, librosa default width=9).
  static List<Float64List> delta(
    List<Float64List> data, {
    int width = 9,
  }) {
    int w = width;
    final t = data.isEmpty ? 0 : data[0].length;
    if (w > t) w = t;
    if (w % 2 == 0) w -= 1;
    if (w < 3) w = 3;
    final half = w ~/ 2;

    double denom = 0.0;
    for (int i = 1; i <= half; i++) {
      denom += 2 * i * i;
    }
    if (denom == 0) denom = 1.0;

    final out = List<Float64List>.generate(
      data.length,
      (_) => Float64List(t),
    );
    for (int m = 0; m < data.length; m++) {
      for (int n = 0; n < t; n++) {
        double sum = 0.0;
        for (int k = 1; k <= half; k++) {
          final iPlus = math.min(n + k, t - 1);
          final iMinus = math.max(n - k, 0);
          sum += k * (data[m][iPlus] - data[m][iMinus]);
        }
        out[m][n] = sum / denom;
      }
    }
    return out;
  }

  /// Aggregate stats: returns Float64List of length 4*nMfcc (mean/std/min/max).
  static Float64List statsMfcc4(List<Float64List> mfccMatrix) {
    final m = mfccMatrix.length;
    final out = Float64List(4 * m);
    for (int i = 0; i < m; i++) {
      final row = mfccMatrix[i];
      if (row.isEmpty) {
        out[i] = 0.0;
        out[m + i] = 0.0;
        out[2 * m + i] = 0.0;
        out[3 * m + i] = 0.0;
        continue;
      }
      double sum = 0.0;
      double mn = double.infinity;
      double mx = -double.infinity;
      for (final v in row) {
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      final mean = sum / row.length;
      double varSum = 0.0;
      for (final v in row) {
        final d = v - mean;
        varSum += d * d;
      }
      final std = math.sqrt(varSum / row.length);
      out[i] = mean;
      out[m + i] = std;
      out[2 * m + i] = mn;
      out[3 * m + i] = mx;
    }
    return out;
  }

  /// Aggregate stats: returns Float64List of length 2*nMfcc (mean/std).
  static Float64List statsMfcc2(List<Float64List> mfccMatrix) {
    final m = mfccMatrix.length;
    final out = Float64List(2 * m);
    for (int i = 0; i < m; i++) {
      final row = mfccMatrix[i];
      if (row.isEmpty) {
        out[i] = 0.0;
        out[m + i] = 0.0;
        continue;
      }
      double sum = 0.0;
      for (final v in row) {
        sum += v;
      }
      final mean = sum / row.length;
      double varSum = 0.0;
      for (final v in row) {
        final d = v - mean;
        varSum += d * d;
      }
      out[i] = mean;
      out[m + i] = math.sqrt(varSum / row.length);
    }
    return out;
  }

  /// Zero-crossing rate per frame, then return mean & std.
  static List<double> zcrStats(
    Float64List audio, {
    int frameLength = nFft,
    int hop = hopLength,
  }) {
    if (audio.length < frameLength) {
      return [0.0, 0.0];
    }
    final nFrames = 1 + (audio.length - frameLength) ~/ hop;
    final zcrs = Float64List(nFrames);
    for (int t = 0; t < nFrames; t++) {
      final start = t * hop;
      int crossings = 0;
      for (int i = 1; i < frameLength; i++) {
        final a = audio[start + i - 1];
        final b = audio[start + i];
        if ((a >= 0 && b < 0) || (a < 0 && b >= 0)) crossings++;
      }
      zcrs[t] = crossings / frameLength;
    }
    double sum = 0.0;
    for (final v in zcrs) {
      sum += v;
    }
    final mean = sum / zcrs.length;
    double varSum = 0.0;
    for (final v in zcrs) {
      final d = v - mean;
      varSum += d * d;
    }
    final std = math.sqrt(varSum / zcrs.length);
    return [mean, std];
  }

  /// RMS per frame, returns Float64List.
  static Float64List rmsPerFrame(
    Float64List audio, {
    int frameLength = nFft,
    int hop = hopLength,
  }) {
    if (audio.length < frameLength) {
      return Float64List.fromList([_rmsOf(audio)]);
    }
    final nFrames = 1 + (audio.length - frameLength) ~/ hop;
    final out = Float64List(nFrames);
    for (int t = 0; t < nFrames; t++) {
      final start = t * hop;
      double sum = 0.0;
      for (int i = 0; i < frameLength; i++) {
        final v = audio[start + i];
        sum += v * v;
      }
      out[t] = math.sqrt(sum / frameLength);
    }
    return out;
  }

  static double _rmsOf(Float64List a) {
    if (a.isEmpty) return 0.0;
    double s = 0.0;
    for (final v in a) {
      s += v * v;
    }
    return math.sqrt(s / a.length);
  }

  /// Spectral centroid per frame.
  static Float64List spectralCentroidPerFrame(
    List<Float64List> magSpec, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
  }) {
    final nBins = nFftLocal ~/ 2 + 1;
    final freqs = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      freqs[k] = k * sr / nFftLocal;
    }
    final out = Float64List(magSpec.length);
    for (int t = 0; t < magSpec.length; t++) {
      final spec = magSpec[t];
      double num = 0.0;
      double den = 0.0;
      for (int k = 0; k < spec.length; k++) {
        num += freqs[k] * spec[k];
        den += spec[k];
      }
      out[t] = (den > 1e-10) ? num / den : 0.0;
    }
    return out;
  }

  /// Spectral bandwidth (p=2).
  static Float64List spectralBandwidthPerFrame(
    List<Float64List> magSpec,
    Float64List centroid, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
  }) {
    final nBins = nFftLocal ~/ 2 + 1;
    final freqs = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      freqs[k] = k * sr / nFftLocal;
    }
    final out = Float64List(magSpec.length);
    for (int t = 0; t < magSpec.length; t++) {
      final spec = magSpec[t];
      double num = 0.0;
      double den = 0.0;
      for (int k = 0; k < spec.length; k++) {
        final d = freqs[k] - centroid[t];
        num += d * d * spec[k];
        den += spec[k];
      }
      out[t] = (den > 1e-10) ? math.sqrt(num / den) : 0.0;
    }
    return out;
  }

  /// Spectral rolloff at roll_percent=0.85 (librosa default).
  static Float64List spectralRolloffPerFrame(
    List<Float64List> magSpec, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    double rollPercent = 0.85,
  }) {
    final nBins = nFftLocal ~/ 2 + 1;
    final freqs = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      freqs[k] = k * sr / nFftLocal;
    }
    final out = Float64List(magSpec.length);
    for (int t = 0; t < magSpec.length; t++) {
      final spec = magSpec[t];
      double total = 0.0;
      for (final v in spec) {
        total += v;
      }
      final thresh = total * rollPercent;
      double cum = 0.0;
      int idx = spec.length - 1;
      for (int k = 0; k < spec.length; k++) {
        cum += spec[k];
        if (cum >= thresh) {
          idx = k;
          break;
        }
      }
      out[t] = freqs[idx];
    }
    return out;
  }

  /// Spectral flatness per frame (geometric mean / arithmetic mean).
  static Float64List spectralFlatnessPerFrame(List<Float64List> magSpec) {
    final out = Float64List(magSpec.length);
    for (int t = 0; t < magSpec.length; t++) {
      final spec = magSpec[t];
      double logSum = 0.0;
      double arithSum = 0.0;
      int count = 0;
      for (final v in spec) {
        if (v > 1e-10) {
          logSum += math.log(v);
          arithSum += v;
          count++;
        }
      }
      if (count == 0) {
        out[t] = 0.0;
        continue;
      }
      final geom = math.exp(logSum / count);
      final arith = arithSum / count;
      out[t] = (arith > 1e-10) ? geom / arith : 0.0;
    }
    return out;
  }

  /// Spectral contrast simplified: 7 bands. Returns mean per band over time.
  static Float64List spectralContrastBandsMean(
    List<Float64List> magSpec, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
    int nBands = 7,
  }) {
    final nBins = nFftLocal ~/ 2 + 1;
    final freqs = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      freqs[k] = k * sr / nFftLocal;
    }
    // Log-spaced band edges from ~200 Hz to Nyquist
    final bandEdges = <double>[];
    final fStart = 200.0;
    final fEnd = sr / 2.0;
    final logStart = math.log(fStart) / math.ln10;
    final logEnd = math.log(fEnd) / math.ln10;
    for (int i = 0; i <= nBands; i++) {
      bandEdges.add(math.pow(10.0, logStart + (logEnd - logStart) * i / nBands)
          .toDouble());
    }

    final out = Float64List(nBands);
    final nFrames = magSpec.length;
    for (int b = 0; b < nBands; b++) {
      double accum = 0.0;
      int frameCount = 0;
      for (int t = 0; t < nFrames; t++) {
        final spec = magSpec[t];
        // Collect bins in band
        final vals = <double>[];
        for (int k = 0; k < spec.length; k++) {
          if (freqs[k] >= bandEdges[b] && freqs[k] < bandEdges[b + 1]) {
            vals.add(spec[k]);
          }
        }
        if (vals.length < 2) continue;
        vals.sort();
        final alpha = math.max(1, vals.length ~/ 5);
        // top alpha = peak, bottom alpha = valley
        double peakSum = 0.0;
        double valleySum = 0.0;
        for (int i = 0; i < alpha; i++) {
          peakSum += vals[vals.length - 1 - i];
          valleySum += vals[i];
        }
        final peakMean = peakSum / alpha;
        final valleyMean = valleySum / alpha;
        // dB contrast
        final contrast = 20.0 *
            (math.log((peakMean + 1e-10) / (valleyMean + 1e-10)) /
                math.ln10);
        accum += contrast;
        frameCount++;
      }
      out[b] = frameCount > 0 ? accum / frameCount : 0.0;
    }
    return out;
  }

  /// Chroma STFT: 12 pitch classes, mean over time.
  static Float64List chromaMean12(
    List<Float64List> magSpec, {
    int sr = audioSampleRate,
    int nFftLocal = nFft,
  }) {
    final nBins = nFftLocal ~/ 2 + 1;
    final out = Float64List(12);
    if (magSpec.isEmpty) return out;
    final accum = Float64List(12);
    int totalFrames = 0;

    for (int t = 0; t < magSpec.length; t++) {
      final spec = magSpec[t];
      final frameChroma = Float64List(12);
      for (int k = 1; k < nBins; k++) {
        final f = k * sr / nFftLocal;
        if (f < 27.5 || f > sr / 2) continue;
        final midi = 69.0 + 12.0 * (math.log(f / 440.0) / math.ln2);
        if (midi.isNaN || midi.isInfinite) continue;
        final pc = (midi.round() % 12 + 12) % 12;
        frameChroma[pc] += spec[k];
      }
      // Normalize frame
      double maxV = 0.0;
      for (final v in frameChroma) {
        if (v > maxV) maxV = v;
      }
      if (maxV > 1e-10) {
        for (int i = 0; i < 12; i++) {
          accum[i] += frameChroma[i] / maxV;
        }
        totalFrames++;
      }
    }
    if (totalFrames == 0) return out;
    for (int i = 0; i < 12; i++) {
      out[i] = accum[i] / totalFrames;
    }
    return out;
  }

  /// Compute full-audio FFT magnitudes ONCE so that downstream metrics
  /// (dominant frequency, spectral entropy, HH dual-peak) can reuse them
  /// without redoing the FFT 3 separate times.
  ///
  /// Returns ({mags: |FFT|, fftSize: padded length}).
  static ({Float64List mags, int fftSize}) computeFullAudioMag(
    Float64List audio,
  ) {
    final n = audio.length;
    if (n < 4) {
      return (mags: Float64List(0), fftSize: 0);
    }
    final window = hannWindow(n);
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    for (int i = 0; i < n; i++) {
      padded[i] = audio[i] * window[i];
    }
    final fft = FFT(fftSize);
    final complex = fft.realFft(padded);
    final nBins = fftSize ~/ 2 + 1;
    final mags = Float64List(nBins);
    for (int k = 0; k < nBins; k++) {
      final re = complex[k].x;
      final im = complex[k].y;
      mags[k] = math.sqrt(re * re + im * im);
    }
    return (mags: mags, fftSize: fftSize);
  }

  /// Find dominant frequency from precomputed magnitudes.
  static List<double> dominantFreqFromMag(
    Float64List mags,
    int fftSize, {
    int sr = audioSampleRate,
    double low = freqBandLow,
    double high = freqBandHigh,
    int? audioLength,
  }) {
    if (mags.isEmpty || fftSize == 0) return [0.0, -120.0];
    final n = audioLength ?? fftSize;
    double peakMag = 0.0;
    double peakFreq = 0.0;
    for (int k = 0; k < mags.length; k++) {
      final f = k * sr / fftSize;
      if (f < low || f > high) continue;
      // Match original scaling: |FFT|*2/N
      final mag = mags[k] * 2.0 / n;
      if (mag > peakMag) {
        peakMag = mag;
        peakFreq = f;
      }
    }
    final db = 20.0 * (math.log(peakMag + 1e-10) / math.ln10);
    return [peakFreq, db];
  }

  /// Spectral entropy from precomputed magnitudes (Shannon, normalized).
  static double spectralEntropyFromMag(Float64List mags) {
    if (mags.isEmpty) return 0.0;
    double total = 0.0;
    for (int k = 0; k < mags.length; k++) {
      total += mags[k] * mags[k];
    }
    if (total <= 1e-10) return 0.0;
    double entropy = 0.0;
    for (int k = 0; k < mags.length; k++) {
      final p = (mags[k] * mags[k]) / total;
      if (p > 1e-10) {
        entropy -= p * (math.log(p) / math.ln2);
      }
    }
    final maxEntropy = math.log(mags.length.toDouble()) / math.ln2;
    return maxEntropy > 0 ? entropy / maxEntropy : 0.0;
  }

  /// Find dominant frequency in band (returns [f, magnitudeDb]).
  static List<double> dominantFrequency(
    Float64List audio, {
    int sr = audioSampleRate,
    double low = freqBandLow,
    double high = freqBandHigh,
  }) {
    final n = audio.length;
    if (n < 4) return [0.0, -120.0];
    final window = hannWindow(n);
    final windowed = Float64List(n);
    for (int i = 0; i < n; i++) {
      windowed[i] = audio[i] * window[i];
    }
    // Use FFT of nearest power-of-two
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    for (int i = 0; i < n; i++) {
      padded[i] = windowed[i];
    }
    final fft = FFT(fftSize);
    final complex = fft.realFft(padded);
    final nBins = fftSize ~/ 2 + 1;
    double peakMag = 0.0;
    double peakFreq = 0.0;
    for (int k = 0; k < nBins; k++) {
      final f = k * sr / fftSize;
      if (f < low || f > high) continue;
      final re = complex[k].x;
      final im = complex[k].y;
      final mag = math.sqrt(re * re + im * im) * 2.0 / n;
      if (mag > peakMag) {
        peakMag = mag;
        peakFreq = f;
      }
    }
    final db = 20.0 * (math.log(peakMag + 1e-10) / math.ln10);
    return [peakFreq, db];
  }

  static int _nextPow2(int n) {
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }

  /// Spectral entropy: Shannon entropy of normalized PSD.
  static double spectralEntropyNorm(Float64List audio) {
    final n = audio.length;
    if (n < 4) return 0.0;
    final window = hannWindow(n);
    final fftSize = _nextPow2(n);
    final padded = Float64List(fftSize);
    for (int i = 0; i < n; i++) {
      padded[i] = audio[i] * window[i];
    }
    final fft = FFT(fftSize);
    final complex = fft.realFft(padded);
    final nBins = fftSize ~/ 2 + 1;
    final psd = Float64List(nBins);
    double total = 0.0;
    for (int k = 0; k < nBins; k++) {
      final re = complex[k].x;
      final im = complex[k].y;
      psd[k] = re * re + im * im;
      total += psd[k];
    }
    if (total <= 1e-10) return 0.0;
    double entropy = 0.0;
    for (int k = 0; k < nBins; k++) {
      final p = psd[k] / total;
      if (p > 1e-10) {
        entropy -= p * (math.log(p) / math.ln2);
      }
    }
    final maxEntropy = math.log(nBins.toDouble()) / math.ln2;
    return maxEntropy > 0 ? entropy / maxEntropy : 0.0;
  }
}
