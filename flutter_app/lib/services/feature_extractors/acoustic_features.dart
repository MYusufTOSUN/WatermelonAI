import 'dart:math' as math;
import 'dart:typed_data';

import '../dsp/dsp_utils.dart';

/// 120-D acoustic feature vector (Koc & Akbalik 2025).
///
/// Layout matches backend/module_d/feature_extractor.py:
///   [0:52]   MFCC stats (mean/std/min/max × 13)
///   [52:78]  Delta MFCC stats (mean/std × 13)
///   [78:80]  ZCR (mean, std)
///   [80:95]  Spectral features (centroid/bw/rolloff/flatness/contrast)
///   [95:99]  Energy (rms_mean, rms_std, log_energy_mean, log_energy_std)
///   [99:111] Chroma (12 pitch classes mean)
///   [111:114] Frequency (f2, f2_db, entropy)
///   [114:120] Time-domain (peak/crest/centroid/attack/decay/duration)
class AcousticFeatureExtractor {
  static const int sampleRate = DspUtils.audioSampleRate;
  static const int featureDim = 120;

  /// Extracts 120-D feature vector from PCM float audio in [-1, 1].
  Float64List extract(Float64List audio) {
    final out = Float64List(featureDim);
    int offset = 0;

    // SHARED COMPUTATION: frames + magSpec ONCE — every spectral feature
    // below derives from these instead of redoing FFTs.
    final frames = DspUtils.frame(audio);
    final magSpec = DspUtils.magnitudeSpectrogram(frames);
    final powerSpec = DspUtils.magToPower(magSpec);

    // 1. MFCC stats (52) — from pre-computed power spec
    final mfccMatrix = DspUtils.mfccFromPower(powerSpec);
    final mfccStats = DspUtils.statsMfcc4(mfccMatrix);
    for (int i = 0; i < mfccStats.length; i++) {
      out[offset + i] = mfccStats[i];
    }
    offset += mfccStats.length;

    // 2. Delta MFCC stats (26)
    final deltaMfcc = DspUtils.delta(mfccMatrix);
    final deltaStats = DspUtils.statsMfcc2(deltaMfcc);
    for (int i = 0; i < deltaStats.length; i++) {
      out[offset + i] = deltaStats[i];
    }
    offset += deltaStats.length;

    // 3. ZCR (2)
    final zcr = DspUtils.zcrStats(audio);
    out[offset++] = zcr[0];
    out[offset++] = zcr[1];

    // 4. Spectral (15)
    final centroid = DspUtils.spectralCentroidPerFrame(magSpec);
    final bandwidth = DspUtils.spectralBandwidthPerFrame(magSpec, centroid);
    final rolloff = DspUtils.spectralRolloffPerFrame(magSpec);
    final flatness = DspUtils.spectralFlatnessPerFrame(magSpec);
    final contrastBands = DspUtils.spectralContrastBandsMean(magSpec);

    out[offset++] = _meanD(centroid);
    out[offset++] = _stdD(centroid);
    out[offset++] = _meanD(bandwidth);
    out[offset++] = _stdD(bandwidth);
    out[offset++] = _meanD(rolloff);
    out[offset++] = _stdD(rolloff);
    out[offset++] = _meanD(flatness);
    out[offset++] = _stdD(flatness);
    for (int i = 0; i < 7; i++) {
      out[offset++] = contrastBands[i];
    }

    // 5. Energy (4)
    final rms = DspUtils.rmsPerFrame(audio);
    out[offset++] = _meanD(rms);
    out[offset++] = _stdD(rms);
    final logE = Float64List(rms.length);
    for (int i = 0; i < rms.length; i++) {
      logE[i] = math.log(rms[i] + 1e-10);
    }
    out[offset++] = _meanD(logE);
    out[offset++] = _stdD(logE);

    // 6. Chroma (12)
    final chroma = DspUtils.chromaMean12(magSpec);
    for (int i = 0; i < 12; i++) {
      out[offset++] = chroma[i];
    }

    // 7. Frequency (3) — f2, f2_db, spectral_entropy
    // Shared full-audio FFT: compute mags ONCE, reuse for both metrics.
    final fullFft = DspUtils.computeFullAudioMag(audio);
    final f2 = DspUtils.dominantFreqFromMag(
      fullFft.mags,
      fullFft.fftSize,
      audioLength: audio.length,
    );
    out[offset++] = f2[0];
    out[offset++] = f2[1];
    out[offset++] = DspUtils.spectralEntropyFromMag(fullFft.mags);

    // 8. Time-domain (6)
    final td = _timeDomain(audio);
    for (int i = 0; i < 6; i++) {
      out[offset++] = td[i];
    }

    assert(offset == featureDim, "Acoustic feature dim mismatch: $offset");
    return out;
  }

  // ----------------------- helpers -----------------------

  static double _meanD(Float64List a) {
    if (a.isEmpty) return 0.0;
    double s = 0.0;
    for (final v in a) {
      s += v;
    }
    return s / a.length;
  }

  static double _stdD(Float64List a) {
    if (a.isEmpty) return 0.0;
    final m = _meanD(a);
    double s = 0.0;
    for (final v in a) {
      s += (v - m) * (v - m);
    }
    return math.sqrt(s / a.length);
  }

  static Float64List _timeDomain(Float64List audio) {
    final n = audio.length;
    if (n == 0) return Float64List(6);
    double peakAmp = 0.0;
    double rmsSum = 0.0;
    double energySum = 0.0;
    double timeWeighted = 0.0;
    final absBuf = Float64List(n);
    for (int i = 0; i < n; i++) {
      final a = audio[i].abs();
      absBuf[i] = a;
      if (a > peakAmp) peakAmp = a;
      rmsSum += audio[i] * audio[i];
      final e = audio[i] * audio[i];
      energySum += e;
      timeWeighted += (i / sampleRate) * e;
    }
    final rms = math.sqrt(rmsSum / n);
    final crest = peakAmp / (rms + 1e-10);
    final tempCentroid =
        energySum > 1e-10 ? timeWeighted / energySum : 0.0;

    // Attack: first index where abs >= 0.9 * peak
    final threshold90 = 0.9 * peakAmp;
    int attackSamples = 0;
    if (peakAmp > 1e-10) {
      for (int i = 0; i < n; i++) {
        if (absBuf[i] >= threshold90) {
          attackSamples = i;
          break;
        }
      }
    }
    final attackTime = attackSamples / sampleRate;

    // Decay rate
    int peakIdx = 0;
    for (int i = 0; i < n; i++) {
      if (absBuf[i] == peakAmp) {
        peakIdx = i;
        break;
      }
    }
    final threshold10 = 0.1 * peakAmp;
    double decayRate = 0.0;
    if (peakIdx < n - 1 && peakAmp > 1e-10) {
      int decaySamples = -1;
      for (int i = peakIdx; i < n; i++) {
        if (absBuf[i] <= threshold10) {
          decaySamples = i - peakIdx;
          break;
        }
      }
      if (decaySamples >= 0) {
        final decayTime = decaySamples / sampleRate;
        decayRate = 1.0 / (decayTime + 1e-10);
      } else {
        final remaining = (n - peakIdx) / sampleRate;
        decayRate = 1.0 / (remaining + 1e-10) * 0.1;
      }
    }

    // Duration norm (effective duration covering 95% energy)
    final cumulative = Float64List(n);
    double accum = 0.0;
    for (int i = 0; i < n; i++) {
      accum += audio[i] * audio[i];
      cumulative[i] = accum;
    }
    final total = (cumulative[n - 1] + 1e-10);
    int startIdx = 0;
    int endIdx = n - 1;
    final lower = 0.025 * total;
    final upper = 0.975 * total;
    for (int i = 0; i < n; i++) {
      if (cumulative[i] >= lower) {
        startIdx = i;
        break;
      }
    }
    for (int i = n - 1; i >= 0; i--) {
      if (cumulative[i] <= upper) {
        endIdx = i;
        break;
      }
    }
    final effectiveDur = (endIdx - startIdx) / sampleRate;
    final totalDur = n / sampleRate;
    final durationNorm = effectiveDur / (totalDur + 1e-10);

    return Float64List.fromList([
      peakAmp,
      crest,
      tempCentroid,
      attackTime,
      decayRate,
      durationNorm,
    ]);
  }
}
