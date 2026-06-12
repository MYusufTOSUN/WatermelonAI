// Offline parity check: runs the EXACT Dart DSP chain the phone uses and
// compares it against the Python (librosa) reference — without a device.
//
// Usage (from flutter_app/):
//   dart run tool/parity_check.dart
//
// Outputs:
//   1. Per-group Pearson r between Dart 120-D features and Python reference
//      on the bundled WAV (full audio — matches how the reference was made).
//   2. The same comparison after the LIVE pipeline's trimSilence, to
//      quantify how much trimming shifts features away from the training
//      distribution.
//   3. dart_features_all_samples.json — Dart-computed acoustic+HH features
//      for all 12 bundled Qilin WAVs, to be fed to the fusion TFLite in
//      Python (tool/parity_predict.py) so we can preview the phone's
//      predictions before any field test.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:wav/wav.dart';

import 'package:watermelon_ripeness/services/feature_extractors/acoustic_features.dart';
import 'package:watermelon_ripeness/services/feature_extractors/hh_features.dart';

// Verbatim copy of AudioRecorderService.trimSilence (that class imports
// Flutter plugins so it can't be imported in a CLI script).
Float64List trimSilence(
  Float64List audio, {
  double thresholdRms = 0.005,
}) {
  if (audio.length < 1024) return audio;
  const win = 512;
  const hop = 256;
  final nFrames = (audio.length - win) ~/ hop;
  int firstActive = 0;
  int lastActive = nFrames - 1;
  bool found = false;
  for (int i = 0; i < nFrames; i++) {
    double s = 0.0;
    for (int j = 0; j < win; j++) {
      final v = audio[i * hop + j];
      s += v * v;
    }
    final rms = math.sqrt(s / win);
    if (rms > thresholdRms) {
      if (!found) {
        firstActive = i;
        found = true;
      }
      lastActive = i;
    }
  }
  if (!found) return audio;
  final start = firstActive * hop;
  final end = (lastActive + 1) * hop + win;
  final hi = end > audio.length ? audio.length : end;
  if (hi - start < 1024) return audio;
  return Float64List.fromList(audio.sublist(start, hi));
}

double pearson(List<double> a, List<double> b) {
  final n = a.length;
  if (n < 2) return 0.0;
  double sumA = 0, sumB = 0;
  for (int i = 0; i < n; i++) {
    sumA += a[i];
    sumB += b[i];
  }
  final ma = sumA / n, mb = sumB / n;
  double num = 0, da = 0, db = 0;
  for (int i = 0; i < n; i++) {
    final xa = a[i] - ma, xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  final den = math.sqrt(da * db);
  return den > 1e-12 ? num / den : 1.0;
}

void compareGroups(String title, List<double> py, Float64List dart) {
  const ranges = [
    [0, 13, 'MFCC mean'],
    [13, 26, 'MFCC std'],
    [26, 39, 'MFCC min'],
    [39, 52, 'MFCC max'],
    [52, 65, 'dMFCC mean'],
    [65, 78, 'dMFCC std'],
    [78, 80, 'ZCR'],
    [80, 88, 'Spektral 4x2'],
    [88, 95, 'Kontrast 7b'],
    [95, 99, 'Enerji'],
    [99, 111, 'Chroma 12'],
    [111, 114, 'f2/db/entropi'],
    [114, 120, 'Zaman alani'],
  ];
  print('--- $title ---');
  double sumR = 0;
  for (final r in ranges) {
    final s = r[0] as int, e = r[1] as int, name = r[2] as String;
    final pr = pearson(py.sublist(s, e), dart.sublist(s, e));
    sumR += pr;
    final flag = pr >= 0.95 ? 'OK ' : (pr >= 0.85 ? 'ORTA' : 'DUSUK');
    print('  ${name.padRight(14)} r=${pr.toStringAsFixed(3)}  $flag');
  }
  print('  ${"ORTALAMA".padRight(14)} r=${(sumR / ranges.length).toStringAsFixed(3)}');
  // Kritik tekil değerler
  print('  f2:  py=${py[111].toStringAsFixed(1)}Hz  dart=${dart[111].toStringAsFixed(1)}Hz');
  print('  entropi: py=${py[113].toStringAsFixed(3)}  dart=${dart[113].toStringAsFixed(3)}');
}

Future<void> main() async {
  final acoustic = AcousticFeatureExtractor();
  final hh = HhFeatureExtractor();

  // ---------- 1) Referans WAV parity ----------
  final refBytes = File('assets/test/wm1_chu1.wav').readAsBytesSync();
  final refAudio = Float64List.fromList(Wav.read(refBytes).toMono());
  final pyJson = jsonDecode(
          File('assets/test/wm1_chu1_python_features.json').readAsStringSync())
      as Map<String, dynamic>;
  final pyFeat = (pyJson['features'] as List)
      .map((e) => (e as num).toDouble())
      .toList();

  print('=== REFERANS WAV (karpuz #1, ${refAudio.length} ornek) ===\n');

  final sw1 = Stopwatch()..start();
  final dartFull = acoustic.extract(refAudio);
  sw1.stop();
  compareGroups('TAM SES (Python referansiyla ayni kosul)', pyFeat, dartFull);
  print('  sure: ${sw1.elapsedMilliseconds} ms\n');

  final trimmed = trimSilence(refAudio);
  print(
      'trim: ${refAudio.length} -> ${trimmed.length} ornek (%${(100 * trimmed.length / refAudio.length).toStringAsFixed(0)} kaldi)');
  final sw2 = Stopwatch()..start();
  final dartTrim = acoustic.extract(trimmed);
  sw2.stop();
  compareGroups('CANLI PIPELINE (trim sonrasi) vs Python tam-ses', pyFeat, dartTrim);
  print('  sure: ${sw2.elapsedMilliseconds} ms\n');

  // HH timing on full audio (the old O(N^2) bottleneck)
  final sw3 = Stopwatch()..start();
  final hhFull = hh.extract(refAudio);
  sw3.stop();
  print('HH 8-D tam ses: ${sw3.elapsedMilliseconds} ms  hh_score=${hhFull[0].toStringAsFixed(3)}\n');

  // ---------- 2) 12 bundled ornek icin feature dump ----------
  final manifest = jsonDecode(
          File('assets/samples/manifest.json').readAsStringSync())
      as Map<String, dynamic>;
  final samples = (manifest['samples'] as List)
      .map((e) => Map<String, dynamic>.from(e as Map))
      .toList();

  final out = <String, dynamic>{};
  print('=== 12 BUNDLED ORNEK (canli pipeline simulasyonu: TAM SES, trim yok) ===');
  for (final s in samples) {
    final asset = s['asset'] as String;
    final fname = asset.split('/').last;
    final bytes = File(asset).readAsBytesSync();
    final audio = Float64List.fromList(Wav.read(bytes).toMono());
    final a = acoustic.extract(audio);
    final h = hh.extract(audio);
    out[fname] = {
      'label': s['label'],
      'acoustic_120': a.toList(),
      'hh_8': h.toList(),
    };
    print('  $fname  f2=${a[111].toStringAsFixed(0)}Hz  hh=${h[0].toStringAsFixed(2)}  label=${s['label']}');
  }
  File('tool/dart_features_all_samples.json')
      .writeAsStringSync(jsonEncode(out));
  print('\nYazildi: tool/dart_features_all_samples.json');
  print('Simdi calistir: python tool/parity_predict.py');
}
