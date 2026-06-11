import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:wav/wav.dart';

import '../services/feature_extractors/isolate_workers.dart';
import '../theme/app_theme.dart';

/// Debug screen that loads a bundled reference Qilin WAV, extracts
/// 120-D acoustic features in Dart, and compares them against the
/// Python librosa reference (also bundled as JSON).
///
/// This lets us **numerically prove** the Dart-side acoustic feature
/// extraction matches (or quantify the gap to) the backend pipeline.
class DebugValidateScreen extends StatefulWidget {
  const DebugValidateScreen({super.key});

  @override
  State<DebugValidateScreen> createState() => _DebugValidateScreenState();
}

class _DebugValidateScreenState extends State<DebugValidateScreen> {
  bool _running = false;
  String? _error;
  _ValidationResult? _result;

  Future<void> _runValidation() async {
    setState(() {
      _running = true;
      _error = null;
      _result = null;
    });
    try {
      // 1. Load reference Python features
      final pyJsonStr =
          await rootBundle.loadString("assets/test/wm1_chu1_python_features.json");
      final pyJson = jsonDecode(pyJsonStr) as Map<String, dynamic>;
      final pyFeatures = (pyJson["features"] as List)
          .map((e) => (e as num).toDouble())
          .toList();

      // 2. Load bundled WAV from assets
      final wavBytes =
          await rootBundle.load("assets/test/wm1_chu1.wav");

      // 3. Decode WAV → Float64List PCM
      final wavObj = Wav.read(wavBytes.buffer.asUint8List());
      final audio = Float64List.fromList(wavObj.toMono());

      // 4. Extract Dart 120-D features in background isolate
      final dartFeatures = await compute(acousticExtractIsolate, audio);

      // 5. Compare
      final result = _compare(pyFeatures, dartFeatures, pyJson);
      setState(() {
        _running = false;
        _result = result;
      });
    } catch (e, st) {
      setState(() {
        _running = false;
        _error = "$e\n$st";
      });
    }
  }

  _ValidationResult _compare(
    List<double> py,
    Float64List dart,
    Map<String, dynamic> pyJson,
  ) {
    final groups = <_GroupComparison>[];
    final ranges = <List<dynamic>>[
      <dynamic>[0, 13, "MFCC mean (13)"],
      <dynamic>[13, 26, "MFCC std (13)"],
      <dynamic>[26, 39, "MFCC min (13)"],
      <dynamic>[39, 52, "MFCC max (13)"],
      <dynamic>[52, 65, "Delta MFCC mean (13)"],
      <dynamic>[65, 78, "Delta MFCC std (13)"],
      <dynamic>[78, 80, "ZCR (2)"],
      <dynamic>[80, 88, "Spectral 4x{mean,std} (8)"],
      <dynamic>[88, 95, "Contrast 7-band (7)"],
      <dynamic>[95, 99, "Energy (4)"],
      <dynamic>[99, 111, "Chroma 12"],
      <dynamic>[111, 114, "f2 / f2_db / entropy (3)"],
      <dynamic>[114, 120, "Time-domain (6)"],
    ];

    double totalMse = 0.0;
    double totalCorrSum = 0.0;
    int groupCount = 0;

    for (final r in ranges) {
      final start = r[0] as int;
      final end = r[1] as int;
      final name = r[2] as String;
      final pySlice = py.sublist(start, end);
      final dartSlice = dart.sublist(start, end);

      double mse = 0.0;
      for (int i = 0; i < pySlice.length; i++) {
        final d = pySlice[i] - dartSlice[i];
        mse += d * d;
      }
      mse /= pySlice.length;

      // Normalized MSE = MSE / (mean(|py|)^2 + epsilon)
      double absMean = 0.0;
      for (final v in pySlice) {
        absMean += v.abs();
      }
      absMean /= pySlice.length;
      final nmse = mse / ((absMean * absMean) + 1e-6);

      // Pearson correlation
      final corr = _pearson(pySlice, dartSlice);

      totalMse += mse;
      totalCorrSum += corr.isFinite ? corr : 0.0;
      groupCount++;

      groups.add(_GroupComparison(
        name: name,
        py: pySlice,
        dart: dartSlice,
        mse: mse,
        nmse: nmse,
        correlation: corr,
      ));
    }

    return _ValidationResult(
      wavPath: pyJson["wav_path"] as String,
      durationS: (pyJson["duration_s"] as num).toDouble(),
      averageCorrelation: totalCorrSum / groupCount,
      averageMse: totalMse / groupCount,
      groups: groups,
    );
  }

  double _pearson(List<double> a, List<double> b) {
    if (a.length < 2) return 0.0;
    final n = a.length;
    double sumA = 0.0, sumB = 0.0;
    for (int i = 0; i < n; i++) {
      sumA += a[i];
      sumB += b[i];
    }
    final meanA = sumA / n;
    final meanB = sumB / n;
    double num = 0.0, denA = 0.0, denB = 0.0;
    for (int i = 0; i < n; i++) {
      final da = a[i] - meanA;
      final db = b[i] - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }
    final denom = math.sqrt(denA * denB);
    return denom > 1e-10 ? num / denom : 0.0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surface,
      appBar: AppBar(
        title: const Text("Doğrulama Testi",
            style: TextStyle(fontWeight: FontWeight.w800)),
        backgroundColor: Colors.white,
        foregroundColor: AppTheme.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildHeader(),
              const SizedBox(height: 14),
              FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor: AppTheme.primary,
                  minimumSize: const Size.fromHeight(54),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                icon: _running
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2.5,
                        ),
                      )
                    : const Icon(Icons.science_rounded),
                label: Text(_running
                    ? "Çıkartılıyor..."
                    : "Bundled WAV ile doğrulamayı çalıştır"),
                onPressed: _running ? null : _runValidation,
              ),
              const SizedBox(height: 16),
              Expanded(
                child: _error != null
                    ? _buildErrorView()
                    : _result == null
                        ? _buildIdleView()
                        : _buildResultView(_result!),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Dart MFCC ↔ Python librosa karşılaştırması",
              style: TextStyle(
                  fontSize: 14, fontWeight: FontWeight.w800)),
          SizedBox(height: 6),
          Text(
            "Bundled Qilin WAV (karpuz #1, 3 sn) Dart-tarafı feature extractor'a verilir, "
            "Python ile çıkarılmış referans 120-D vektör ile karşılaştırılır. "
            "Yüksek korelasyon (>0.95) Dart pipeline'ının numerical olarak doğru çalıştığını gösterir.",
            style: TextStyle(
                fontSize: 12, color: AppTheme.slate, height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _buildIdleView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.science_outlined,
              size: 64, color: Colors.grey.shade400),
          const SizedBox(height: 12),
          Text("Henüz çalıştırılmadı",
              style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade600,
                  fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _buildErrorView() {
    return SingleChildScrollView(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.red.shade50,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.red.shade200),
        ),
        child: Text(_error ?? "",
            style: TextStyle(
                fontSize: 11, color: Colors.red.shade800)),
      ),
    );
  }

  Widget _buildResultView(_ValidationResult r) {
    final goodCorr = r.averageCorrelation >= 0.9;
    final corrColor = goodCorr ? AppTheme.primary : AppTheme.accent;
    return ListView(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: AppTheme.cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                      goodCorr
                          ? Icons.check_circle_rounded
                          : Icons.warning_amber_rounded,
                      color: corrColor),
                  const SizedBox(width: 8),
                  Text(
                    goodCorr
                        ? "Dart pipeline doğrulandı"
                        : "Anlamlı sapma var",
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                        color: corrColor),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text("Referans: ${r.wavPath}",
                  style: const TextStyle(
                      fontSize: 11, color: AppTheme.slate)),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _StatBox(
                      label: "Ortalama korelasyon",
                      value: r.averageCorrelation.toStringAsFixed(3),
                      good: goodCorr,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: _StatBox(
                      label: "Ortalama MSE",
                      value: r.averageMse.toStringAsExponential(2),
                      good: r.averageMse < 100,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Text("Grup bazlı karşılaştırma",
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  color: AppTheme.slate)),
        ),
        const SizedBox(height: 8),
        ...r.groups.map(_GroupCard.new),
      ],
    );
  }
}

class _StatBox extends StatelessWidget {
  final String label;
  final String value;
  final bool good;
  const _StatBox({
    required this.label,
    required this.value,
    required this.good,
  });
  @override
  Widget build(BuildContext context) {
    final color = good ? AppTheme.primary : AppTheme.accent;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 11, color: AppTheme.slate)),
          const SizedBox(height: 4),
          Text(value,
              style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  color: color)),
        ],
      ),
    );
  }
}

class _ValidationResult {
  final String wavPath;
  final double durationS;
  final double averageCorrelation;
  final double averageMse;
  final List<_GroupComparison> groups;

  _ValidationResult({
    required this.wavPath,
    required this.durationS,
    required this.averageCorrelation,
    required this.averageMse,
    required this.groups,
  });
}

class _GroupComparison {
  final String name;
  final List<double> py;
  final List<double> dart;
  final double mse;
  final double nmse;
  final double correlation;

  _GroupComparison({
    required this.name,
    required this.py,
    required this.dart,
    required this.mse,
    required this.nmse,
    required this.correlation,
  });
}

class _GroupCard extends StatelessWidget {
  final _GroupComparison group;
  const _GroupCard(this.group);

  @override
  Widget build(BuildContext context) {
    final good = group.correlation >= 0.9;
    final color = good ? AppTheme.primary : AppTheme.accent;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(group.name,
                    style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700)),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text("r=${group.correlation.toStringAsFixed(3)}",
                    style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        color: color)),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            "MSE=${group.mse.toStringAsExponential(2)} · "
            "nMSE=${group.nmse.toStringAsFixed(3)}",
            style: const TextStyle(
                fontSize: 11, color: AppTheme.slate),
          ),
          const SizedBox(height: 6),
          Text("Python : ${_format(group.py)}",
              style: const TextStyle(
                  fontSize: 10,
                  color: AppTheme.slate,
                  fontFamily: 'monospace')),
          Text("Dart   : ${_format(group.dart)}",
              style: TextStyle(
                  fontSize: 10,
                  color: color,
                  fontFamily: 'monospace')),
        ],
      ),
    );
  }

  String _format(List<double> v) {
    final shown = v.take(5).map((x) => x.toStringAsFixed(2)).join(", ");
    return v.length > 5 ? "[$shown, ...]" : "[$shown]";
  }
}
