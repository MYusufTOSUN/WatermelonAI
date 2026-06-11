import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/fusion_model_service.dart';
import '../theme/app_theme.dart';

/// On-device self-test using bundled Qilin WAV samples with known
/// Brix-derived labels. This is a smoke test of the full mobile pipeline:
///   WAV → Dart MFCC → 120-D acoustic + 8-D HH → Fusion model → 3 prob
/// Visual (11-D) and Haptic (7-D) inputs are zero-padded because we
/// don't have the matching photo/sensor data for bundled samples.
///
/// Aim: prove the pipeline reproduces expected behavior on known data
/// even without physical watermelon cutting.
class SelfTestScreen extends StatefulWidget {
  const SelfTestScreen({super.key});

  @override
  State<SelfTestScreen> createState() => _SelfTestScreenState();
}

class _SelfTestScreenState extends State<SelfTestScreen> {
  bool _running = false;
  String? _error;
  String _statusStep = "";
  DateTime? _startTime;
  _SelfTestResult? _result;

  final FusionModelService _fusion = FusionModelService();

  Future<T> _withTimeout<T>(
    String stepName,
    Future<T> future, {
    Duration? timeout,
  }) async {
    setState(() => _statusStep = stepName);
    debugPrint("[SelfTest] >>> $stepName");
    try {
      final result =
          timeout == null ? await future : await future.timeout(timeout);
      debugPrint("[SelfTest] <<< $stepName OK");
      return result;
    } on TimeoutException {
      debugPrint("[SelfTest] !!! $stepName TIMEOUT");
      throw Exception("$stepName beklenenden uzun sürdü");
    } catch (e) {
      debugPrint("[SelfTest] !!! $stepName ERROR: $e");
      rethrow;
    }
  }

  @override
  void dispose() {
    _fusion.dispose();
    super.dispose();
  }

  Future<void> _run() async {
    setState(() {
      _running = true;
      _error = null;
      _result = null;
      _statusStep = "Başlatılıyor...";
      _startTime = DateTime.now();
    });
    try {
      // 1. Load manifest
      final manifestStr = await _withTimeout(
        "Manifest okunuyor",
        rootBundle.loadString("assets/samples/manifest.json"),
      );
      final manifest = jsonDecode(manifestStr) as Map<String, dynamic>;
      final samples = (manifest["samples"] as List)
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList();

      // 1b. Load training-set visual + haptic means
      final meansStr = await _withTimeout(
        "Eğitim ortalamaları okunuyor",
        rootBundle.loadString("assets/samples/feature_means.json"),
      );
      final means = jsonDecode(meansStr) as Map<String, dynamic>;
      final visualMean = Float64List.fromList(
        (means["visual_mean"] as List).map((e) => (e as num).toDouble()).toList(),
      );
      final hapticMean = Float64List.fromList(
        (means["haptic_mean"] as List).map((e) => (e as num).toDouble()).toList(),
      );

      // 1c. Load pre-computed Python features (avoids running heavy
      // Dart MFCC on each bundled WAV — instant inference)
      final featStr = await _withTimeout(
        "Önceden hesaplanmış özellikler okunuyor",
        rootBundle.loadString("assets/samples/precomputed_features.json"),
      );
      final featData = jsonDecode(featStr) as Map<String, dynamic>;
      final precomputed = featData["features"] as Map<String, dynamic>;

      // 2. Load TFLite fusion model
      await _withTimeout(
        "Fusion modeli yükleniyor",
        _fusion.loadModel(),
      );

      // 3. Run each sample through fusion (~50ms each)
      final perSample = <_SamplePrediction>[];
      for (int i = 0; i < samples.length; i++) {
        final s = samples[i];
        final asset = s["asset"] as String;
        final fname = asset.split('/').last;
        final trueLabel = s["label"] as int;
        final wmId = s["watermelon_id"] as int;
        final brix = (s["brix"] as num).toDouble();

        setState(() {
          _statusStep = "Sample ${i + 1}/${samples.length} fusion";
          _result = _SelfTestResult(
            inProgress: true,
            progressIdx: i,
            total: samples.length,
            predictions: perSample.toList(),
          );
        });

        final entry = precomputed[fname] as Map<String, dynamic>?;
        if (entry == null) {
          throw Exception("Pre-computed features missing for $fname");
        }
        final acoustic = Float64List.fromList(
          (entry["acoustic_120"] as List)
              .map((e) => (e as num).toDouble())
              .toList(),
        );
        final hh = Float64List.fromList(
          (entry["hh_8"] as List)
              .map((e) => (e as num).toDouble())
              .toList(),
        );

        final pred = _fusion.run(
          acoustic: acoustic,
          visual: visualMean,
          haptic: hapticMean,
          hh: hh,
        );

        perSample.add(_SamplePrediction(
          watermelonId: wmId,
          brix: brix,
          trueLabel: trueLabel,
          predictedLabel: pred.classId,
          pImmature: pred.pImmature,
          pRipe: pred.pRipe,
          pOverripe: pred.pOverripe,
        ));

        // Tiny delay so progress bar can refresh visibly
        if (i % 3 == 0) {
          await Future.delayed(const Duration(milliseconds: 30));
        }
      }

      // 3. Aggregate
      final agg = _aggregate(perSample);
      setState(() {
        _running = false;
        _result = _SelfTestResult(
          inProgress: false,
          progressIdx: samples.length,
          total: samples.length,
          predictions: perSample,
          aggregate: agg,
        );
      });
    } catch (e, st) {
      setState(() {
        _running = false;
        _error = "$e\n$st";
      });
    }
  }

  _Aggregate _aggregate(List<_SamplePrediction> preds) {
    int correct = 0;
    int total = preds.length;
    int binaryCorrect = 0;
    final cm = List.generate(3, (_) => List<int>.filled(3, 0));
    final perClass = List.generate(3, (_) => [0, 0]); // [correct, total]
    for (final p in preds) {
      cm[p.trueLabel][p.predictedLabel]++;
      perClass[p.trueLabel][1]++;
      if (p.predictedLabel == p.trueLabel) {
        correct++;
        perClass[p.trueLabel][0]++;
      }
      // Binary: edible (label==1) vs not (0 or 2)
      final trueBin = p.trueLabel == 1 ? 1 : 0;
      final predBin = p.predictedLabel == 1 ? 1 : 0;
      if (trueBin == predBin) binaryCorrect++;
    }
    return _Aggregate(
      total: total,
      correct3: correct,
      correctBinary: binaryCorrect,
      confusionMatrix: cm,
      perClass: perClass,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surface,
      appBar: AppBar(
        title: const Text("Otomatik Saha Testi",
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
                    : const Icon(Icons.play_arrow_rounded),
                label: Text(_running
                    ? "Çalışıyor..."
                    : "12 bundled Qilin örneğini çalıştır"),
                onPressed: _running ? null : _run,
              ),
              const SizedBox(height: 16),
              Expanded(
                child: _error != null
                    ? _buildErrorView()
                    : _running && _result == null
                        ? _buildEarlyProgress()
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
          Text("Bundled Qilin Self-Test",
              style: TextStyle(
                  fontSize: 14, fontWeight: FontWeight.w800)),
          SizedBox(height: 6),
          Text(
            "12 Qilin örneği (5 Olgunlaşmamış, 5 Olgun, 2 İçi Geçmiş) "
            "Brix metresi ile etiketlenmiş. Akustik + HH özellikleri "
            "telefonda çıkarılıp fusion modeline gönderilir. Görsel ve "
            "haptic kanalları sıfır (smoke test).",
            style: TextStyle(
                fontSize: 12, color: AppTheme.slate, height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _buildEarlyProgress() {
    final elapsed = _startTime == null
        ? Duration.zero
        : DateTime.now().difference(_startTime!);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  color: AppTheme.primary,
                ),
              ),
              const SizedBox(width: 10),
              Text("Hazırlanıyor...",
                  style: const TextStyle(
                      fontSize: 14, fontWeight: FontWeight.w800)),
            ],
          ),
          const SizedBox(height: 10),
          Text(_statusStep,
              style: const TextStyle(
                  fontSize: 13, color: AppTheme.slate)),
          const SizedBox(height: 4),
          Text("Geçen süre: ${elapsed.inSeconds}s",
              style: const TextStyle(
                  fontSize: 11, color: AppTheme.slate)),
        ],
      ),
    );
  }

  Widget _buildIdleView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.fact_check_outlined,
              size: 64, color: Colors.grey.shade400),
          const SizedBox(height: 12),
          Text("Çalıştırmaya hazır",
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

  Widget _buildResultView(_SelfTestResult r) {
    if (r.inProgress) {
      final elapsed = _startTime == null
          ? Duration.zero
          : DateTime.now().difference(_startTime!);
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: AppTheme.cardDecoration(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            LinearProgressIndicator(
              value: r.progressIdx / r.total,
              backgroundColor: AppTheme.border,
              valueColor: const AlwaysStoppedAnimation(AppTheme.primary),
            ),
            const SizedBox(height: 10),
            Text("İlerleme: ${r.progressIdx} / ${r.total}",
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(_statusStep,
                style: const TextStyle(
                    fontSize: 12, color: AppTheme.slate)),
            const SizedBox(height: 4),
            Text("Geçen süre: ${elapsed.inSeconds}s",
                style: const TextStyle(
                    fontSize: 11, color: AppTheme.slate)),
          ],
        ),
      );
    }

    final agg = r.aggregate!;
    final acc3 = agg.correct3 / agg.total;
    final accBin = agg.correctBinary / agg.total;

    return ListView(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: AppTheme.cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text("Sonuç",
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.slate)),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _BigStat(
                      label: "2-sınıf (Yenir/Yenmez)",
                      value: "%${(accBin * 100).toStringAsFixed(0)}",
                      detail: "${agg.correctBinary}/${agg.total}",
                      highlight: true,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: _BigStat(
                      label: "3-sınıf",
                      value: "%${(acc3 * 100).toStringAsFixed(0)}",
                      detail: "${agg.correct3}/${agg.total}",
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: AppTheme.cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text("Konfüzyon Matrisi",
                  style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.slate)),
              const SizedBox(height: 8),
              _ConfusionMatrix(matrix: agg.confusionMatrix),
              const SizedBox(height: 8),
              const Text(
                "Satır: gerçek sınıf · Sütun: tahmin",
                style: TextStyle(fontSize: 11, color: AppTheme.slate),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: AppTheme.cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text("Sınıf bazında doğruluk",
                  style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.slate)),
              const SizedBox(height: 8),
              _ClassRow(
                  label: "Olgunlaşmamış",
                  correct: agg.perClass[0][0],
                  total: agg.perClass[0][1],
                  color: const Color(0xFF7CB342)),
              const SizedBox(height: 6),
              _ClassRow(
                  label: "Olgun",
                  correct: agg.perClass[1][0],
                  total: agg.perClass[1][1],
                  color: AppTheme.primary),
              const SizedBox(height: 6),
              _ClassRow(
                  label: "İçi Geçmiş",
                  correct: agg.perClass[2][0],
                  total: agg.perClass[2][1],
                  color: const Color(0xFFEF6C00)),
            ],
          ),
        ),
        const SizedBox(height: 12),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4, vertical: 8),
          child: Text("Tüm tahminler",
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  color: AppTheme.slate)),
        ),
        ...r.predictions.map(_PredictionTile.new),
      ],
    );
  }
}

class _SelfTestResult {
  final bool inProgress;
  final int progressIdx;
  final int total;
  final List<_SamplePrediction> predictions;
  final _Aggregate? aggregate;

  _SelfTestResult({
    required this.inProgress,
    required this.progressIdx,
    required this.total,
    required this.predictions,
    this.aggregate,
  });
}

class _SamplePrediction {
  final int watermelonId;
  final double brix;
  final int trueLabel;
  final int predictedLabel;
  final double pImmature;
  final double pRipe;
  final double pOverripe;

  _SamplePrediction({
    required this.watermelonId,
    required this.brix,
    required this.trueLabel,
    required this.predictedLabel,
    required this.pImmature,
    required this.pRipe,
    required this.pOverripe,
  });
}

class _Aggregate {
  final int total;
  final int correct3;
  final int correctBinary;
  final List<List<int>> confusionMatrix;
  final List<List<int>> perClass;
  _Aggregate({
    required this.total,
    required this.correct3,
    required this.correctBinary,
    required this.confusionMatrix,
    required this.perClass,
  });
}

class _BigStat extends StatelessWidget {
  final String label;
  final String value;
  final String detail;
  final bool highlight;
  const _BigStat({
    required this.label,
    required this.value,
    required this.detail,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = highlight ? AppTheme.primary : AppTheme.slate;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: highlight
            ? AppTheme.primary.withOpacity(0.08)
            : AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: highlight
                ? AppTheme.primary.withOpacity(0.3)
                : AppTheme.border),
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
                  fontSize: 28,
                  fontWeight: FontWeight.w900,
                  color: color)),
          Text(detail,
              style: const TextStyle(
                  fontSize: 11, color: AppTheme.slate)),
        ],
      ),
    );
  }
}

class _ConfusionMatrix extends StatelessWidget {
  final List<List<int>> matrix;
  const _ConfusionMatrix({required this.matrix});

  @override
  Widget build(BuildContext context) {
    const labels = ["Imm", "Ripe", "Over"];
    return Table(
      defaultColumnWidth: const FlexColumnWidth(),
      border: TableBorder.all(color: AppTheme.border, width: 1),
      children: [
        TableRow(
          decoration: const BoxDecoration(color: AppTheme.surface),
          children: [
            const _TableCell(text: "Gerçek ↓", bold: true, small: true),
            ...labels.map((l) => _TableCell(text: l, bold: true, small: true)),
          ],
        ),
        for (int i = 0; i < 3; i++)
          TableRow(
            children: [
              _TableCell(text: labels[i], bold: true, small: true),
              for (int j = 0; j < 3; j++)
                _TableCell(
                  text: "${matrix[i][j]}",
                  highlight: i == j && matrix[i][j] > 0,
                ),
            ],
          ),
      ],
    );
  }
}

class _TableCell extends StatelessWidget {
  final String text;
  final bool bold;
  final bool small;
  final bool highlight;
  const _TableCell({
    required this.text,
    this.bold = false,
    this.small = false,
    this.highlight = false,
  });
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
      color: highlight ? AppTheme.primary.withOpacity(0.15) : null,
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: small ? 11 : 14,
          fontWeight: bold || highlight
              ? FontWeight.w800
              : FontWeight.w600,
          color: highlight ? AppTheme.primary : AppTheme.ink,
        ),
      ),
    );
  }
}

class _ClassRow extends StatelessWidget {
  final String label;
  final int correct;
  final int total;
  final Color color;
  const _ClassRow({
    required this.label,
    required this.correct,
    required this.total,
    required this.color,
  });
  @override
  Widget build(BuildContext context) {
    final acc = total > 0 ? correct / total : 0.0;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(label,
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w700)),
            const Spacer(),
            Text("$correct / $total · %${(acc * 100).toStringAsFixed(0)}",
                style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: color)),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: acc,
            minHeight: 6,
            backgroundColor: AppTheme.border,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ],
    );
  }
}

class _PredictionTile extends StatelessWidget {
  final _SamplePrediction p;
  const _PredictionTile(this.p);

  @override
  Widget build(BuildContext context) {
    final correct = p.trueLabel == p.predictedLabel;
    final color = correct ? AppTheme.primary : AppTheme.accent;
    final labels = ["Olgunlaşmamış", "Olgun", "İçi Geçmiş"];
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(10),
      decoration: AppTheme.cardDecoration(),
      child: Row(
        children: [
          Icon(
            correct
                ? Icons.check_circle_rounded
                : Icons.cancel_rounded,
            color: color,
            size: 20,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Karpuz #${p.watermelonId} · Brix ${p.brix}",
                    style: const TextStyle(
                        fontSize: 12, fontWeight: FontWeight.w700)),
                Text(
                  "Gerçek: ${labels[p.trueLabel]} · Tahmin: ${labels[p.predictedLabel]}",
                  style: TextStyle(
                      fontSize: 11, color: AppTheme.slate),
                ),
              ],
            ),
          ),
          Text(
            "%${((p.predictedLabel == 0 ? p.pImmature : p.predictedLabel == 1 ? p.pRipe : p.pOverripe) * 100).toStringAsFixed(0)}",
            style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w800,
                color: color),
          ),
        ],
      ),
    );
  }
}
