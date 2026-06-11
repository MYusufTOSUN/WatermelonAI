import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/ripeness_result.dart';

/// MobileNetV3 3-sınıf görsel olgunluk sınıflandırıcısı
///
/// Giriş:  (1, 224, 224, 3) float32 — [-1, 1] aralığında (MobileNetV3 preprocess)
/// Çıkış:  (1, 3) float32 — [Olgunlaşmamış, Olgun, İçi Geçmiş]
class VisualClassifierService {
  static const String _modelAsset =
      "assets/models/visual_classifier_fp16.tflite";
  static const int _inputSize = 224;

  Interpreter? _interpreter;
  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    if (_interpreter != null) return;
    try {
      _interpreter = await Interpreter.fromAsset(_modelAsset);
      _interpreter!.allocateTensors();
      print("[VisualClassifier] Model yüklendi: $_modelAsset");
    } catch (e) {
      print("[VisualClassifier] Model yükleme hatası: $e");
      rethrow;
    }
  }

  /// Fotoğraftan sınıflandırma yap
  Future<RipenessResult> classifyImage(File imageFile) async {
    if (_interpreter == null) {
      await loadModel();
    }

    final bytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception("Görsel okunamadı: ${imageFile.path}");
    }

    final input = _preprocess(decoded);
    final output = List.generate(1, (_) => List.filled(3, 0.0))
        .map((e) => e.cast<double>())
        .toList();

    _interpreter!.run(input, output);

    final probs = output[0];
    double pImmature = probs[0].toDouble();
    double pRipe = probs[1].toDouble();
    double pOverripe = probs[2].toDouble();

    // Sayısal güvenlik: softmax sonrası toplam ~1 olur, yine de normalize et
    final total = pImmature + pRipe + pOverripe;
    if (total > 0) {
      pImmature /= total;
      pRipe /= total;
      pOverripe /= total;
    }

    int classId = 0;
    double best = pImmature;
    if (pRipe > best) {
      best = pRipe;
      classId = 1;
    }
    if (pOverripe > best) {
      classId = 2;
    }

    return RipenessResult(
      classId: classId,
      pImmature: pImmature,
      pRipe: pRipe,
      pOverripe: pOverripe,
    );
  }

  /// Görseli 224x224'e ölçekle + [-1, 1] normalizasyon
  List<List<List<List<double>>>> _preprocess(img.Image src) {
    final resized =
        img.copyResize(src, width: _inputSize, height: _inputSize);

    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (_) => List.generate(_inputSize, (_) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final p = resized.getPixel(x, y);
        // MobileNetV3 preprocess: pixel / 127.5 - 1.0
        input[0][y][x][0] = (p.r / 127.5) - 1.0;
        input[0][y][x][1] = (p.g / 127.5) - 1.0;
        input[0][y][x][2] = (p.b / 127.5) - 1.0;
      }
    }
    return input;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
