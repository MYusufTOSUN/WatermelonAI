import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import '../models/multimodal_result.dart';

/// Generates a complete PDF report of a single watermelon analysis
/// containing EVERY measurement, and returns the saved file path.
class PdfReportService {
  static const PdfColor _green = PdfColor.fromInt(0xFF1B7F3A);
  static const PdfColor _red = PdfColor.fromInt(0xFFE53935);
  static const PdfColor _amber = PdfColor.fromInt(0xFFFFB300);
  static const PdfColor _slate = PdfColor.fromInt(0xFF5F6B5C);
  static const PdfColor _border = PdfColor.fromInt(0xFFE3E8DC);

  Future<File> generate({
    required MultimodalResult result,
    required String imagePath,
  }) async {
    final doc = pw.Document();
    final fusion = result.fusion;
    final vl = result.viLiquid;

    final isEdible = vl?.isEdible ?? fusion.isEdible;
    final verdict = isEdible ? "AL" : "ALMA";
    final verdictColor = isEdible ? _green : _red;
    final label = vl?.simpleLabel ?? fusion.simpleLabel;
    final confidence = vl?.verdictConfidence ?? fusion.edibleConfidence;

    Uint8List? imgBytes;
    try {
      final f = File(imagePath);
      if (f.existsSync()) imgBytes = f.readAsBytesSync();
    } catch (_) {}

    final ts = result.timestamp;
    final tarih =
        "${ts.day.toString().padLeft(2, '0')}.${ts.month.toString().padLeft(2, '0')}.${ts.year} "
        "${ts.hour.toString().padLeft(2, '0')}:${ts.minute.toString().padLeft(2, '0')}";

    doc.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(28),
        build: (context) => [
          // Başlık
          pw.Row(
            mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text("Karpuz Analiz Raporu",
                      style: pw.TextStyle(
                          fontSize: 22, fontWeight: pw.FontWeight.bold)),
                  pw.SizedBox(height: 2),
                  pw.Text("Karpuz Dedektifi · Çok Modlu Olgunluk Testi",
                      style: const pw.TextStyle(
                          fontSize: 10, color: _slate)),
                ],
              ),
              pw.Text(tarih,
                  style: const pw.TextStyle(fontSize: 10, color: _slate)),
            ],
          ),
          pw.SizedBox(height: 16),

          // Karar bandı
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(16),
            decoration: pw.BoxDecoration(
              color: verdictColor,
              borderRadius: pw.BorderRadius.circular(10),
            ),
            child: pw.Row(
              children: [
                pw.Text(verdict,
                    style: pw.TextStyle(
                        color: PdfColors.white,
                        fontSize: 30,
                        fontWeight: pw.FontWeight.bold)),
                pw.SizedBox(width: 16),
                pw.Expanded(
                  child: pw.Column(
                    crossAxisAlignment: pw.CrossAxisAlignment.start,
                    children: [
                      pw.Text(
                          isEdible
                              ? "Bu karpuz iyi gibi"
                              : "Bu karpuz iyi değil",
                          style: pw.TextStyle(
                              color: PdfColors.white,
                              fontSize: 14,
                              fontWeight: pw.FontWeight.bold)),
                      pw.Text(
                          "$label · %${(confidence * 100).toStringAsFixed(0)} güven",
                          style: const pw.TextStyle(
                              color: PdfColors.white, fontSize: 11)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          pw.SizedBox(height: 16),

          // Foto
          if (imgBytes != null) ...[
            pw.ClipRRect(
              horizontalRadius: 8,
              verticalRadius: 8,
              child: pw.Image(
                pw.MemoryImage(imgBytes),
                height: 200,
                width: double.infinity,
                fit: pw.BoxFit.cover,
              ),
            ),
            pw.SizedBox(height: 16),
          ],

          // Yapay zeka olasılıkları
          _sectionTitle("Yapay Zeka Tahmini (Sınıf Olasılıkları)"),
          _probTable(fusion),
          pw.SizedBox(height: 14),

          // Akustik ölçümler
          _sectionTitle("Akustik Ölçümler (Vuruş Sesi)"),
          _kvTable([
            ["Vuruş sesi rezonansı (f2)", "${result.f2Hz.toStringAsFixed(0)} Hz"],
            ["f2 genliği", "${result.f2Db.toStringAsFixed(1)} dB"],
            ["İçi boş riski (akustik)", "%${(result.hhScore * 100).toStringAsFixed(0)}"],
            ["Vuruş enerjisi (sinyal RMS)", result.signalRms.toStringAsFixed(4)],
          ]),
          pw.SizedBox(height: 14),

          // Haptik / temas
          _sectionTitle("Dokunsal Ölçümler"),
          _kvTable([
            ["Temas kalitesi", "%${(result.contactQuality * 100).toStringAsFixed(0)}"],
          ]),
          pw.SizedBox(height: 14),

          // Vi-Liquid aktif haptik
          if (vl != null) ...[
            _sectionTitle("Vi-Liquid Aktif Haptik (Bilgi Amaçlı)"),
            _kvTable([
              ["Titreşim yankı tonu (SRR f2)", "${vl.f2Hz.toStringAsFixed(0)} Hz"],
              ["Elastiklik İndeksi (EI = f2² × m^⅔)", vl.ei.toStringAsExponential(2)],
              ["EI (normalize)", vl.eiNormalized.toStringAsFixed(3)],
              ["Karpuz ağırlığı (fotodan tahmin)", "${vl.massKg.toStringAsFixed(1)} kg"],
              ["Birleşik skor", vl.finalScore.toStringAsFixed(3)],
              ["Titreşim cevabı uyarısı", vl.isHollowHeart ? "Zayıf (şüpheli)" : "Normal"],
            ]),
            pw.SizedBox(height: 6),
            pw.Text(
                "Not: Vi-Liquid titreşim ölçümleri bilgi amaçlıdır ve nihai kararı "
                "etkilemez. Karar, doğrulanmış yapay zeka modelinden gelir.",
                style: pw.TextStyle(
                    fontSize: 8,
                    color: _slate,
                    fontStyle: pw.FontStyle.italic)),
            pw.SizedBox(height: 14),
          ],

          // Açıklama
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(10),
            decoration: pw.BoxDecoration(
              color: const PdfColor.fromInt(0xFFFFF8E1),
              borderRadius: pw.BorderRadius.circular(8),
              border: pw.Border.all(color: _amber),
            ),
            child: pw.Text(
              "Bu rapor yardımcı bilgi içindir. Olgunluk tahmini çok modlu "
              "sensör analizi ile üretilmiştir; kesin sonuç için karpuz kesilerek "
              "doğrulanmalıdır. Sistem 19 karpuzluk Qilin veri setinde LOWO "
              "çapraz doğrulamasıyla 2-sınıf %61.5 baseline doğruluğa sahiptir.",
              style: const pw.TextStyle(fontSize: 9, color: _slate),
            ),
          ),

          pw.SizedBox(height: 20),
          pw.Center(
            child: pw.Text(
              "Karpuz Dedektifi — Selçuk Üniversitesi Mühendislik Tasarımı Projesi",
              style: const pw.TextStyle(fontSize: 8, color: _slate),
            ),
          ),
        ],
      ),
    );

    final dir = await getTemporaryDirectory();
    final path =
        "${dir.path}/karpuz_rapor_${DateTime.now().millisecondsSinceEpoch}.pdf";
    final file = File(path);
    await file.writeAsBytes(await doc.save());
    return file;
  }

  pw.Widget _sectionTitle(String t) => pw.Padding(
        padding: const pw.EdgeInsets.only(bottom: 6),
        child: pw.Text(t,
            style: pw.TextStyle(
                fontSize: 13,
                fontWeight: pw.FontWeight.bold,
                color: _green)),
      );

  pw.Widget _kvTable(List<List<String>> rows) {
    return pw.Table(
      border: pw.TableBorder.all(color: _border, width: 0.5),
      columnWidths: const {
        0: pw.FlexColumnWidth(3),
        1: pw.FlexColumnWidth(2),
      },
      children: [
        for (final r in rows)
          pw.TableRow(children: [
            pw.Padding(
              padding: const pw.EdgeInsets.all(7),
              child: pw.Text(r[0], style: const pw.TextStyle(fontSize: 10)),
            ),
            pw.Padding(
              padding: const pw.EdgeInsets.all(7),
              child: pw.Text(r[1],
                  style: pw.TextStyle(
                      fontSize: 10, fontWeight: pw.FontWeight.bold)),
            ),
          ]),
      ],
    );
  }

  pw.Widget _probTable(dynamic fusion) {
    final rows = [
      ["Henüz ham", fusion.pImmature as double, const PdfColor.fromInt(0xFF7CB342)],
      ["Tam kıvamında", fusion.pRipe as double, _green],
      ["Geçmiş olabilir", fusion.pOverripe as double, const PdfColor.fromInt(0xFFEF6C00)],
    ];
    return pw.Column(
      children: [
        for (final r in rows)
          pw.Padding(
            padding: const pw.EdgeInsets.only(bottom: 6),
            child: pw.Row(
              children: [
                pw.SizedBox(
                  width: 110,
                  child: pw.Text(r[0] as String,
                      style: const pw.TextStyle(fontSize: 10)),
                ),
                pw.Expanded(
                  child: pw.Stack(
                    children: [
                      pw.Container(
                        height: 12,
                        decoration: pw.BoxDecoration(
                          color: _border,
                          borderRadius: pw.BorderRadius.circular(6),
                        ),
                      ),
                      pw.Container(
                        height: 12,
                        width: 380 * (r[1] as double).clamp(0.0, 1.0),
                        decoration: pw.BoxDecoration(
                          color: r[2] as PdfColor,
                          borderRadius: pw.BorderRadius.circular(6),
                        ),
                      ),
                    ],
                  ),
                ),
                pw.SizedBox(width: 8),
                pw.SizedBox(
                  width: 40,
                  child: pw.Text(
                      "%${((r[1] as double) * 100).toStringAsFixed(0)}",
                      style: pw.TextStyle(
                          fontSize: 10, fontWeight: pw.FontWeight.bold)),
                ),
              ],
            ),
          ),
      ],
    );
  }
}
