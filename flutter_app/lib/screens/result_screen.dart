import 'dart:io';

import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';

import '../models/multimodal_result.dart';
import '../services/pdf_report_service.dart';
import '../theme/app_theme.dart';
import 'capture/capture_wizard_screen.dart';
import 'home_screen.dart';

/// Sade sonuç ekranı (halk diline uygun).
///
/// Görünür kısım:
///   - Tek BÜYÜK karar kartı (AL / ALMA + tek satır neden)
///   - Karpuz fotoğrafı
///   - "Sonucu işaretle" küçük buton (geçmişe etiket koymak için)
///
/// Detaylar (varsayılan kapalı):
///   - Ölçüm değerleri (rezonans, içi boş riski, temas kalitesi)
///   - Olasılıklar (Tam kıvamında/Henüz ham/Geçmiş olabilir)
///   - Vi-Liquid (ileri seviye)
class ResultScreen extends StatefulWidget {
  final MultimodalResult result;
  final String imagePath;
  final String? audioPath;

  const ResultScreen({
    super.key,
    required this.result,
    required this.imagePath,
    this.audioPath,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _showDetails = false;

  MultimodalResult get result => widget.result;
  String get imagePath => widget.imagePath;

  bool _pdfBusy = false;

  Future<void> _shareAudio() async {
    final path = widget.audioPath;
    if (path == null || !File(path).existsSync()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Ses kaydı bulunamadı")),
      );
      return;
    }
    await Share.shareXFiles(
      [XFile(path)],
      text: "Karpuz vuruş sesi kaydı (analiz için)",
    );
  }

  Future<void> _sharePdf() async {
    if (_pdfBusy) return;
    setState(() => _pdfBusy = true);
    try {
      final file = await PdfReportService()
          .generate(result: result, imagePath: imagePath);
      await Share.shareXFiles(
        [XFile(file.path)],
        text: "Karpuz analiz raporu",
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("PDF oluşturulamadı: $e")),
        );
      }
    } finally {
      if (mounted) setState(() => _pdfBusy = false);
    }
  }

  // Halk-dostu birleşik sonuç: Vi-Liquid varsa onu öncele, yoksa ML fusion
  bool get _isEdible =>
      result.viLiquid?.isEdible ?? result.fusion.isEdible;
  String get _verdictWord => _isEdible ? "AL" : "ALMA";
  String get _verdictHeadline =>
      result.viLiquid?.simpleHeadline ?? result.fusion.simpleHeadline;
  String get _verdictReason =>
      result.viLiquid?.simpleReason ?? result.fusion.simpleReason;
  String get _verdictLabel =>
      result.viLiquid?.simpleLabel ?? result.fusion.simpleLabel;
  double get _verdictConfidence => result.viLiquid?.verdictConfidence ??
      result.fusion.edibleConfidence;
  Color get _verdictColor =>
      result.viLiquid?.verdictColor ?? result.fusion.verdictColor;
  IconData get _verdictIcon {
    if (result.viLiquid?.isHollowHeart ?? false) {
      return Icons.warning_rounded;
    }
    return _isEdible ? Icons.check_circle_rounded : Icons.dangerous_rounded;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surface,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(context),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (result.isLowQuality) ...[
                      _buildQualityWarning(),
                      const SizedBox(height: 12),
                    ],
                    _buildBigVerdict(),
                    const SizedBox(height: 16),
                    _buildImage(),
                    const SizedBox(height: 16),
                    _buildSimpleStatusCard(),
                    const SizedBox(height: 12),
                    _buildPdfButton(),
                    const SizedBox(height: 12),
                    _buildDetailsToggle(),
                    if (_showDetails) ...[
                      const SizedBox(height: 12),
                      _buildOlcumKartlari(),
                      const SizedBox(height: 12),
                      _buildOlasiliklarKart(),
                      if (result.viLiquid != null) ...[
                        const SizedBox(height: 12),
                        _buildViLiquidKart(),
                      ],
                      if (widget.audioPath != null) ...[
                        const SizedBox(height: 12),
                        _buildShareAudioButton(),
                      ],
                    ],
                    const SizedBox(height: 14),
                    _buildDisclaimer(),
                  ],
                ),
              ),
            ),
            _buildActions(context),
          ],
        ),
      ),
    );
  }

  // ───────────────────────────── App Bar ─────────────────────────────
  Widget _buildAppBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.close_rounded, size: 28),
            onPressed: () => Navigator.pushAndRemoveUntil(
              context,
              MaterialPageRoute(builder: (_) => const HomeScreen()),
              (_) => false,
            ),
          ),
          const SizedBox(width: 4),
          const Text("Sonuç",
              style: TextStyle(
                  fontSize: 18, fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }

  // ─────────────────────────── BÜYÜK Verdict ──────────────────────────
  Widget _buildBigVerdict() {
    final color = _verdictColor;
    return Container(
      padding: const EdgeInsets.fromLTRB(22, 24, 22, 22),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [color, color.withOpacity(0.78)],
        ),
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.32),
            blurRadius: 26,
            offset: const Offset(0, 14),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(_verdictIcon, color: Colors.white, size: 64),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _verdictWord,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 44,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1.5,
                        height: 1.0,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _verdictHeadline,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.20),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                const Icon(Icons.psychology_alt_rounded,
                    color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    "%${(_verdictConfidence * 100).toStringAsFixed(0)} eminim",
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────── Foto ──────────────────────────────────
  Widget _buildImage() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: Image.file(
        File(imagePath),
        height: 180,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    );
  }

  // ─────────────────────────── Sade Status Kartı ─────────────────────
  Widget _buildSimpleStatusCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.cardDecoration(),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: _verdictColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(_verdictIcon, color: _verdictColor, size: 26),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _verdictLabel,
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: _verdictColor,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _verdictReason,
                  style: const TextStyle(
                      fontSize: 13,
                      color: AppTheme.slate,
                      height: 1.4),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────── Detay Aç/Kapa ─────────────────────────
  Widget _buildQualityWarning() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF3E0),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.amber, width: 1.2),
      ),
      child: Row(
        children: [
          const Icon(Icons.mic_off_rounded,
              color: Color(0xFFE65100), size: 26),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Vuruş kaydı zayıf",
                    style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w800,
                        color: Color(0xFFE65100))),
                SizedBox(height: 3),
                Text(
                  "Ses çok düşük yakalandı, sonuç güvenilir olmayabilir. "
                  "Mikrofonu karpuza yaklaştırıp parmak boğumuyla daha sert "
                  "vur ve tekrar dene.",
                  style: TextStyle(
                      fontSize: 12, color: Color(0xFF5D4037), height: 1.35),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPdfButton() {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          backgroundColor: AppTheme.primary.withOpacity(0.10),
          foregroundColor: AppTheme.primary,
          elevation: 0,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14)),
        ),
        icon: _pdfBusy
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(
                    strokeWidth: 2.2, color: AppTheme.primary),
              )
            : const Icon(Icons.picture_as_pdf_rounded, size: 20),
        label: Text(
          _pdfBusy ? "Hazırlanıyor..." : "Raporu PDF olarak paylaş",
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
        ),
        onPressed: _pdfBusy ? null : _sharePdf,
      ),
    );
  }

  Widget _buildShareAudioButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppTheme.slate,
          side: BorderSide(color: AppTheme.border),
          padding: const EdgeInsets.symmetric(vertical: 12),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12)),
        ),
        icon: const Icon(Icons.ios_share_rounded, size: 18),
        label: const Text("Ses kaydını paylaş / kaydet",
            style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        onPressed: _shareAudio,
      ),
    );
  }

  Widget _buildDetailsToggle() {
    return TextButton.icon(
      style: TextButton.styleFrom(
        foregroundColor: AppTheme.slate,
        minimumSize: const Size.fromHeight(40),
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
      ),
      icon: Icon(
        _showDetails
            ? Icons.keyboard_arrow_up_rounded
            : Icons.keyboard_arrow_down_rounded,
        size: 22,
      ),
      label: Text(
        _showDetails ? "Detayları gizle" : "Daha fazla detay",
        style: const TextStyle(
            fontSize: 13, fontWeight: FontWeight.w700),
      ),
      onPressed: () => setState(() => _showDetails = !_showDetails),
    );
  }

  // ─────────────────────────── Ölçüm Kartları (detay) ────────────────
  // NOT: "İçi boş riski" (basitleştirilmiş canlı HH skoru) gösterilmiyor —
  // güvenilmez bir proxy olup karara dahil değildir; %92 gibi yüksek değerler
  // doğru "AL" kararıyla çelişip kullanıcıyı yanıltıyordu. İçi geçmiş tespiti
  // doğrulanmış yapay zeka modeli tarafından "Geçmiş olabilir" olasılığıyla
  // zaten yapılır.
  Widget _buildOlcumKartlari() {
    final contact = result.contactQuality;
    final q = result.recordingQuality;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("Ölçüm değerleri",
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.slate)),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _MiniMetric(
                  icon: Icons.graphic_eq_rounded,
                  label: "Ses tonu",
                  value: "${result.f2Hz.toStringAsFixed(0)} Hz",
                  hint: "vuruş sesi rezonansı",
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MiniMetric(
                  icon: Icons.music_note_rounded,
                  label: "Vuruş enerjisi",
                  value: result.signalRms.toStringAsFixed(3),
                  hint: "ses gücü",
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _MiniMetric(
                  icon: Icons.touch_app_rounded,
                  label: "Temas kalitesi",
                  value: "%${(contact * 100).toStringAsFixed(0)}",
                  hint: contact > 0.6 ? "İyi" : "Daha sıkı tut",
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MiniMetric(
                  icon: Icons.verified_rounded,
                  label: "Kayıt kalitesi",
                  value: "%${(q * 100).toStringAsFixed(0)}",
                  hint: q < 0.3 ? "Zayıf — tekrar dene" : "İyi",
                  warn: q < 0.3,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ─────────────────────────── Olasılıklar ───────────────────────────
  /// Vi-Liquid varsa onun BİRLEŞİK olasılıklarını gösterir (üst kart ile
  /// tutarlı). Yoksa ham DNN çıktısı görünür. Bu sayede Hollow Heart
  /// tetiği aktifken kullanıcı "ALMA" + "İçi Geçmiş %80" şeklinde
  /// tutarlı sayılar görür.
  Widget _buildOlasiliklarKart() {
    final vl = result.viLiquid;
    final pImmature = vl?.combinedPImmature ?? result.fusion.pImmature;
    final pRipe = vl?.combinedPRipe ?? result.fusion.pRipe;
    final pOverripe = vl?.combinedPOverripe ?? result.fusion.pOverripe;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("Olasılıklar",
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.slate)),
          const SizedBox(height: 10),
          _ProbRow(
              label: "Henüz ham",
              value: pImmature,
              color: const Color(0xFF7CB342)),
          const SizedBox(height: 8),
          _ProbRow(
              label: "Tam kıvamında",
              value: pRipe,
              color: AppTheme.primary),
          const SizedBox(height: 8),
          _ProbRow(
              label: "Geçmiş olabilir",
              value: pOverripe,
              color: const Color(0xFFEF6C00)),
        ],
      ),
    );
  }

  // ─────────────────────────── Vi-Liquid (detay) ─────────────────────
  // NOT: "İçi boş alarmı / şüpheli" uyarıları kaldırıldı — bunlar telefonda
  // güvenilmez olan SRR f2 ve basitleştirilmiş HH skoruna dayanıyordu ve
  // doğru kararla çelişip kullanıcıyı yanıltıyordu. Burada yalnızca tarafsız
  // fiziksel ölçümler (titreşim yankı tonu, ağırlık) bilgi amaçlı gösterilir.
  Widget _buildViLiquidKart() {
    final vl = result.viLiquid!;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.vibration_rounded,
                  color: AppTheme.primary, size: 18),
              SizedBox(width: 6),
              Expanded(
                child: Text("Titreşim ölçümleri (bilgi amaçlı)",
                    style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.slate)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _MiniMetric(
                  icon: Icons.graphic_eq_rounded,
                  label: "Yankı tonu",
                  value: "${vl.f2Hz.toStringAsFixed(0)} Hz",
                  hint: "titreşim cevabı",
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MiniMetric(
                  icon: Icons.fitness_center_rounded,
                  label: "Karpuz ağırlığı",
                  value: "${vl.massKg.toStringAsFixed(1)} kg",
                  hint: "fotodan tahmin",
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ─────────────────────────── Disclaimer ────────────────────────────
  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E1),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFFE082)),
      ),
      child: const Row(
        children: [
          Icon(Icons.info_outline_rounded,
              color: Color(0xFFE65100), size: 18),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              "Bu tahmin yardımcı bilgi içindir. Kesin olmak için karpuzu kestiğinde sonucu işaretleyebilirsin.",
              style: TextStyle(fontSize: 11, color: Color(0xFF5D4037)),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────── Buton Bar ─────────────────────────────
  Widget _buildActions(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: AppTheme.border)),
      ),
      child: Row(
        children: [
          Expanded(
            child: OutlinedButton.icon(
              onPressed: () => Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (_) => const HomeScreen()),
                (_) => false,
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.slate,
                minimumSize: const Size.fromHeight(54),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16)),
              ),
              icon: const Icon(Icons.home_rounded),
              label: const Text("Ana ekran",
                  style: TextStyle(fontWeight: FontWeight.w700)),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            flex: 2,
            child: FilledButton.icon(
              onPressed: () => Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (_) => const CaptureWizardScreen()),
              ),
              style: FilledButton.styleFrom(
                backgroundColor: AppTheme.primary,
                foregroundColor: Colors.white,
                minimumSize: const Size.fromHeight(54),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16)),
              ),
              icon: const Icon(Icons.refresh_rounded),
              label: const Text("Yeni karpuz",
                  style: TextStyle(
                      fontSize: 15, fontWeight: FontWeight.w800)),
            ),
          ),
        ],
      ),
    );
  }
}

// ───────────────────────────── Yardımcı widget'lar ────────────────────

class _MiniMetric extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final String hint;
  final bool warn;
  const _MiniMetric({
    required this.icon,
    required this.label,
    required this.value,
    required this.hint,
    this.warn = false,
  });
  @override
  Widget build(BuildContext context) {
    final color = warn ? AppTheme.accent : AppTheme.primary;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(height: 4),
          Text(label,
              style: const TextStyle(
                  fontSize: 11,
                  color: AppTheme.slate,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 2),
          Text(value,
              style: const TextStyle(
                  fontSize: 15, fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(hint,
              style: TextStyle(
                  fontSize: 10, color: color.withOpacity(0.85))),
        ],
      ),
    );
  }
}

class _ProbRow extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  const _ProbRow({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label,
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600)),
            Text("%${(value * 100).toStringAsFixed(0)}",
                style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                    color: color)),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: value.clamp(0.0, 1.0),
            minHeight: 7,
            backgroundColor: AppTheme.border,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ],
    );
  }
}
