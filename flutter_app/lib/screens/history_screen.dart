import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final history = state.history;
    final stats = state.fieldStats;

    return Scaffold(
      backgroundColor: AppTheme.surface,
      appBar: AppBar(
        title: const Text("Geçmiş ölçümler",
            style: TextStyle(fontWeight: FontWeight.w800)),
        backgroundColor: Colors.white,
        foregroundColor: AppTheme.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
      body: Column(
        children: [
          _buildStatsBar(stats),
          Expanded(
            child: history.isEmpty
                ? _buildEmptyState()
                : ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: history.length,
                    separatorBuilder: (_, __) =>
                        const SizedBox(height: 10),
                    itemBuilder: (_, i) => _HistoryTile(
                      entry: history[i],
                      onFeedback: (gt) => context
                          .read<AppState>()
                          .setGroundTruth(history[i]["id"], gt),
                      onClear: () => context
                          .read<AppState>()
                          .clearGroundTruth(history[i]["id"]),
                      onDelete: () => _confirmDelete(context, history[i]),
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Future<void> _confirmDelete(
      BuildContext context, Map<String, dynamic> entry) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("Kayıt sil?"),
        content: const Text("Bu ölçümü kalıcı olarak silmek istiyor musun?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Vazgeç"),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            style: FilledButton.styleFrom(
                backgroundColor: AppTheme.accent),
            child: const Text("Sil"),
          ),
        ],
      ),
    );
    if (confirm == true && context.mounted) {
      await context.read<AppState>().deleteEntry(entry["id"]);
    }
  }

  Widget _buildStatsBar(Map<String, dynamic> stats) {
    final total = stats["total"] as int? ?? 0;
    final withGt = stats["withGroundTruth"] as int? ?? 0;
    final correct = stats["correct"] as int? ?? 0;
    final acc = stats["accuracy"] as double?;
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.cardDecoration(),
      child: Row(
        children: [
          Expanded(
            child: _StatTile(
              label: "Toplam",
              value: "$total",
              icon: Icons.list_alt_rounded,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: _StatTile(
              label: "Kestin",
              value: "$withGt",
              icon: Icons.assignment_turned_in_rounded,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: _StatTile(
              label: "Doğru tahmin",
              value: acc == null
                  ? "—"
                  : "%${(acc * 100).toStringAsFixed(0)}",
              icon: Icons.verified_rounded,
              highlight: acc != null,
              detail: acc == null ? null : "$correct / $withGt",
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.inbox_rounded,
                size: 64, color: Colors.grey.shade400),
            const SizedBox(height: 12),
            Text("Henüz ölçüm yok",
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: Colors.grey.shade700)),
            const SizedBox(height: 4),
            Text(
              "Bir karpuz test ettiğinde burada görünecek.\nKestiğinde sonucu işaretlemeyi unutma.",
              textAlign: TextAlign.center,
              style: TextStyle(
                  fontSize: 12, color: Colors.grey.shade500, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final bool highlight;
  final String? detail;
  const _StatTile({
    required this.label,
    required this.value,
    required this.icon,
    this.highlight = false,
    this.detail,
  });

  @override
  Widget build(BuildContext context) {
    final color = highlight ? AppTheme.primary : AppTheme.slate;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 4),
        Text(label,
            style: const TextStyle(
                fontSize: 11,
                color: AppTheme.slate,
                fontWeight: FontWeight.w600)),
        const SizedBox(height: 2),
        Text(value,
            style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: color)),
        if (detail != null)
          Text(detail!,
              style: const TextStyle(
                  fontSize: 11, color: AppTheme.slate)),
      ],
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final Map<String, dynamic> entry;
  final void Function(int groundTruth) onFeedback;
  final VoidCallback onClear;
  final VoidCallback onDelete;
  const _HistoryTile({
    required this.entry,
    required this.onFeedback,
    required this.onClear,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final rawVerdict = entry["verdict"] as String? ?? "?";
    final rawLabel = entry["label"] as String? ?? "";
    final confidence = (entry["confidence"] as num?)?.toDouble() ?? 0.0;
    final imagePath = entry["imagePath"] as String?;
    final classId = entry["classId"] as int? ?? -1;
    final groundTruth = entry["groundTruthClass"] as int?;
    final f2 = (entry["f2Hz"] as num?)?.toDouble();
    // Vi-Liquid combined verdict mantığı: classId=1 = "Tam kıvamında" = AL,
    // diğerleri ALMA. (Eski kayıtlarda "Yenir"/"Yenmez" string olabilir,
    // ikisini de hesaba kat.)
    final isEdible = rawVerdict == "Yenir" || rawVerdict == "AL"
        ? true
        : rawVerdict == "Yenmez" || rawVerdict == "ALMA"
            ? false
            : classId == 1;
    final verdict = isEdible ? "AL" : "ALMA";
    // Halk dili etiket
    final label = classId == 0
        ? "Henüz ham"
        : classId == 1
            ? "Tam kıvamında"
            : classId == 2
                ? "Geçmiş olabilir"
                : rawLabel;
    final color = isEdible ? AppTheme.primary : AppTheme.accent;

    final hasGt = groundTruth != null;
    final isCorrect = hasGt &&
        _binary(classId) == _binary(groundTruth);

    Widget thumb;
    if (imagePath != null && File(imagePath).existsSync()) {
      thumb = ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.file(File(imagePath),
            width: 64, height: 64, fit: BoxFit.cover),
      );
    } else {
      thumb = Container(
        width: 64,
        height: 64,
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(
            isEdible
                ? Icons.check_circle_rounded
                : Icons.dangerous_rounded,
            color: color,
            size: 32),
      );
    }

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              thumb,
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.12),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(verdict,
                              style: TextStyle(
                                  color: color,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w800)),
                        ),
                        const SizedBox(width: 8),
                        Text("%${(confidence * 100).toStringAsFixed(0)}",
                            style: TextStyle(
                                fontSize: 12,
                                color: AppTheme.slate,
                                fontWeight: FontWeight.w700)),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(label,
                        style: const TextStyle(
                            fontSize: 15, fontWeight: FontWeight.w700)),
                    const SizedBox(height: 2),
                    Text(
                      "${_formatTimestamp(entry["timestamp"] as String?)}"
                      "${f2 != null ? "  ·  f2=${f2.toStringAsFixed(0)}Hz" : ""}",
                      style: const TextStyle(
                          fontSize: 11, color: AppTheme.slate),
                    ),
                  ],
                ),
              ),
              IconButton(
                tooltip: "Sil",
                icon: const Icon(Icons.delete_outline_rounded,
                    size: 20, color: AppTheme.slate),
                onPressed: onDelete,
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (hasGt)
            _buildGroundTruthBadge(isCorrect, groundTruth!)
          else
            _buildFeedbackButton(context),
        ],
      ),
    );
  }

  int _binary(int c) => c == 1 ? 1 : 0;

  String _formatTimestamp(String? iso) {
    if (iso == null) return "";
    try {
      final dt = DateTime.parse(iso);
      final y = dt.year;
      final m = dt.month.toString().padLeft(2, '0');
      final d = dt.day.toString().padLeft(2, '0');
      final hh = dt.hour.toString().padLeft(2, '0');
      final mm = dt.minute.toString().padLeft(2, '0');
      return "$d.$m.$y · $hh:$mm";
    } catch (_) {
      return "";
    }
  }

  Widget _buildGroundTruthBadge(bool isCorrect, int gt) {
    final color = isCorrect ? AppTheme.primary : AppTheme.accent;
    final label = isCorrect ? "Doğru tahmin" : "Yanlış tahmin";
    final gtLabel = gt == 0
        ? "Gerçek: Henüz hamdı"
        : gt == 1
            ? "Gerçek: Tam kıvamındaydı"
            : "Gerçek: İçi geçmişti";
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(
              isCorrect
                  ? Icons.check_circle_rounded
                  : Icons.cancel_rounded,
              color: color,
              size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: color)),
                Text(gtLabel,
                    style: const TextStyle(
                        fontSize: 11, color: AppTheme.slate)),
              ],
            ),
          ),
          TextButton(
            style: TextButton.styleFrom(
                foregroundColor: AppTheme.slate, minimumSize: Size.zero),
            onPressed: onClear,
            child: const Text("Düzenle",
                style: TextStyle(fontSize: 11)),
          ),
        ],
      ),
    );
  }

  Widget _buildFeedbackButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppTheme.primary,
          side: BorderSide(color: AppTheme.primary.withOpacity(0.5)),
          padding: const EdgeInsets.symmetric(vertical: 10),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12)),
        ),
        icon: const Icon(Icons.fact_check_rounded, size: 18),
        label: const Text("Kestin mi? Sonucu işaretle",
            style: TextStyle(
                fontSize: 13, fontWeight: FontWeight.w700)),
        onPressed: () => _showFeedbackDialog(context),
      ),
    );
  }

  void _showFeedbackDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 36),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius:
              BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 16),
            const Text("Kestiğinde nasıldı?",
                style: TextStyle(
                    fontSize: 17, fontWeight: FontWeight.w800)),
            const SizedBox(height: 4),
            const Text(
              "Tahmin doğru muydu öğrenmemize yardım et.",
              style: TextStyle(fontSize: 12, color: AppTheme.slate),
            ),
            const SizedBox(height: 20),
            _gtOption(context,
                color: const Color(0xFF7CB342),
                icon: Icons.eco_rounded,
                title: "Henüz hamdı",
                subtitle: "Sertti, tatlı değildi",
                onTap: () {
                  onFeedback(0);
                  Navigator.pop(context);
                }),
            const SizedBox(height: 10),
            _gtOption(context,
                color: AppTheme.primary,
                icon: Icons.verified_rounded,
                title: "Tam kıvamındaydı",
                subtitle: "Olgun ve lezzetliydi",
                onTap: () {
                  onFeedback(1);
                  Navigator.pop(context);
                }),
            const SizedBox(height: 10),
            _gtOption(context,
                color: const Color(0xFFEF6C00),
                icon: Icons.warning_rounded,
                title: "İçi geçmişti",
                subtitle: "Boşluk vardı, fazla yumuşaktı",
                onTap: () {
                  onFeedback(2);
                  Navigator.pop(context);
                }),
          ],
        ),
      ),
    );
  }

  Widget _gtOption(
    BuildContext context, {
    required Color color,
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: color.withOpacity(0.08),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withOpacity(0.25)),
          ),
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 24),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w800,
                            color: color)),
                    const SizedBox(height: 2),
                    Text(subtitle,
                        style: const TextStyle(
                            fontSize: 12, color: AppTheme.slate)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded,
                  color: AppTheme.slate),
            ],
          ),
        ),
      ),
    );
  }
}
