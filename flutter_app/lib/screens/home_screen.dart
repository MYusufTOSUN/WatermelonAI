import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import 'capture/capture_wizard_screen.dart';
import 'debug_validate_screen.dart';
import 'history_screen.dart';
import 'self_test_screen.dart';

/// Sade ana ekran — halk dili.
///
/// Kullanıcının ilk gözüne çarpan tek bir büyük "Karpuz testini başlat"
/// butonu + geçmiş + "Nasıl çalışır" 3 madde. Teknik testler ⋯ menüsünde.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final stats = state.fieldStats;
    final total = stats["total"] as int? ?? 0;
    final correct = stats["correct"] as int? ?? 0;
    final acc = stats["accuracy"] as double?;

    return Scaffold(
      backgroundColor: AppTheme.surface,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildHeader(context),
              const SizedBox(height: 24),
              _buildPrimaryCard(context),
              const SizedBox(height: 14),
              _buildHistoryCard(context, total, correct, acc),
              const SizedBox(height: 28),
              const _SectionTitle("Nasıl çalışır?"),
              const SizedBox(height: 10),
              const _HowItWorksRow(
                icon: Icons.photo_camera_rounded,
                step: "1",
                text: "Karpuzun fotoğrafını çek",
              ),
              const SizedBox(height: 10),
              const _HowItWorksRow(
                icon: Icons.graphic_eq_rounded,
                step: "2",
                text: "Karpuza parmağınla vur, ses kaydı",
              ),
              const SizedBox(height: 10),
              const _HowItWorksRow(
                icon: Icons.vibration_rounded,
                step: "3",
                text: "Telefonu karpuza yasla, biraz beklet",
              ),
              const SizedBox(height: 20),
              _buildPrivacyChip(),
            ],
          ),
        ),
      ),
    );
  }

  // ─────────────────────────── Header ────────────────────────────────
  Widget _buildHeader(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 46,
          height: 46,
          decoration: BoxDecoration(
            color: AppTheme.primary,
            borderRadius: BorderRadius.circular(14),
          ),
          child: const Icon(Icons.eco_rounded,
              color: Colors.white, size: 26),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Karpuz Dedektifi",
                  style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.ink)),
              Text("Olgun mu, değil mi anlar",
                  style: TextStyle(
                      fontSize: 12, color: AppTheme.slate)),
            ],
          ),
        ),
        _MoreMenuButton(),
      ],
    );
  }

  // ─────────────────────────── Ana büyük buton ───────────────────────
  Widget _buildPrimaryCard(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const CaptureWizardScreen()),
      ),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: AppTheme.heroGradient(),
        child: Row(
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.18),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Icon(Icons.center_focus_strong_rounded,
                  color: Colors.white, size: 36),
            ),
            const SizedBox(width: 16),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Karpuz testini başlat",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.w800,
                      height: 1.1,
                    ),
                  ),
                  SizedBox(height: 6),
                  Text(
                    "3 adım · yaklaşık 1 dakika",
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_rounded,
                color: Colors.white, size: 28),
          ],
        ),
      ),
    );
  }

  // ─────────────────────────── Geçmiş kart ───────────────────────────
  Widget _buildHistoryCard(
      BuildContext context, int total, int correct, double? acc) {
    final subtitle = total == 0
        ? "Henüz ölçüm yok"
        : acc == null
            ? "$total ölçüm var"
            : "$total ölçüm · ${(acc * 100).toStringAsFixed(0)}/100 doğru tahmin";
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const HistoryScreen()),
      ),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: AppTheme.cardDecoration(),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.10),
                borderRadius: BorderRadius.circular(14),
              ),
              child: const Icon(Icons.history_rounded,
                  color: AppTheme.primary, size: 26),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Geçmiş ölçümlerin",
                      style: TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w800)),
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
    );
  }

  // ─────────────────────────── Privacy chip ──────────────────────────
  Widget _buildPrivacyChip() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.primary.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.primary.withOpacity(0.2)),
      ),
      child: const Row(
        children: [
          Icon(Icons.shield_moon_rounded,
              color: AppTheme.primary, size: 18),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              "Her şey telefonunda çalışır. İnternete bir şey göndermez.",
              style: TextStyle(
                  fontSize: 12, color: AppTheme.ink, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}

// ───────────────────────────── ⋯ Menüsü ───────────────────────────────

class _MoreMenuButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      icon: const Icon(Icons.more_horiz_rounded, size: 28),
      tooltip: "Diğer",
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14)),
      onSelected: (v) {
        switch (v) {
          case "onboarding":
            context.read<AppState>().resetOnboarding();
            break;
          case "self_test":
            Navigator.push(context,
                MaterialPageRoute(builder: (_) => const SelfTestScreen()));
            break;
          case "debug_validate":
            Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) => const DebugValidateScreen()));
            break;
        }
      },
      itemBuilder: (_) => [
        const PopupMenuItem(
          value: "onboarding",
          child: Row(
            children: [
              Icon(Icons.play_circle_outline_rounded,
                  size: 18, color: AppTheme.slate),
              SizedBox(width: 8),
              Text("Tanıtımı tekrar oynat"),
            ],
          ),
        ),
        const PopupMenuItem(
          value: "self_test",
          child: Row(
            children: [
              Icon(Icons.fact_check_outlined,
                  size: 18, color: AppTheme.slate),
              SizedBox(width: 8),
              Text("Geliştirici: hızlı test"),
            ],
          ),
        ),
        const PopupMenuItem(
          value: "debug_validate",
          child: Row(
            children: [
              Icon(Icons.science_outlined,
                  size: 18, color: AppTheme.slate),
              SizedBox(width: 8),
              Text("Geliştirici: doğrulama"),
            ],
          ),
        ),
      ],
    );
  }
}

// ───────────────────────────── Section Title ──────────────────────────

class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Text(text,
          style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w800,
              color: AppTheme.slate,
              letterSpacing: 0.3)),
    );
  }
}

class _HowItWorksRow extends StatelessWidget {
  final IconData icon;
  final String step;
  final String text;
  const _HowItWorksRow({
    required this.icon,
    required this.step,
    required this.text,
  });
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: AppTheme.cardDecoration(),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.10),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Center(
              child: Text(step,
                  style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.primary)),
            ),
          ),
          const SizedBox(width: 12),
          Icon(icon, color: AppTheme.primary, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(text,
                style: const TextStyle(
                    fontSize: 14, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}
