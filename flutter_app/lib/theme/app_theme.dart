import 'package:flutter/material.dart';

/// Centralized palette + reusable decoration helpers for the watermelon app.
class AppTheme {
  // ----- Brand palette -----
  static const Color primary = Color(0xFF1B7F3A); // forest green
  static const Color primaryDark = Color(0xFF0F5524);
  static const Color accent = Color(0xFFE53935); // watermelon red
  static const Color amber = Color(0xFFFFB300);
  static const Color ink = Color(0xFF1B1F1A);
  static const Color slate = Color(0xFF5F6B5C);
  static const Color cardBg = Colors.white;
  static const Color surface = Color(0xFFF6F8F4);
  static const Color border = Color(0xFFE3E8DC);

  // ----- Decorations -----
  static BoxDecoration cardDecoration({
    Color color = Colors.white,
    double radius = 20,
    bool shadow = false,
  }) {
    return BoxDecoration(
      color: color,
      borderRadius: BorderRadius.circular(radius),
      border: Border.all(color: border, width: 1),
      boxShadow: shadow
          ? [
              BoxShadow(
                color: Colors.black.withOpacity(0.04),
                blurRadius: 16,
                offset: const Offset(0, 6),
              )
            ]
          : null,
    );
  }

  static BoxDecoration heroGradient() {
    return BoxDecoration(
      gradient: const LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [primary, primaryDark],
      ),
      borderRadius: BorderRadius.circular(28),
      boxShadow: [
        BoxShadow(
          color: primary.withOpacity(0.28),
          blurRadius: 24,
          offset: const Offset(0, 12),
        )
      ],
    );
  }

  /// Returns the verdict color for a given probability.
  static Color verdictGreen() => primary;
  static Color verdictRed() => accent;
}
