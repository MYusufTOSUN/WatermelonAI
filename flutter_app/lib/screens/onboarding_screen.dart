import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<_Slide> _slides = const [
    _Slide(
      icon: Icons.photo_camera_rounded,
      title: "Karpuzun fotoğrafını çek",
      subtitle:
          "Karpuzun nasıl göründüğüne bakıyoruz. Rengi, kabuğu, lekesi.",
    ),
    _Slide(
      icon: Icons.graphic_eq_rounded,
      title: "Karpuza vur, sesi kaydet",
      subtitle:
          "Olgun karpuz tok, ham karpuz parlak ses verir. Mikrofon ile dinleriz.",
    ),
    _Slide(
      icon: Icons.vibration_rounded,
      title: "Telefonu karpuza yasla",
      subtitle:
          "Telefon hafif titrer, karpuzdan dönen cevabı ölçer. İçi boş karpuzlar burada yakalanır.",
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: scheme.background,
      body: SafeArea(
        child: Column(
          children: [
            Align(
              alignment: Alignment.topRight,
              child: TextButton(
                onPressed: () => _finish(context),
                child: Text("Atla",
                    style: TextStyle(
                        color: scheme.secondary, fontWeight: FontWeight.bold)),
              ),
            ),
            Expanded(
              child: PageView.builder(
                controller: _pageController,
                onPageChanged: (v) => setState(() => _currentPage = v),
                itemCount: _slides.length,
                itemBuilder: (ctx, i) => _SlideView(slide: _slides[i]),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: List.generate(
                        _slides.length, (i) => _dot(i, scheme)),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      if (_currentPage == _slides.length - 1) {
                        _finish(context);
                      } else {
                        _pageController.nextPage(
                          duration: const Duration(milliseconds: 300),
                          curve: Curves.easeIn,
                        );
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: scheme.primary,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 28, vertical: 14),
                    ),
                    child: Text(
                      _currentPage == _slides.length - 1 ? "Başla" : "İleri",
                      style: const TextStyle(
                          fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                  )
                ],
              ),
            )
          ],
        ),
      ),
    );
  }

  void _finish(BuildContext context) {
    context.read<AppState>().completeOnboarding();
  }

  Widget _dot(int index, ColorScheme scheme) {
    final active = _currentPage == index;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      margin: const EdgeInsets.only(right: 8),
      height: 8,
      width: active ? 24 : 8,
      decoration: BoxDecoration(
        color: active ? scheme.primary : Colors.grey.withOpacity(0.4),
        borderRadius: BorderRadius.circular(4),
      ),
    );
  }
}

class _Slide {
  final IconData icon;
  final String title;
  final String subtitle;
  const _Slide({
    required this.icon,
    required this.title,
    required this.subtitle,
  });
}

class _SlideView extends StatelessWidget {
  final _Slide slide;
  const _SlideView({required this.slide});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 180,
            height: 180,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [scheme.primary, scheme.primary.withOpacity(0.6)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(44),
              boxShadow: [
                BoxShadow(
                  color: scheme.primary.withOpacity(0.3),
                  blurRadius: 30,
                  offset: const Offset(0, 12),
                )
              ],
            ),
            child: Icon(slide.icon, size: 90, color: Colors.white),
          ),
          const SizedBox(height: 48),
          Text(slide.title,
              style: const TextStyle(
                  fontSize: 28, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center),
          const SizedBox(height: 16),
          Text(slide.subtitle,
              style: const TextStyle(
                  fontSize: 16, color: Colors.black87, height: 1.5),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }
}
