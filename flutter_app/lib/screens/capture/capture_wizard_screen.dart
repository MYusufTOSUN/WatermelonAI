import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import 'package:flutter/foundation.dart';

import '../../providers/app_state.dart';
import '../../services/audio_recorder_service.dart';
import '../../services/feature_extractors/isolate_workers.dart';
import '../../services/multimodal_analyzer.dart';
import '../../services/sensor_recorder_service.dart';
import '../../services/vi_liquid/mass_estimator.dart';
import '../../services/vi_liquid/vibration_service.dart';
import '../../theme/app_theme.dart';
import '../result_screen.dart';

enum _Step { photo, audio, haptic, analyzing }

class CaptureWizardScreen extends StatefulWidget {
  const CaptureWizardScreen({super.key});

  @override
  State<CaptureWizardScreen> createState() => _CaptureWizardScreenState();
}

class _CaptureWizardScreenState extends State<CaptureWizardScreen> {
  _Step _step = _Step.photo;

  // Camera
  CameraController? _camera;
  List<CameraDescription>? _cameras;
  bool _cameraReady = false;
  String? _errorMessage;

  // Captured data
  String? _photoPath;
  String? _audioPath;
  SensorRecording? _sensors;

  // Audio/Sensor/Vibration recorders
  final AudioRecorderService _audio = AudioRecorderService();
  final SensorRecorderService _sensorRec = SensorRecorderService();
  final VibrationService _vibration = VibrationService();

  // Recording state
  bool _audioRecording = false;
  int _audioCountdown = 0;
  Timer? _audioTimer;

  bool _sensorRecording = false;
  int _sensorCountdown = 0;
  Timer? _sensorTimer;

  // Vi-Liquid haptic params
  double _massKg = 4.0; // user-adjustable slider, 1-10 kg
  MassEstimate? _massEstimate; // photo-derived (auto, can be overridden)
  bool _massEstimating = false;

  final MultimodalAnalyzer _analyzer = MultimodalAnalyzer();

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      // Permissions
      final cam = await Permission.camera.request();
      final mic = await Permission.microphone.request();
      if (!cam.isGranted || !mic.isGranted) {
        setState(() {
          _errorMessage =
              "Kamera ve mikrofon izni vermeden multimodal analiz yapılamaz.";
        });
        return;
      }

      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        setState(() => _errorMessage = "Cihazda kamera bulunamadı.");
        return;
      }
      _camera = CameraController(
        _cameras!.first,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _camera!.initialize();

      // Init vibration
      unawaited(_vibration.init());

      // Preload models in background
      unawaited(_analyzer.loadModels());

      if (!mounted) return;
      setState(() => _cameraReady = true);
    } catch (e) {
      setState(() => _errorMessage = "Başlatma hatası: $e");
    }
  }

  @override
  void dispose() {
    _audioTimer?.cancel();
    _sensorTimer?.cancel();
    _camera?.dispose();
    _audio.dispose();
    _analyzer.dispose();
    super.dispose();
  }

  // -------------------- STEP 1: PHOTO --------------------
  Future<void> _capturePhoto() async {
    if (_camera == null || !_camera!.value.isInitialized) return;
    try {
      final shot = await _camera!.takePicture();
      final dir = await getApplicationDocumentsDirectory();
      final samplesDir = Directory("${dir.path}/samples");
      if (!await samplesDir.exists()) {
        await samplesDir.create(recursive: true);
      }
      final saved =
          "${samplesDir.path}/wm_photo_${DateTime.now().millisecondsSinceEpoch}.jpg";
      await File(shot.path).copy(saved);
      if (!mounted) return;
      setState(() {
        _photoPath = saved;
        _step = _Step.audio;
        _massEstimating = true;
      });
      // Auto-estimate mass from photo in background (isolate)
      try {
        final bytes = await File(saved).readAsBytes();
        final est = await compute(massEstimateIsolate, bytes);
        if (!mounted) return;
        setState(() {
          _massEstimate = est;
          _massKg = est.massKg;
          _massEstimating = false;
        });
      } catch (e) {
        if (!mounted) return;
        setState(() => _massEstimating = false);
      }
    } catch (e) {
      _showError("Fotoğraf hatası: $e");
    }
  }

  // -------------------- STEP 2: AUDIO --------------------
  Future<void> _startAudio() async {
    if (_audioRecording) return;
    try {
      await _audio.start();
      setState(() {
        _audioRecording = true;
        _audioCountdown = 3;
      });
      _audioTimer = Timer.periodic(const Duration(seconds: 1), (t) async {
        if (_audioCountdown <= 1) {
          t.cancel();
          await _stopAudio();
        } else {
          setState(() => _audioCountdown -= 1);
        }
      });
    } catch (e) {
      _showError("Mikrofon hatası: $e");
    }
  }

  Future<void> _stopAudio() async {
    if (!_audioRecording) return;
    final path = await _audio.stop();
    if (!mounted) return;
    setState(() {
      _audioRecording = false;
      _audioPath = path;
      if (path != null) _step = _Step.haptic;
    });
  }

  // -------------------- STEP 3: HAPTIC (Vi-Liquid) --------------------
  Future<void> _startSensor() async {
    if (_sensorRecording) return;

    // Start vibration first so accelerometer captures reflected response
    await _vibration.startContinuous(durationMs: 3000);

    await _sensorRec.start();
    setState(() {
      _sensorRecording = true;
      _sensorCountdown = 3;
    });
    _sensorTimer = Timer.periodic(const Duration(seconds: 1), (t) async {
      if (_sensorCountdown <= 1) {
        t.cancel();
        await _stopSensor();
      } else {
        setState(() => _sensorCountdown -= 1);
      }
    });
  }

  Future<void> _stopSensor() async {
    if (!_sensorRecording) return;
    await _vibration.cancel();
    final rec = await _sensorRec.stop();
    if (!mounted) return;
    setState(() {
      _sensorRecording = false;
      _sensors = rec;
    });
    _runAnalysis();
  }

  // -------------------- STEP 4: ANALYZE --------------------
  Future<void> _runAnalysis() async {
    if (_photoPath == null || _audioPath == null || _sensors == null) return;
    setState(() => _step = _Step.analyzing);

    try {
      await _analyzer.loadModels();
      final result = await _analyzer.analyze(
        photo: File(_photoPath!),
        audioWavPath: _audioPath!,
        sensors: _sensors!,
        massKg: _massKg,
      );

      if (!mounted) return;
      // Vi-Liquid'in BİRLEŞİK kararını öncele (Hollow Heart tetiği vs.
      // dahil). Yoksa DNN ham çıktısına düş.
      final vl = result.viLiquid;
      final finalClassId = vl?.finalClassId ?? result.fusion.classId;
      final finalLabel = vl?.simpleLabel ?? result.fusion.detailLabel;
      final finalVerdict = vl?.verdict ?? result.fusion.edibleVerdict;
      final finalConfidence =
          vl?.verdictConfidence ?? result.fusion.edibleConfidence;
      final pImm = vl?.combinedPImmature ?? result.fusion.pImmature;
      final pRipe = vl?.combinedPRipe ?? result.fusion.pRipe;
      final pOver = vl?.combinedPOverripe ?? result.fusion.pOverripe;

      final entryId = DateTime.now().millisecondsSinceEpoch.toString();
      await context.read<AppState>().addHistoryResult({
        "id": entryId,
        "classId": finalClassId,
        "label": finalLabel,
        "verdict": finalVerdict,
        "confidence": finalConfidence,
        "pImmature": pImm,
        "pRipe": pRipe,
        "pOverripe": pOver,
        "isHollowHeart": vl?.isHollowHeart ?? false,
        "rawDnnClassId": result.fusion.classId,
        "f2Hz": result.f2Hz,
        "viLiquidF2Hz": vl?.f2Hz,
        "hhScore": result.hhScore,
        "contactQuality": result.contactQuality,
        "imagePath": _photoPath!,
        "audioPath": _audioPath!,
        "timestamp": DateTime.now().toIso8601String(),
      });

      await Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            result: result,
            imagePath: _photoPath!,
            audioPath: _audioPath,
          ),
        ),
      );
    } catch (e) {
      _showError("Analiz hatası: $e");
      if (mounted) setState(() => _step = _Step.haptic);
    }
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: Colors.red.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  // -------------------- BUILD --------------------
  @override
  Widget build(BuildContext context) {
    if (_errorMessage != null) {
      return _buildErrorScreen(_errorMessage!);
    }
    return Scaffold(
      backgroundColor: AppTheme.surface,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            _buildStepIndicator(),
            const SizedBox(height: 12),
            Expanded(child: _buildStepBody()),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new_rounded),
            onPressed: () => Navigator.maybePop(context),
          ),
          const SizedBox(width: 4),
          const Text("Karpuz testi",
              style: TextStyle(
                  fontSize: 18, fontWeight: FontWeight.w700)),
          const Spacer(),
        ],
      ),
    );
  }

  Widget _buildStepIndicator() {
    final stepIndex = _step == _Step.photo
        ? 0
        : _step == _Step.audio
            ? 1
            : _step == _Step.haptic
                ? 2
                : 3;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: List.generate(3, (i) {
          final active = i <= stepIndex && stepIndex < 3;
          final done = i < stepIndex && stepIndex < 3 ||
              stepIndex == 3 && i <= 2;
          final color = done
              ? AppTheme.primary
              : (active ? AppTheme.primary : AppTheme.border);
          return Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 3),
              height: 6,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildStepBody() {
    switch (_step) {
      case _Step.photo:
        return _buildPhotoStep();
      case _Step.audio:
        return _buildAudioStep();
      case _Step.haptic:
        return _buildHapticStep();
      case _Step.analyzing:
        return _buildAnalyzingStep();
    }
  }

  // ---------- STEP 1 ----------
  Widget _buildPhotoStep() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Column(
        children: [
          const _StepHeader(
            stepNumber: 1,
            title: "Karpuzun fotoğrafını çek",
            subtitle:
                "Karpuzu yeşil çerçevenin içine al. Işık yeterli olsun.",
          ),
          const SizedBox(height: 16),
          Expanded(
            child: Container(
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppTheme.border),
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (_cameraReady && _camera != null)
                    CameraPreview(_camera!)
                  else
                    const Center(
                      child: CircularProgressIndicator(
                        color: AppTheme.primary,
                      ),
                    ),
                  Center(
                    child: Container(
                      width: 240,
                      height: 240,
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: Colors.greenAccent.shade400,
                          width: 3,
                        ),
                        borderRadius: BorderRadius.circular(24),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
          _PrimaryButton(
            icon: Icons.photo_camera_rounded,
            label: "Fotoğrafı çek",
            enabled: _cameraReady,
            onPressed: _capturePhoto,
          ),
        ],
      ),
    );
  }

  // ---------- STEP 2 ----------
  Widget _buildAudioStep() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Column(
        children: [
          const _StepHeader(
            stepNumber: 2,
            title: "Karpuza parmağınla vur",
            subtitle:
                "Mikrofonu karpuza yaklaştır, 2-3 kez sertçe vur. Telefon sesi 3 saniye kaydeder.",
          ),
          const SizedBox(height: 20),
          Expanded(
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    width: _audioRecording ? 220 : 180,
                    height: _audioRecording ? 220 : 180,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _audioRecording
                          ? Colors.red.shade50
                          : AppTheme.primary.withOpacity(0.08),
                    ),
                    child: Center(
                      child: Container(
                        width: 130,
                        height: 130,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _audioRecording
                              ? AppTheme.accent
                              : AppTheme.primary,
                          boxShadow: [
                            BoxShadow(
                              color: (_audioRecording
                                      ? AppTheme.accent
                                      : AppTheme.primary)
                                  .withOpacity(0.35),
                              blurRadius: 24,
                              offset: const Offset(0, 10),
                            )
                          ],
                        ),
                        child: Icon(
                          _audioRecording
                              ? Icons.mic_rounded
                              : Icons.mic_none_rounded,
                          color: Colors.white,
                          size: 56,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    _audioRecording
                        ? "Kaydediliyor... $_audioCountdown sn"
                        : "Hazır olduğunda başlat",
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: _audioRecording
                          ? AppTheme.accent
                          : AppTheme.slate,
                    ),
                  ),
                ],
              ),
            ),
          ),
          _PrimaryButton(
            icon: _audioRecording
                ? Icons.stop_rounded
                : Icons.fiber_manual_record_rounded,
            label: _audioRecording
                ? "Kaydediliyor..."
                : "Kayıt başlat (3 sn)",
            enabled: true,
            onPressed: _audioRecording ? _stopAudio : _startAudio,
            color: _audioRecording ? AppTheme.accent : AppTheme.primary,
          ),
        ],
      ),
    );
  }

  // ---------- STEP 3: Vi-Liquid Aktif Haptic ----------
  Widget _buildHapticStep() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Column(
        children: [
          const _StepHeader(
            stepNumber: 3,
            title: "Telefonu karpuza yasla",
            subtitle:
                "Telefonun arkasını karpuza yaslı tut. Başlat'a basınca telefon 3 saniye titrer; biz titreşim cevabını dinleriz.",
          ),
          const SizedBox(height: 14),
          _buildMassSlider(),
          const SizedBox(height: 18),
          _buildHapticAnimation(),
          const SizedBox(height: 14),
          Text(
            _sensorRecording
                ? "Titreşim açık · $_sensorCountdown sn kaldı"
                : "Hazır olduğunda başlat",
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: _sensorRecording
                  ? AppTheme.amber
                  : AppTheme.slate,
            ),
          ),
          const SizedBox(height: 14),
          _PrimaryButton(
            icon: _sensorRecording
                ? Icons.stop_rounded
                : Icons.vibration_rounded,
            label: _sensorRecording
                ? "Sıkı tut..."
                : "Başlat (3 sn titrer)",
            enabled: !_sensorRecording,
            onPressed: _startSensor,
            color: _sensorRecording ? AppTheme.amber : AppTheme.primary,
          ),
          const SizedBox(height: 10),
          _buildHapticInfoCard(),
        ],
      ),
    );
  }

  Widget _buildMassSlider() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.fitness_center_rounded,
                  size: 20, color: AppTheme.primary),
              const SizedBox(width: 8),
              const Expanded(
                child: Text("Karpuz ağırlığı",
                    style: TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w700)),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.primary.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text("${_massKg.toStringAsFixed(1)} kg",
                    style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                        color: AppTheme.primary)),
              ),
            ],
          ),
          const SizedBox(height: 6),
          if (_massEstimating)
            const Row(
              children: [
                SizedBox(
                  width: 12,
                  height: 12,
                  child: CircularProgressIndicator(
                    strokeWidth: 1.5,
                    color: AppTheme.primary,
                  ),
                ),
                SizedBox(width: 8),
                Text("Fotodan ağırlık tahmin ediliyor...",
                    style: TextStyle(
                        fontSize: 11, color: AppTheme.slate)),
              ],
            )
          else if (_massEstimate != null)
            Row(
              children: [
                const Icon(Icons.auto_awesome_rounded,
                    size: 14, color: AppTheme.primary),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    "Fotodan tahmin ettik. İstersen aşağıdaki çubukla değiştir.",
                    style: const TextStyle(
                        fontSize: 11,
                        color: AppTheme.primary,
                        fontStyle: FontStyle.italic),
                  ),
                ),
              ],
            ),
          const SizedBox(height: 4),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: AppTheme.primary,
              inactiveTrackColor: AppTheme.primary.withOpacity(0.18),
              thumbColor: AppTheme.primary,
              overlayColor: AppTheme.primary.withOpacity(0.2),
              trackHeight: 4,
            ),
            child: Slider(
              min: 1.0,
              max: 10.0,
              divisions: 90,
              value: _massKg,
              onChanged: _sensorRecording
                  ? null
                  : (v) => setState(() => _massKg = v),
            ),
          ),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 10),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text("hafif (1 kg)",
                    style: TextStyle(
                        fontSize: 11, color: AppTheme.slate)),
                Text("ağır (10 kg)",
                    style: TextStyle(
                        fontSize: 11, color: AppTheme.slate)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHapticAnimation() {
    return SizedBox(
      height: 180,
      child: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          width: _sensorRecording ? 200 : 160,
          height: _sensorRecording ? 200 : 160,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: _sensorRecording
                ? AppTheme.amber.withOpacity(0.12)
                : AppTheme.primary.withOpacity(0.08),
          ),
          child: Center(
            child: Container(
              width: 110,
              height: 110,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _sensorRecording ? AppTheme.amber : AppTheme.primary,
                boxShadow: [
                  BoxShadow(
                    color: (_sensorRecording
                            ? AppTheme.amber
                            : AppTheme.primary)
                        .withOpacity(0.35),
                    blurRadius: 22,
                    offset: const Offset(0, 8),
                  )
                ],
              ),
              child: const Icon(
                Icons.vibration_rounded,
                color: Colors.white,
                size: 48,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHapticInfoCard() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.primary.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.primary.withOpacity(0.2)),
      ),
      child: const Row(
        children: [
          Icon(Icons.tips_and_updates_rounded,
              color: AppTheme.primary, size: 18),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              "Telefon titreşirken karpuzdan dönen cevabı dinleriz. "
              "İçi boş karpuzlarda cevap zayıf olur.",
              style: TextStyle(
                  fontSize: 11,
                  color: AppTheme.ink,
                  height: 1.4),
            ),
          ),
        ],
      ),
    );
  }

  // ---------- STEP 4 ----------
  Widget _buildAnalyzingStep() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppTheme.primary.withOpacity(0.08),
            ),
            child: const Center(
              child: SizedBox(
                width: 56,
                height: 56,
                child: CircularProgressIndicator(
                  strokeWidth: 5,
                  color: AppTheme.primary,
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            "Karpuzu inceliyorum...",
            style: TextStyle(
                fontSize: 17, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          const Text(
            "Fotoğraf, ses ve titreşim cevaplarını topluyorum.",
            style: TextStyle(fontSize: 13, color: AppTheme.slate),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorScreen(String msg) {
    return Scaffold(
      backgroundColor: AppTheme.surface,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline_rounded,
                  size: 64, color: Colors.red.shade400),
              const SizedBox(height: 16),
              Text(
                msg,
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 15, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: () => Navigator.pop(context),
                style: FilledButton.styleFrom(
                    backgroundColor: AppTheme.primary,
                    minimumSize: const Size.fromHeight(52)),
                child: const Text("Geri dön"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StepHeader extends StatelessWidget {
  final int stepNumber;
  final String title;
  final String subtitle;
  const _StepHeader({
    required this.stepNumber,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: AppTheme.primary,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Center(
                  child: Text("$stepNumber",
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 15,
                          fontWeight: FontWeight.w800)),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(title,
                    style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w800)),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(subtitle,
              style: const TextStyle(
                  fontSize: 13,
                  color: AppTheme.slate,
                  height: 1.4)),
        ],
      ),
    );
  }
}

class _PrimaryButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool enabled;
  final VoidCallback onPressed;
  final Color? color;
  const _PrimaryButton({
    required this.icon,
    required this.label,
    required this.enabled,
    required this.onPressed,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return FilledButton.icon(
      style: FilledButton.styleFrom(
        backgroundColor: color ?? AppTheme.primary,
        foregroundColor: Colors.white,
        disabledBackgroundColor: AppTheme.border,
        minimumSize: const Size.fromHeight(58),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        textStyle: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w700,
        ),
      ),
      onPressed: enabled ? onPressed : null,
      icon: Icon(icon, size: 24),
      label: Text(label),
    );
  }
}
