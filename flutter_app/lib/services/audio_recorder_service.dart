import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:wav/wav.dart';

import 'dsp/dsp_utils.dart';

/// Wraps the `record` package to capture short audio clips at 44.1 kHz
/// and return float64 PCM samples in [-1, 1].
class AudioRecorderService {
  final AudioRecorder _recorder = AudioRecorder();
  String? _currentPath;

  Future<bool> hasPermission() => _recorder.hasPermission();

  Future<void> start() async {
    final dir = await getApplicationDocumentsDirectory();
    final samplesDir = Directory("${dir.path}/samples");
    if (!await samplesDir.exists()) {
      await samplesDir.create(recursive: true);
    }
    final path =
        "${samplesDir.path}/wm_audio_${DateTime.now().millisecondsSinceEpoch}.wav";
    _currentPath = path;
    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: DspUtils.audioSampleRate,
        numChannels: 1,
        bitRate: 16 * DspUtils.audioSampleRate,
        // KRITIK: varsayilan ses kaynagi (defaultSource / voiceRecognition)
        // bircok Android cihazda — ozellikle Samsung'da — yuksek-geciren
        // filtre + gurultu bastirma + AGC uygular. Bu DSP, karpuz vurus
        // rezonansinin bulundugu 50-250 Hz bandini kirpip yok ediyor ve
        // model "ham" sanıyor (saha hatasinin kok nedeni). UNPROCESSED ham
        // mikrofon akisini verir; desteklenmeyen cihazlarda sistem mic'e
        // duser. echo/gain/noise-suppression efektleri de kapatilir.
        androidConfig: AndroidRecordConfig(
          audioSource: AndroidAudioSource.unprocessed,
          audioManagerMode: AudioManagerMode.modeNormal,
          manageBluetooth: false,
        ),
        iosConfig: IosRecordConfig(
          // iOS measurement mode: AGC/filtreleme olmadan duz mikrofon yaniti
          categoryOptions: [],
        ),
      ),
      path: path,
    );
  }

  /// Stops recording and returns the path to the WAV file.
  Future<String?> stop() async {
    return await _recorder.stop();
  }

  Future<void> cancel() async {
    if (await _recorder.isRecording()) {
      await _recorder.cancel();
    }
  }

  Future<bool> isRecording() => _recorder.isRecording();

  String? get currentPath => _currentPath;

  /// Loads a WAV file into a normalized Float64List in [-1, 1].
  static Future<Float64List> loadWavAsFloat64(String path) async {
    final bytes = await File(path).readAsBytes();
    final wav = Wav.read(bytes);
    if (wav.channels.isEmpty) {
      return Float64List(0);
    }
    final mono = wav.toMono();
    // wav package returns Float64List in [-1, 1] already
    return Float64List.fromList(mono);
  }

  /// Trims silence on both sides using a gentle energy threshold.
  ///
  /// IMPORTANT: must stay GENTLE — the fusion DNN was trained on features
  /// from near-full 3-second recordings; aggressively cropping to the
  /// knock transient (v13) shifted frame statistics (mean/std/min/max)
  /// outside the training distribution and broke classification.
  static Float64List trimSilence(
    Float64List audio, {
    double thresholdRms = 0.005,
  }) {
    if (audio.length < 1024) return audio;
    const win = 512;
    const hop = 256;
    final nFrames = (audio.length - win) ~/ hop;
    if (nFrames < 1) return audio;
    int firstActive = 0;
    int lastActive = nFrames - 1;
    bool found = false;
    for (int i = 0; i < nFrames; i++) {
      double s = 0.0;
      for (int j = 0; j < win; j++) {
        final v = audio[i * hop + j];
        s += v * v;
      }
      final rms = math.sqrt(s / win);
      if (rms > thresholdRms) {
        if (!found) {
          firstActive = i;
          found = true;
        }
        lastActive = i;
      }
    }
    if (!found) return audio;
    final start = firstActive * hop;
    final end = (lastActive + 1) * hop + win;
    final hi = end > audio.length ? audio.length : end;
    if (hi - start < 1024) return audio;
    return Float64List.fromList(audio.sublist(start, hi));
  }

  void dispose() {
    _recorder.dispose();
  }
}
