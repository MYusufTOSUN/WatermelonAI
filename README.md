# Karpuz AI — Multimodal Watermelon Ripeness Detection

Tahribatsız (non-destructive) karpuz olgunluk tespitinde çok-modlu (görsel + akustik + haptic) bir derin öğrenme pipeline'ı + tamamı cihaz üzerinde çalışan Flutter Android uygulaması.

Koç & Akbalık (2025) referans alınmış, Late Fusion DNN tabanlı mimari. 19 karpuzluk Qilin Dataset üzerinde dürüst LOWO değerlendirmesiyle 2-sınıf (Yenir/Yenmez) baseline'ı **%61.5**.

---

## Hızlı Bakış

| Bileşen | Durum | Sonuç |
|---|---|---|
| Backend pipeline (5 modül + TFLite) | ✅ | 5/5 modül doğrulandı |
| Görsel sınıflandırıcı (MobileNetV3-Small) | ✅ | 1.9 MB FP16 TFLite |
| Akustik 120-D feature extractor (Koç & Akbalık) | ✅ | librosa-uyumlu |
| Multimodal fusion DNN | ✅ | 78 KB FP16 TFLite, 4-input |
| LOWO 2-sınıf RFC | ✅ | %61.5 doğruluk |
| Flutter Android APK | ✅ | 80 MB, multimodal canlı |
| 3-adım sihirbazlı UX | ✅ | Foto → Ses → Dokunuş |
| MRD-YOLO mobile TFLite | ⚠️ | Backend'de PT, mobile entegrasyon gelecek çalışma |

---

## Proje Yapısı

```
watermelon/
├── backend/             # Python ML pipeline
│   ├── module_a/        # Visual analyzer (MRD-YOLO + heuristic)
│   ├── module_b/        # Pressure / haptic analyzer
│   ├── module_c/        # SRR signal reconstruction
│   ├── module_d/        # 120-D acoustic feature extractor
│   ├── module_e/        # Classifier + late fusion + HH detector
│   └── pipeline/        # Training scripts
├── data/
│   ├── models/          # Trained models (.joblib, .tflite)
│   └── processed/       # Cached features
├── flutter_app/         # Mobile uygulama (Dart)
│   ├── lib/
│   │   ├── services/    # On-device DSP + feature extraction
│   │   ├── screens/     # UI (home, capture wizard, result)
│   │   └── theme/       # App theme
│   └── assets/models/   # Bundled TFLite files
├── scripts/             # Training + validation scripts
├── docs/REPORT.md       # Detaylı teknik rapor
├── releases/            # APK build outputs
└── ROADMAP.md           # Teslim yol haritası
```

---

## Mobil Uygulama: 3-Adım Multimodal Capture

```
┌─ Adım 1 ──────┬─ Adım 2 ──────┬─ Adım 3 ────────┬─ Sonuç ──┐
│  📷 Foto     │  🎤 Ses kayıt │  📳 Dokunuş     │ Yenir/   │
│  → 11-D      │  → 120-D MFCC │  → 7-D haptic   │ Yenmez   │
│   (HSV+Tx)   │  → 8-D HH     │                 │ + güven  │
└──────────────┴───────────────┴─────────────────┴──────────┘
                        ↓
       fusion_model_fp16.tflite (4 input → 3 sınıf)
```

Tüm öznitelik çıkarımı Dart tarafında, internet bağlantısı **yok**:
- **FFT**: `fftea` paketi
- **MFCC**: librosa-uyumlu mel filterbank + DCT-II
- **HSV / Sobel / Laplacian**: `image` paketi
- **Accelerometer**: `sensors_plus` 100 Hz

APK indirme: [`releases/watermelon_ai_v2_multimodal.apk`](releases/watermelon_ai_v2_multimodal.apk) (~80 MB)

---

## Kurulum (Geliştirme)

### Backend (Python 3.10+)
```bash
pip install -r requirements.txt
python scripts/validate_pipeline.py   # 5/5 modül doğrulaması
python scripts/eval_binary_ripeness.py # 2-sınıf LOWO
```

### Flutter Uygulaması
```bash
cd flutter_app
flutter pub get
flutter build apk --release
# Çıktı: build/app/outputs/flutter-apk/app-release.apk
```

> ⚠️ Windows + non-ASCII path: `android.overridePathCheck=true` gerekli. Mevcut `android/gradle.properties` zaten ayarlı.

---

## Sonuçlar

### Backend LOWO (Leave-One-Watermelon-Out)

| Görev | Model | Accuracy |
|---|---|---|
| 3-sınıf | KNN | %38.8 |
| 3-sınıf | RFC | %52.4 |
| **2-sınıf (Yenir/Yenmez)** | **RFC** | **%61.5** |
| 2-sınıf | KNN | %46.8 |

Detaylı sonuçlar: [`data/models/binary_lowo_results.json`](data/models/binary_lowo_results.json), [`data/models/training_results.json`](data/models/training_results.json)

---

## Teslim Belgeleri

- 📄 **Teknik Rapor**: [`docs/REPORT.md`](docs/REPORT.md)
- 📱 **APK**: [`releases/watermelon_ai_v2_multimodal.apk`](releases/watermelon_ai_v2_multimodal.apk)
- 🗺️ **Yol Haritası**: [`ROADMAP.md`](ROADMAP.md)

---

## Kaynakça (Kısa)

- Koç, M. & Akbalık, H. (2025). *Multi-modal Late Fusion for Watermelon Ripeness Detection.*
- Jing, X. *MRD-YOLO*: https://github.com/XuebinJing/Melon-Ripeness-Detection
- Yamamoto et al. (1980). *Acoustic Impulse Response Method for Watermelon Ripeness.*
- McFee, B. et al. *librosa: Audio and Music Signal Analysis in Python.*

---

## Lisans

Akademik / araştırma amaçlı kullanım. Ticari kullanım için yazar onayı gerekir.
