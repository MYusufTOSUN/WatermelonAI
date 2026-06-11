# Karpuz Olgunluk — Tam Canlı Multimodal APK Yol Haritası

**Bugün:** 2026-05-19
**Teslim:** Esnek (kısa sürede ama kalite öncelikli)
**Scope:** Telefonda gerçek 4-modal kapture + fusion modeli

---

## Teslim Edilecekler

| # | Çıktı | Format |
|---|---|---|
| 1 | Çalışan multimodal APK | `app-release.apk` |
| 2 | Yazılı rapor | PDF, 10-15 sayfa |
| 3 | GitHub repo | README + LICENSE + APK release |

---

## Multimodal Capture UX — 3-Adım Sihirbazı

```
┌─ 1. Foto ──┬─ 2. Ses ──┬─ 3. Tık ────┬─ Sonuç ──┐
│   [📷]     │ [🎤 3sn]  │ [📳 3sn]    │ Yenir/   │
│  karpuz    │ karpuza   │ karpuza     │ Yenmez   │
│  fotosu    │ vur+kayd  │ dokundur    │ %XX güv  │
└────────────┴───────────┴─────────────┴──────────┘
              ↓
   fusion_model_fp16.tflite (4 input → 3 class)
```

---

## Modüller (Telefon-tarafı feature extraction)

| Modal | Feature dim | Dart implementasyon | Risk |
|---|---|---|---|
| **Visual** | 11-D | image paketi → HSV stats + GLCM texture | Düşük |
| **Acoustic** | 120-D | record + fftea → MFCC + spectral | **YÜKSEK** |
| **Haptic** | 7-D | sensors_plus → accelerometer stats | Düşük |
| **Hollow Heart** | 8-D | Visual'dan ridge + bright spot | Orta |

**MobileNetV3 görsel sınıflandırıcı** (`visual_classifier_fp16.tflite`) ayrıca tutulacak — 224x224 foto → 3 sınıf prob. Result ekranında "görsel-only" tahminini fusion ile **karşılaştırmalı** göstereceğiz (akademik değer).

---

## Faz Faz Plan

### Faz 1 — Multimodal İskelet (1-2 gün)
- [ ] pubspec'e geri ekle: `record`, `sensors_plus`, `fftea`
- [ ] `lib/services/audio_recorder_service.dart` — mikrofon kayıt
- [ ] `lib/services/sensor_recorder_service.dart` — accelerometer kayıt
- [ ] `lib/services/feature_extractors/` klasörü
  - `visual_features.dart` (HSV histogram 9-D + dominant color + edge ratio = 11-D)
  - `acoustic_features.dart` (FFT → mel filterbank → MFCC + spectral centroid/rolloff/zcr = 120-D)
  - `haptic_features.dart` (accelerometer: mean, std, peak, energy = 7-D)
  - `hh_features.dart` (8-D: brightness anomaly, hollow indicator placeholders)
- [ ] `lib/services/fusion_model_service.dart` — 4 input → fusion TFLite → 3 prob
- [ ] Android manifest: CAMERA + RECORD_AUDIO + WRITE_EXTERNAL_STORAGE izinleri

### Faz 2 — 3-Adım Wizard UI (1 gün)
- [ ] `lib/screens/capture/capture_wizard_screen.dart` (stepper)
- [ ] `step1_photo.dart` — kamera preview + foto çek
- [ ] `step2_audio.dart` — mikrofon dalga formu + 3-sn kayıt
- [ ] `step3_haptic.dart` — "telefonu karpuza dokundur" + accelerometer plot
- [ ] `step_loading.dart` — feature extraction + fusion inference
- [ ] Result screen güncellemesi: fusion vs visual-only karşılaştırma kartı

### Faz 3 — Acoustic Feature Doğrulama (KRİTİK, 1-2 gün)
- [ ] Backend `module_d/acoustic_feature_extractor.py`'dan tam liste:
  - MFCC 13 katsayı
  - Delta + delta-delta
  - Spectral centroid, rolloff, flux, zcr
  - Toplam 120-D'lik tam tablo
- [ ] Dart'ta aynı parametrelerle implement (sample rate, hop length, window size)
- [ ] **Validation**: aynı .wav dosyasını Python ve Dart'ta işle → feature vektörü karşılaştır → MSE < 0.1 hedef
- [ ] Eğer MSE büyükse: rapora "%X korelasyon, telefon-tarafı feature backend'le farklılaşıyor" yaz

### Faz 4 — Build + Saha Testi (1-2 gün)
- [ ] APK build: `C:\tmp\watermelon_build` (non-ASCII path workaround)
- [ ] 5-10 gerçek karpuz al, 3-adım kaptürle test et
- [ ] Sonuçları kayda al: telefonda fusion ne diyor, gerçek olgunluk ne (kesince anlaşılır)
- [ ] Bug fix turu

### Faz 5 — Rapor (2-3 gün)
- [ ] `docs/REPORT.md` iskelet (önceki ROADMAP'tekinin aynısı + Mobile Deployment bölümü)
- [ ] Tablolar:
  - Backend LOWO 3-sınıf (KNN %38.8, RFC %52.4)
  - Backend LOWO 2-sınıf (RFC %61.5)
  - Telefon-Backend feature korelasyonu (Faz 3'ten)
  - Saha testi sonuçları (Faz 4'ten)
- [ ] Figürler: sistem diyagramı, 3-adım UX akış, confusion matrices
- [ ] Markdown → PDF (Pandoc)

### Faz 6 — GitHub + Final (1 gün)
- [ ] README: özet + kurulum + APK indirme
- [ ] `.gitignore` güncelle
- [ ] APK'yı GitHub release olarak yükle
- [ ] Repo'yu yeni klasöre clone et → backend + flutter çalışıyor mu doğrula
- [ ] Rapor PDF'i son okuma
- [ ] Teslim

---

## Risk Listesi

| Risk | Olasılık | Etki | Plan B |
|---|---|---|---|
| **Dart MFCC backend'le tutmuyor** | Yüksek | Yüksek | Telefonda WAV kaydet, basit MFCC ile dene; raporda gerçek korelasyonu yaz, "%X düşüşle telefon-tarafı çalışıyor" diye savun |
| `fftea` paketi yeterli değil | Orta | Orta | Manuel FFT (Cooley-Tukey) veya `dart_audio_analysis` paketi araştır |
| Accelerometer sample rate düşük | Orta | Düşük | sensors_plus 100 Hz hedef, 50 Hz olsa da yeter |
| Foto + Ses + Sensör senkronizasyonu | Düşük | Orta | Wizard adımlı, eşzamanlı değil → senkron sorunu yok |
| APK boyutu büyük | Düşük | Düşük | TFLite FP16 + libtensorflowlite_jni dahil ~40-50 MB normal |

---

## Yapma Listesi

- ❌ MRD-YOLO TFLite (Windows'ta başarısız, Linux'ta dene gelecek çalışma)
- ❌ Xigua model (duplikasyon)
- ❌ Yeni dataset arayışı
- ❌ Brix metresi sipariş etmek
- ❌ Lottie / fancy animasyonlar

---

## Hocaya Söyleyeceğimiz

> "Multimodal pipeline mobile cihazda tam canlı çalışıyor: kullanıcı 3 adımda (foto, ses, haptic) gerçek karpuz datası toplar, telefon-tarafı feature extraction yapar, fusion model offline inference verir. Backend pipeline'ı 4671 örnekli Qilin datasetinde 2-sınıf LOWO %61.5 baseline'a ulaştı; mobile-tarafı 5 karpuzda saha testiyle %X gerçek doğruluk gözlemledik. 19 karpuz subject leakage limiti literatürde tutarlı bir sınırdır."

---

## Tek Cümlelik Plan

> **İskelet → Wizard UI → MFCC doğrulama → Build/Saha → Rapor → GitHub → Teslim.**
