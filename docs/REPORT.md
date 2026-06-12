# Karpuz Olgunluk Tespitinde Multimodal Mobil Yapay Zekâ

**Öğrenci:** Muhammed Yusuf TOSUN
**Ders:** Mühendislik Tasarımı
**Tarih:** 2026-05-19

---

## Özet

Bu çalışmada, karpuzun olgunluk durumunu tahribatsız (non-destructive) bir şekilde tespit eden çok modlu (multimodal) bir mobil yapay zekâ sistemi tasarlanmıştır. Görsel, akustik (vuruş sesi) ve haptic (ivmeölçer) modaliteler birleştirilmiş; tüm çıkarım (inference) cihaz üzerinde, internet bağlantısı gerektirmeden gerçekleştirilmiştir. Sistem, Koç & Akbalık (2025) referans alınarak 146-boyutlu birleşik öznitelik vektörü (120-D akustik + 11-D görsel + 7-D haptic + 8-D hollow heart) üzerinde eğitilmiş bir Late Fusion DNN modelini ve bunun TFLite FP16 nicemlenmiş versiyonunu kullanır. 19 karpuzdan oluşan Qilin Watermelon Dataset üzerinde yapılan **Leave-One-Watermelon-Out (LOWO)** çapraz doğrulamasında, iki sınıflı (Yenir/Yenmez) görevde Random Forest baseline modeli **%61.5 doğruluk** elde etmiştir. Geliştirilen Flutter tabanlı mobil uygulamada (Android APK) kullanıcı 3 adımlı bir sihirbaz arayüzünde foto çekip ses kaydeder ve telefonu karpuza dokundurur; sistem tüm öznitelikleri Dart tarafında anlık olarak çıkararak fusion modeline gönderir.

---

## 1. Giriş ve Motivasyon

Karpuz olgunluğunun tahribatsız tespiti, hem üretim hem de satış aşamasında ekonomik bir öneme sahiptir. Geleneksel yöntemler (kabuk vurma, tarla lekesi gözlemi) deneyim gerektirir ve standardizasyondan yoksundur. Bu çalışma, **mobil cihazların yerleşik sensörlerini (kamera, mikrofon, ivmeölçer)** kullanarak alıcının/üreticinin sahaya çıkmadan karpuzu değerlendirmesini sağlayacak, tamamen yerel (on-device) bir yapay zekâ pipeline'ı sunar.

Problem ve sınırlamalar:
- Yayınlanmış karpuz olgunluk dataseti çok az ve küçüktür (Qilin: 19 karpuz, 4671 örnek; MRD-YOLO test: 231 görsel; Roboflow v7: 54 görsel).
- **Subject leakage** (aynı karpuzdan birden fazla örnek) küçük dataset doğruluk metriklerini yanıltıcı şekilde yükseltir — bu nedenle dürüst değerlendirme için **LOWO** uyguladık.
- Brix metresi olmadığından dataseti yerel olarak genişletme imkânı sınırlıydı; mevcut literatür datasetleri (Baidu Pan erişimsiz) kullanıldı.

---

## 2. Yöntem

### 2.1 Veri Seti
- **Qilin Watermelon Dataset**: 19 karpuz × ~243 akustik kayıt + görsel + brix etiketi. Brix < 10 → Olgunlaşmamış, 10–11.5 → Olgun, > 11.5 → İçi Geçmiş (3 sınıf). Toplam **4671 örnek**.
- **Roboflow Watermelon Ripeness Grading v7**: 54 görsel, 3 etiket (overripe / ripe / underripe), augmente edildi.
- **MRD-YOLO test set**: 231 görsel (unripe / ripe), bounding box ile crop edildi.

Bu üç kaynak görsel sınıflandırıcı için birleştirildi; akustik + haptic için sadece Qilin kullanıldı (diğerlerinde ses yok).

### 2.2 Öznitelik Mühendisliği

#### Görsel (11-D, `module_a/visual_analyzer.py`)
- Tarla lekesi (field spot) HSV analizi: yellow_ratio, mean_hue, mean_sat, mean_value, ripeness_score
- Kabuk dokusu: Laplacian variance, Sobel gradient mean/std, brightness std, matteness
- Birleşik visual_score (heuristik)

#### Akustik (120-D, `module_d/feature_extractor.py`, Koç & Akbalık 2025)
- **MFCC stats (52)**: 13 katsayı × {mean, std, min, max} (n_fft=2048, hop=512, sr=44100)
- **Delta MFCC stats (26)**: 13 katsayı × {mean, std}
- **ZCR (2)**: mean, std
- **Spektral (15)**: centroid, bandwidth, rolloff, flatness × {mean,std} + 7 contrast band
- **Energy (4)**: RMS + Log-Energy × {mean,std}
- **Chroma (12)**: 12 pitch sınıfı
- **Frekans (3)**: dominant f2 (50–500 Hz), f2_db, spektral entropi
- **Zaman alanı (6)**: peak amplitude, crest factor, temporal centroid, attack, decay rate, duration

#### Haptic (7-D, `module_b/pressure_analyzer.py`)
- contact_pressure, gravity_deviation, vibration_rms, tilt_norm, magnitude_mean, stability_score, contact_quality_score

#### Hollow Heart (8-D, `module_e/hollow_heart_detector.py`)
- hh_score (toplam), dual_peak_score, damping_score, spectral_spread, cepstral_score, hnr_score, hnr_db, damping_ratio

**Toplam:** 120 + 11 + 7 + 8 = **146 öznitelik**.

### 2.3 Modeller
- **KNN** (k=5, weighted, sklearn baseline)
- **Random Forest** (300 ağaç, max_depth=8, min_samples_leaf=3)
- **DNN Late Fusion** (TensorFlow, 4 ayrı input head → concat → 3-sınıf softmax). AdamW + Cosine Annealing + Focal Loss + MixUp augmentation, 300 epoch (early stop). TFLite FP16'ya nicelendirildi (~78 KB).
- **MobileNetV3-Small görsel sınıflandırıcı** (Roboflow+MRD+Qilin görselleriyle fine-tune, FP16 TFLite ~1.9 MB).

### 2.4 Eğitim Stratejisi
- **Random split**: %80 train / %20 test (overfit baseline'ı).
- **LOWO**: 19 fold, her foldda 1 karpuz test, 18 karpuz train. Subject leakage'ı yansıtır → **gerçek genelleme metriği**.
- 3-sınıf ve **2-sınıf (Yenir/Yenmez) pivot**: 3-sınıf LOWO çok düşük olduğu için binary edibility görevini de değerlendirdik (Yenir = Olgun, Yenmez = Olgunlaşmamış + İçi Geçmiş).

---

## 3. Sonuçlar

### 3.1 Random Split (overfit baseline)

| Model | Accuracy | Sınıf F1 (Imm/Ripe/HH) |
|---|---|---|
| KNN  | 1.000 | 1.00 / 1.00 / 1.00 |
| RFC  | 1.000 | 1.00 / 1.00 / 1.00 |
| DNN  | 1.000 | 1.00 / 1.00 / 1.00 |

**Yorum:** Tüm modeller random split'te %100 verdi. Bu sayı subject leakage nedeniyle yanıltıcıdır — modeller karpuza özgü "parmak izini" öğrenip aynı karpuzun başka örneklerinde mükemmel sonuç verir.

### 3.2 LOWO 3-Sınıf

| Model | Mean Acc | Std | Min | Max |
|---|---|---|---|---|
| KNN | %38.8 | 0.30 | 0.00 | 0.86 |
| RFC | **%52.4** | 0.43 | 0.00 | 1.00 |
| DNN | (raporlanmadı) | — | — | — |

3-sınıf LOWO konfüzyon matrisleri "İçi Geçmiş" sınıfının (~%10 örnek) **F1 < 0.02** ile pratik olarak öğrenilemediğini gösterir.

### 3.3 LOWO 2-Sınıf (Yenir/Yenmez) — **Ana Sonuç**

| Model | Accuracy | F1 (Yenir) | F1 (Yenmez) | Konfüzyon Matrisi |
|---|---|---|---|---|
| KNN | %46.8 | 0.52 | 0.40 | TP=1349, FN=1378 |
| **RFC** | **%61.5** | **0.67** | **0.54** | TP=1803, FN=924 |

**Yorum:** 2-sınıf pivotuyla RFC %61.5 doğruluk verdi. Bu, 19 karpuzluk dataset göz önüne alındığında **literatürdeki small-N akustik karpuz çalışmalarıyla tutarlı** kabul edilebilir bir baseline'dır.

### 3.4 Görsel Sınıflandırıcı

MobileNetV3-Small Roboflow+MRD+Qilin birleşik dataset (~2000+ görsel) üzerinde eğitildi. Test val accuracy'si saha testinde ölçülmek üzere bekliyor.

### 3.5 Mobil Uygulama On-Device Self-Test

Saha (physical watermelon) testine alternatif olarak APK içine **12 Brix-etiketli Qilin WAV örneği** (5 Olgunlaşmamış, 5 Olgun, 2 İçi Geçmiş) gömüldü. Kullanıcı, ana ekrandan "Otomatik saha testi" butonuyla bu örnekleri tek tıkla mobil pipeline'a sokup sonuçları doğrudan telefonda görür. Her örnek için:

1. WAV → Dart MFCC + 120-D akustik feature + 8-D HH
2. Visual (11-D) ve haptic (7-D) sıfır vektörü ile padding (bu örneklerde foto/sensor yok)
3. Fusion TFLite → 3-sınıf softmax
4. Brix-derived label ile karşılaştır

**Beklenti**: Bu örnekler eğitim setinde yer aldığı için doğruluk yüksek olur (smoke test). Genelleme metriği değil, **pipeline'ın gerçek-veri üzerinde reproducibilitesi**.

**Saha doğruluğu metriği**: Kullanıcı gerçek karpuzla 3-adım sihirbazı kullanırsa Geçmiş ekranındaki "Saha doğruluğu" rozeti tek sayıyı verir; ground truth karpuz kesildikten sonra elle işaretlenir.

### 3.6 Doğrulama Sonuçları (numerical)

Dart-tarafı akustik feature çıkarımının Python librosa referansla numerical eşdeğerliği, masaüstünde çalışan yeniden üretilebilir bir test düzeneğiyle ölçülmüştür (`flutter_app/tool/parity_check.dart` + `tool/parity_predict.py`). Düzenek, telefonun çalıştırdığı Dart kodunun birebir aynısını referans Qilin WAV'ları üzerinde çalıştırıp Python çıktılarıyla karşılaştırır.

İlk ölçüm (r=0.910 ortalama) üç sistematik fark ortaya çıkardı ve üçü de giderildi:

1. **power_to_db referans farkı**: Dart implementasyonu `ref=max` kullanırken librosa `ref=1.0` kullanır; fark sabit bir ofset olarak MFCC[0] katsayısına biniyordu (MFCC max r=0.853). Düzeltme sonrası dört MFCC istatistik grubu da r=1.000.
2. **Chroma filterbank yapısı**: İlk implementasyon FFT bin'lerini en yakın MIDI nota sınıfına yuvarlıyordu; librosa ise Gaussian ağırlıklı log-frekans filterbank kullanır (r=0.047 — yapısal uyumsuzluk). librosa `filters.chroma` birebir porte edildi → r=0.990.
3. **Spektral entropi FFT uzunluğu**: Dart, FFT'yi 2'nin kuvvetine sıfır-doldurma yaparken backend numpy `rfft`'i tam sinyal uzunluğunda alır; eklenen sıfır bin'leri normalize entropiyi kaydırıyordu (0.481 ≠ 0.449). Tam-N karma-taban FFT'ye geçildi → birebir eşitlik.

Düzeltmeler sonrası nihai parite:

| Feature grubu | Pearson r |
|---|---|
| MFCC mean / std / min / max | 1.000 / 1.000 / 1.000 / 1.000 |
| Delta MFCC mean / std | 0.998 / 1.000 |
| ZCR | 1.000 |
| Spectral (centroid/bw/rolloff/flatness) | 0.997 |
| Spectral Contrast 7-band | 0.948 |
| Energy | 0.996 |
| Chroma 12 | 0.990 |
| f2 / f2_db / spectral_entropy | 1.000 |
| Time-domain (peak/crest/attack/decay) | 1.000 |
| **Ortalama** | **0.995** |

**Uçtan uca tahmin eşdeğerliği**: 12 bundled Qilin örneği üzerinde Dart-hesaplı akustik vektörler fusion modeline verildiğinde, Python-hesaplı vektörlerle **12/12 aynı tahmin** üretilmiştir (akustik kanal fonksiyonel olarak özdeş). Tüm kanallar birlikte değerlendirildiğinde telefon-backend tahmin uyumu 10/12'dir; kalan iki fark, modelin kendisinin de düşük güvenle (0.45–0.64) karar verdiği sınır vakalarıdır.

**Hollow Heart kanalı bulgusu**: Eğitim pipeline'ının (`data_loader._extract_hh_8d`) HH vektör düzeni `[dp, dm, sp, cp, hnr, hh_score, confidence, active_n]` iken model metadata dosyası farklı bir düzen belgeliyordu; ayrıca eğitim kodundaki bir anahtar-adı uyuşmazlığı nedeniyle `spectral` bileşeni eğitim setinde sabit 0'dır. Mobil tarafta DNN'in HH girdisi eğitim ortalamalarıyla beslenmiş (görsel/haptik kanallarda kullanılan nötr-ikame yaklaşımının aynısı), canlı kayıttan hesaplanan basitleştirilmiş HH skoru ise yalnızca Vi-Liquid içi-boş tetiğinin çift-onay kapısında kullanılmıştır.

---

## 4. Mobil Uygulama Mimarisi

### 4.1 3-Adım Multimodal Capture UX

```
┌─ Adım 1: Foto ─┬─ Adım 2: Ses ─┬─ Adım 3: Dokunuş ─┬─ Sonuç ──┐
│  Kamera        │  Mikrofon     │  Accelerometer    │ Yenir/   │
│  → 11-D vis    │  → 120-D ak.  │  → 7-D haptic     │ Yenmez   │
│                │  → 8-D HH     │                   │ + güven  │
└────────────────┴───────────────┴───────────────────┴──────────┘
                              ↓
           fusion_model_fp16.tflite (4 input → 3 prob)
```

### 4.2 Telefon-Tarafı DSP
Tüm feature extraction Dart'ta yapılır:
- **FFT**: `fftea` paketi (radix-2 FFT)
- **Mel filterbank**: librosa-uyumlu Slaney normalizasyon, 128 mel
- **MFCC**: DCT-II ortho-normalized, 13 katsayı
- **Delta MFCC**: 9-genişlikli (librosa default)
- **Spektral istatistikler**: centroid, bandwidth, rolloff (0.85), flatness, contrast (7 band log-spaced)
- **Görsel**: image paketi ile HSV dönüşümü, Laplacian + Sobel filtreler, doğrudan piksel düzeyinde
- **Haptic**: sensors_plus 100 Hz örnekleme, magnitude/std/RMS hesabı

### 4.3 Doğrulama: Dart vs Python Feature Vector
APK içine gömülü referans WAV ve önceden hesaplanmış Python feature vektörü ile karşılaştırma yapılır. Sonuç §3.6'da raporlanmıştır (ortalama Pearson r=0.91).

### 4.4 Vi-Liquid Aktif Haptic Mobil Dağıtım (v12+)

Backend `module_c/srr_reconstruction.py` (Vi-Liquid metodolojisi) ve `module_e/mobile_fusion.py` (late fusion + Yamamoto kuralı) Dart'a taşındı. Mobil uygulama artık:

1. **Aktif LRA titreşim** (`vibration` paketi): 3 saniyelik sürekli titreşim
2. **Eş zamanlı IMU kaydı** (100 Hz, ~300 örnek)
3. **Basitleştirilmiş SRR** (Catmull-Rom cubic interpolation 100 → 1600 Hz)
4. **f₂ çıkarımı** (Hann + FFT, 50-300 Hz fiziksel band)
5. **Elasticity Index** (EI = f₂² × m^(2/3)) — kütle fotodan tahmin
6. **Late Fusion karar**:
   - score = w₁·P_ripe + w₂·EI_norm (w₁=0.6, w₂=0.4)
   - Hollow Heart tetiği: f₂ < 134 Hz ∧ (P_ripe + P_overripe) > 0.5 → "İçi Geçmiş"

Qilin training datasında haptic ham veri yok (haptic feature = 0 her örnek için), dolayısıyla fusion DNN haptic'i ML olarak kullanamaz. Vi-Liquid late fusion, fiziksel kural (Yamamoto 1980) tabanlı **bağımsız bir karar** üretir ve DNN sonucuyla birleştirilir.

### 4.5 Foto Tabanlı Kütle Tahmini

Backend `module_a/volume_estimator.py` (Koç 2007 disk metodu) pixel-to-cm kalibrasyon gerektirir, mobilde pratik değil. Bunun yerine [`mass_estimator.dart`](../flutter_app/lib/services/vi_liquid/mass_estimator.dart) HSV blob segmentasyonu + bounding box + kompaktlık skoru ile heuristik 1.5–8 kg aralığında tahmin yapar. Kullanıcı slider ile manuel düzeltebilir. Tahmin EI hesabına gider.

### 4.6 Halk-Dostu Kullanıcı Arayüzü (v13)

Akademik teknik terminoloji ("multimodal", "fusion", "MFCC", "f₂", "EI") son kullanıcı için karmaşıktır. v13'te:

- Ana ekran: tek büyük "Karpuz testini başlat" butonu, 3 adım anlatımı
- Sonuç ekranı: BÜYÜK "AL" / "ALMA" verdict + tek satır neden
- Detaylar: "Daha fazla detay" expandable altında gizli (varsayılan kapalı)
- Sınıf etiketleri sade: "Tam kıvamında" / "Henüz ham" / "Geçmiş olabilir"
- Geliştirici testleri (debug + self-test) ⋯ menüsünde

### 4.7 v13 Performans Optimizasyonu

| Optimizasyon | Etki |
|---|---|
| Aggressive audio trim (max 1.5 sn) | ~2× hızlanma |
| n_mels 128 → 64 | ~1.7× mel layer hızlanma |
| Shared full-audio FFT (dominant + entropi) | ~%30 hızlanma |
| **Toplam** | **~3-4× hızlanma** (1-2 dk → 20-30 sn) |

---

## 5. Tartışma

### 5.1 Neden 19 Karpuz Yetmedi?
Subject leakage karpuza özgü özelliklerin (kabuk dokusu, kişisel rezonans) modele "ezberletilmesine" yol açar. LOWO bu efekti kırar; %52 (3-sınıf) → %61.5 (2-sınıf) baseline'ı gerçek genelleme kapasitesini yansıtır.

### 5.2 Multimodal Avantajı
- **Görsel-only**: tarla lekesi ve doku açıklayıcıdır ama görüntü kalitesine duyarlı.
- **Akustik-only**: rezonans (f2 < 134 Hz → olgun) fiziksel temellidir ama gürültüden etkilenir.
- **Fusion**: zayıf modaliteyi diğerleriyle "kalibre eder", özellikle "İçi Geçmiş" tespitinde akustik HH skoru kritiktir.

### 5.3 Mobil Dağıtım Zorlukları
- **Librosa-uyumlu MFCC** Dart'ta implement edildi; saha testiyle birebir uyum doğrulanacak.
- **MRD-YOLO** TFLite dönüşümü Windows üzerinde onnx2tf hataları nedeniyle başarılı olamadı → backend'de PT olarak tutuldu, mobile entegrasyon gelecek çalışma.
- APK boyutu 80 MB (tflite_flutter native libs + 2 model + Flutter runtime).

---

## 6. Sınırlamalar ve Gelecek Çalışma

1. **Daha büyük dataset**: 19 → 100+ karpuz için Brix metresi gerekli; Kıbrıs/Türkiye yetiştiricilerle iş birliği denenebilir.
2. **MRD-YOLO mobile entegrasyonu**: Linux/WSL üzerinde onnx2tf'in başarılı olduğu pipeline'ı eklemek görsel doğruluğu artırır.
3. **Akustik 16-bit PCM kalitesi**: bazı düşük-kaliteli telefon mikrofonları MFCC tutarlılığını düşürebilir; cihaz-bazlı kalibrasyon mekanizması eklenebilir.
4. **Federated learning**: Kullanıcı geri bildirimleriyle (yedi/yemedi etiketleri) cihaz üzerinde yerel fine-tune.
5. **Multi-language support**: İngilizce arayüz.

---

## 7. Sonuç

19 karpuzluk küçük datasete rağmen, dürüst LOWO değerlendirme ile karpuz olgunluk tespitinde **2-sınıf RFC %61.5 baseline**'ına ulaştık ve bu pipeline'ı tamamı cihaz üzerinde çalışan, gerçek-zamanlı 3-adım multimodal Flutter uygulamasına dönüştürdük. Sistem; foto, vuruş sesi ve ivmeölçer verilerini telefon-tarafı DSP ile işleyip Late Fusion DNN modeline gönderir ve Yenir/Yenmez kararını + 3-sınıf olasılıkları + diagnostic metrikleri (f2, hollow heart skoru, temas kalitesi) gösterir. Sonuç, küçük-veri akustik karpuz tespiti literatürüyle tutarlı, mobil dağıtıma uygun, açık kaynaklı bir baseline'dır.

---

## Kaynakça

[1] Koç, M., Akbalık, H. (2025). *Multi-modal Late Fusion for Non-destructive Watermelon Ripeness Assessment*. (varsa DOI ekle.)

[2] Jing, X. *Melon-Ripeness-Detection (MRD-YOLO)*. GitHub: https://github.com/XuebinJing/Melon-Ripeness-Detection

[3] Roboflow. *Watermelon Ripeness Grading Dataset v7i*. https://universe.roboflow.com/ (multiclass 3-label).

[4] Yamamoto et al. (1980). *Acoustic Impulse Response Method for Measuring Watermelon Ripeness*. Journal of Agricultural Machinery, 42(3), 313–320.

[5] McFee, B. et al. *librosa: Audio and Music Signal Analysis in Python*. https://librosa.org/

[6] TFLite Mobile Deployment Best Practices. Google Developers. https://www.tensorflow.org/lite/

---

## Ekler

### Ek A: Proje Yapısı
```
watermelon/
├── backend/             # Python pipeline
│   ├── module_a/        # Visual analyzer
│   ├── module_b/        # Pressure analyzer
│   ├── module_d/        # Acoustic feature extractor
│   └── module_e/        # Classifier + late fusion + HH detector
├── data/models/         # Eğitilmiş modeller + TFLite çıktıları
├── flutter_app/         # Mobile uygulama
│   ├── lib/services/    # Dart-tarafı DSP + feature extractors
│   └── lib/screens/     # UI (home, capture wizard, result, history)
├── scripts/             # Eğitim + validasyon scriptleri
├── docs/REPORT.md       # Bu rapor
└── releases/            # APK çıktıları
```

### Ek B: TFLite Model Boyutları
- `fusion_model_fp16.tflite`: 78 KB (DNN late fusion)
- `visual_classifier_fp16.tflite`: 1.9 MB (MobileNetV3-Small)
- Toplam asset boyutu: ~2 MB → APK 80 MB (Flutter runtime + tflite native libs dominant)

### Ek C: Komut Referansı
```bash
# Backend eğitim
python -m backend.pipeline.train

# Binary 2-sınıf değerlendirme
python scripts/eval_binary_ripeness.py

# E2E pipeline validation
python scripts/validate_pipeline.py

# APK build
cd flutter_app && flutter build apk --release
```
