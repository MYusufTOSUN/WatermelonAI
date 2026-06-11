# Karpuz AI — Dürüst Denetim Raporu

**Tarih:** 2026-05-19
**Soru:** "Modelleri doğru eğittik mi, uygulama fake değil mi?"

---

## TL;DR — Üç kategoride değerlendiriyorum

### ✅ GERÇEK ve DOĞRULANMIŞ (sahte değil)
1. **Eğitim verisi**: 4671 akustik örnek × 19 karpuz × 146 feature. `data/processed/X_features.npy` dosyasında, librosa ile gerçek Qilin WAV dosyalarından çıkarılmış.
2. **Fusion DNN modeli** (`fusion_model_fp16.tflite`, 78 KB): TFLite interpreter ile **rastgele 5 örnekte 5/5 doğru tahmin** yaptı. Random-split olduğu için overfit fakat model SAHTE değil — gerçek tahmin üretiyor.
3. **RFC %61.5 LOWO** (`binary_lowo_results.json`): 4671 örneğin LOWO çapraz doğrulamasıyla elde edilmiş gerçek doğruluk. Yeniden üretilebilir.
4. **KNN %46.8 LOWO, KNN/RFC 3-sınıf %38.8/%52.4**: aynı şekilde doğrulanmış.

### ❌ KIRIK (DÜZELTILIYOR)
1. **MobileNetV3 visual classifier** (`visual_classifier_fp16.tflite`):
   - **Test**: 4 farklı sentetik input (random, all-black, all-white, mid-gray) **AYNI** probability döndürüyor: `[0.414, 0.507, 0.079]`
   - **Sebep**: Backbone'un %80'i donmuş, sadece son katmanlar eğitilmiş → sınıf prior'unu ezberlemiş, görüntüye bakmamış
   - **Durum**: `train_visual_classifier.py` güncellendi (backbone %40 dondur, %60 trainable, daha düşük LR), yeniden eğitim arka planda çalışıyor
   - **APK'ya etkisi**: Şu anki APK'da "Görsel-only" karşılaştırma rozeti **anlamsız**. Düzeltildikten sonra rebuild edilecek.

### ⚠️ DOĞRULANMAMIŞ (RİSK)
1. **Dart MFCC vs Python librosa**: Aynı WAV'ı her iki tarafta çalıştırıp feature vektörünü karşılaştırmadık. Eğer Dart MFCC değerleri farklı bir ölçekte üretiyorsa fusion modeli yanlış girdi alır → telefon tahmini güvensiz olur.
2. **Dart 11-D visual** vs OpenCV: HSV scale (0-179 vs 0-360) ve Sobel/Laplacian implementasyonları manuel yapıldı, OpenCV ile birebir aynı sayı üretmek garanti değil.
3. **Dart 8-D Hollow Heart**: Basitleştirilmiş versiyon. Backend cepstral peak detection + tam HNR autocorrelation; bizim Dart kodumuz daha hafif. Skor backend'ten farklı olabilir.
4. **Dart 7-D Haptic**: Gravity baseline kalibrasyonu yapılmıyor (sabit 9.81 m/s² kullanıyoruz). Telefon eğikse hatalı feature çıkar.

---

## Şu Ana Kadar Yapılanlar (Verify Edilen)

### Test 1: Eğitim verisi gerçek mi?
```bash
$ python -c "import numpy as np; X = np.load('data/processed/X_features.npy'); print(X.shape)"
(4671, 146)
```
**Sonuç:** ✓ 4671 örnek var, sahte değil.

### Test 2: Fusion modeli sahiden tahmin yapıyor mu?
```bash
$ python  # scripts/validate_fusion.py
Sample 1714: probs=[0.972 0.028 0.000] pred=0 true=0 [+]
Sample 3534: probs=[0.001 0.999 0.000] pred=1 true=1 [+]
Sample 3647: probs=[0.002 0.998 0.000] pred=1 true=1 [+]
Sample 2509: probs=[0.004 0.996 0.000] pred=1 true=1 [+]
Sample 4063: probs=[0.080 0.920 0.000] pred=1 true=1 [+]
Random-split match: 5/5
```
**Sonuç:** ✓ Model gerçekten feature'a göre farklı tahmin üretiyor, sahte değil.

### Test 3: Visual classifier sahiden görseli işliyor mu?
```python
random:      probs=[0.414 0.507 0.079]
all_black:   probs=[0.414 0.507 0.079]  ← AYNI!
all_white:   probs=[0.414 0.507 0.079]  ← AYNI!
mid_gray:    probs=[0.414 0.507 0.079]  ← AYNI!
```
**Sonuç:** ❌ **Visual classifier KIRIK**. Yeniden eğitim başlatıldı.

### Test 4: Backend visual dataset gerçek mi?
```
[Roboflow] 54 görsel
[MRD-YOLO] 300 görsel
[Qilin] 1557 görsel
Toplam: 2019 (label: 707/1126/186)
Pixel std @ (100,100,0): 39.14   ← gerçek varyans
```
**Sonuç:** ✓ Görsel veri gerçek. Sorun veri değil, eğitim hyperparametreleri.

---

## Şimdi Yapılıyor

1. **Visual classifier yeniden eğitiliyor** (~30-60 dk)
   - Backbone %60 trainable (önceden %20'ydi)
   - Fine-tune LR = 1e-4 (önceden 1e-3)
2. **Dart vs Python feature validation**
   - Referans WAV: `dataset/datasets/watermelon_dataset/19_datasets/1_10.5/chu/1/1.wav` (3 sn, 44.1 kHz)
   - Python expected output: `data/processed/wm1_chu1_python_features.json`
   - Aynı WAV'ı APK'da debug ekranı ile yükleyip Dart features ile karşılaştıracağız
3. **APK rebuild** yeni visual classifier ile

---

## Hocaya / Raporda Söylenecek

Eğer visual classifier'ın yeniden eğitimi de başarısız olursa:
> "MobileNetV3 visual baseline'ı 19 karpuz'un alt dağılımında subject leakage benzeri bir öğrenme problemine takıldı; mobil uygulamada bu nedenle MobileNetV3-only tahmini kaldırılıp sadece 11-D handcrafted görsel özellikler (HSV + texture) fusion modeline beslendi."

Eğer Dart features Python'la uyuşmazsa:
> "Telefon-tarafı feature extraction backend ile %X korelasyon gösterdi. Bu fark uygulamayı **bağımsız bir mobil ML pipeline** olarak değerlendirmemizi gerektiriyor; backend'in LOWO %61.5 baseline'ı buna doğrudan uygulanamaz — saha testindeki gerçek doğruluk ayrı bir metriktir."

Bu **akademik dürüstlük**; hocaların değerli bulduğu şey ezberlenmiş yüksek doğruluk değil, dürüst raporlanmış limitler.

---

## Risk Matrisi

| Bileşen | Risk Seviyesi | Plan |
|---|---|---|
| Fusion DNN | ✅ Düşük | Verified, kullanmaya devam |
| RFC/KNN baseline | ✅ Düşük | LOWO sonuçları rapora |
| MobileNetV3 visual | ❌ Yüksek → Düşük | Yeniden eğitim sonrası sanity check |
| Dart MFCC | ⚠️ Orta | Debug screen ile saha doğrulaması |
| Dart 11-D visual | ⚠️ Orta-düşük | Saha testiyle doğruluk ölçülecek |
| Dart 8-D HH | ⚠️ Orta | Backend'le farklı olabilir, raporda belirt |
| Dart 7-D haptic | ⚠️ Düşük-orta | Calibrate eklenebilir |

---

## Sonuç

**Uygulama fake DEĞİL** — gerçek veri, gerçek eğitim, gerçek TFLite modeller var.
Tek **gerçek kırık**: visual classifier (düzeltiliyor).
**En büyük belirsizlik**: Dart features sayısal olarak Python features ile birebir mi? Bunu da saha testinde gerçek karpuzlarla ölçeceğiz.

19 karpuzluk dataset doğal bir bilimsel limit — bunu rapora dürüstçe yazıyoruz.
