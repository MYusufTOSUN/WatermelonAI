# Karpuz AI — APK Sürümleri

## watermelon_ai_v15_fixed.apk (EN GÜNCEL — bunu kur)

**Sürüm:** 2.2.0 (Doğruluk regresyonu düzeltmesi)
**Boyut:** ~87 MB
**Tarih:** 2026-06-11

### v15'te neler düzeldi
- 🔴 **v13/v14 regresyonu giderildi**: n_mels 64 → 128 geri alındı + agresif ses kırpma kaldırıldı. v13'te hız için yapılan bu iki değişiklik, fusion DNN'in eğitim dağılımıyla uyumsuz feature üretip gerçek olgun karpuzlara "Henüz ham → ALMA" dedirtiyordu.
- ⚡ **Gerçek darboğaz bulundu ve çözüldü**: 1-2 dakikalık analiz süresinin asıl sebebi Hollow Heart cepstrum hesabındaki O(N²) DCT'ydi (~17 milyar işlem). FFT-tabanlı cepstruma (O(N log N)) çevrildi → tam 3 saniyelik ses birkaç saniyede işleniyor, doğruluk bozulmadan.
- 🛡️ **Hollow Heart tetiğine çift onay**: SRR f₂ < 134 Hz tek başına yetmiyor; artık akustik HH skoru ≥ 0.45 de gerekiyor. İyi karpuzlara yanlış "İçi geçmiş" damgası kesildi.
- ➕ Mel filterbank sparse optimizasyonu (matematiksel birebir, bedava hız)

### Test talimatı
1. v14'ü kaldır, v15'i kur
2. **⋯ menü → "Geliştirici: doğrulama"** çalıştır → ortalama korelasyon **≥ 0.95** bekleniyor (v13/v14'te bozulmuştu)
3. Aynı 2 iyi karpuzla karpuz testini tekrar yap → **AL** çıkması bekleniyor
4. Kestiğin karpuzları Geçmiş'ten işaretle

---

## watermelon_ai_v13_simple.apk (önceki — DOĞRULUK SORUNLU, kullanma)

**Sürüm:** 2.1.0 (Halk-Dostu UI + Performans)
**Boyut:** ~87 MB
**Tarih:** 2026-06-09

### v13'te neler değişti — Halk Dostu + Hızlı

**Halk-dostu arayüz:**
- Ana ekran adı: "Karpuz Dedektifi" — "Olgun mu değil mi anlar"
- Tek büyük buton: "Karpuz testini başlat (3 adım · 1 dakika)"
- Sonuç ekranında BÜYÜK "AL" / "ALMA" verdict + tek satır neden
- "Detayları gör" expandable kapalı varsayılan (teknik kartlar gizli)
- Sınıf etiketleri: "Tam kıvamında" / "Henüz ham" / "Geçmiş olabilir"
- Geliştirici testleri (self-test + doğrulama) ana ekrandan kaldırıldı, ⋯ menüsünde

**Performans (1-2 dk → ~20-30 sn):**
- Aggressive audio trim (3 sn → 1.5 sn aktif bölge)
- Mel filterbank 128 → 64 (2x hızlanma)
- Shared full-audio FFT (dominant + entropi tek FFT)

**Diğer:**
- [`docs/PIPELINE_AUDIT.md`](../docs/PIPELINE_AUDIT.md) — Su şişesi vakası dahil tüm sayıların "gerçek mi?" denetimi

### Test akışı (halk düzeyinde)
1. APK'yı kur
2. "Karpuz testini başlat"a bas
3. Adım 1: Karpuzun fotoğrafını çek
4. Adım 2: Karpuza parmağınla vur (ses kaydı)
5. Adım 3: Telefonu karpuza yasla (titreşim)
6. Beklemen yeterli (~20-30 sn) → "AL" veya "ALMA" görünür
7. Kestiğinde Geçmiş ekranından sonucu işaretle

---

## watermelon_ai_v12_viliquid.apk (önceki)

**Sürüm:** 2.0.0 (Vi-Liquid + Görselden Kütle)
**Boyut:** ~87 MB
**Tarih:** 2026-05-20

### v12'de yeni — Tam Pipeline
1. **Adım 1: Foto** → Otomatik HSV segmentasyon + bbox + heuristic **kütle tahmini** (1.5-8 kg)
2. **Adım 2: Akustik** → Mikrofon kayıt → 120-D MFCC + 8-D HH
3. **Adım 3: Aktif Haptik (Vi-Liquid)** → **Telefonu titret** (3 sn) + IMU 100 Hz + **SRR 1600 Hz** + f₂ çıkar
4. **Sonuç:** Late Fusion = ML model (acoustic+visual+HH) **+** Vi-Liquid fiziksel kuralı (f₂ + EI)
   - Birleşik karar + Hollow Heart tetiği (f₂ < 134Hz + visual ripe → Defective)

### Yeni servis dosyaları
- `vibration_service.dart` — LRA motor titreşim kontrolü
- `srr_processor.dart` — Cubic interpolation 100→1600 Hz + FFT f₂ çıkarımı
- `mobile_fusion_engine.dart` — Backend `mobile_fusion.py`'ın Dart portu (EI normalization + HH trigger)
- `mass_estimator.dart` — Foto'dan heuristik kütle tahmini (HSV blob + bbox)

### Result ekranı yenilikleri
- **Combined Verdict** kartı (Vi-Liquid + ML fusion birleştirilmiş)
- **Vi-Liquid Aktif Haptic** kartı: f₂ + EI + Kütle + HH tetik durumu
- **ML Fusion** mini kartı (görsel + akustik + HH DNN)
- Foto'dan tahmin edilen kütle slider'da pre-fill

---

## watermelon_ai_v5_selftest.apk (önceki)

**Sürüm:** 1.3.0 (saha testi alternatifi)
**Boyut:** ~86 MB
**Tarih:** 2026-05-19

### v5'te neler değişti
- ✅ **Otomatik Saha Testi**: Ana ekranda yeni buton — 12 bundled Qilin WAV (5 Olgunlaşmamış, 5 Olgun, 2 İçi Geçmiş) üzerinde tam mobil pipeline'ı tek tıkla çalıştırır
- ✅ Sonuç ekranında: konfüzyon matrisi + sınıf bazında doğruluk + her örneğin tahmini
- 🎯 **Saha testi alternatifi**: Karpuz kesemezsen bu test ile pipeline'ın gerçek-veri üzerinde doğruluğunu kanıtlayabilirsin
- ✅ Brix etiketli bilinen örnekler, ground truth doğrudan dataset'ten

### Test akışı
1. APK'yı kur, aç
2. Ana ekran → "Otomatik saha testi (bundled Qilin)" butonuna bas
3. "12 bundled Qilin örneğini çalıştır" → bekle (~20 saniye)
4. Sonuç: 2-sınıf doğruluk + 3-sınıf doğruluk + konfüzyon matrisi
5. Screenshot al, rapora ek olarak koy

### Bu testin akademik anlamı
- Bu örnekler eğitim setinde olduğu için yüksek doğruluk beklenir (smoke test)
- **Genelleme metriği**: Backend pipeline LOWO baseline (RFC 2-sınıf %61.5)
- Bu mobil test pipeline'ın **çalıştığını** kanıtlar; LOWO ise gerçek genellemeyi
- İki sayı birlikte rapora gider

---

## watermelon_ai_v4_verified.apk (önceki)

**Sürüm:** 1.2.0 (verified pipeline)
**Boyut:** ~81 MB
**Tarih:** 2026-05-19

### v4'te neler değişti
- ✅ **Fusion modeli doğrulandı**: Python tarafında 5 random sample üzerinde 5/5 doğru tahmin → fusion DNN gerçekten çalışıyor, sahte değil
- ✅ **Visual classifier yeniden eğitildi**: Backbone'un %60'ı trainable, val %75'e yükseldi (önceden sınıf prior'a sıkışmıştı). Test %30 (subject leakage limit)
- ✅ **"Doğrulama testi (debug)" ekranı**: Ana ekrandan erişilebilir; bundled Qilin WAV ile Dart-tarafı MFCC'yi Python librosa referansla karşılaştırır, Pearson korelasyonu + MSE gösterir
- ⚠️ **Visual-only karşılaştırma gizlendi**: MobileNetV3 19-karpuz subject leakage'tan etkilenmeye devam ediyor; sadece **fusion modeli (multimodal)** kullanıcıya raporlanıyor
- 📊 Dürüst denetim raporu: [`docs/AUDIT.md`](../docs/AUDIT.md)

### Bu sürümde garantilenen
1. **4671 gerçek eğitim örneği** (Qilin dataset) — verified
2. **Fusion TFLite gerçek tahmin yapıyor** — verified (5/5 doğru)
3. **2-sınıf LOWO RFC %61.5** — verified (binary_lowo_results.json)
4. **Tüm öznitelik extraction kod tarafında implement** (DSP, MFCC, mel filterbank, FFT, HSV, Sobel) — sayısal eşdeğerlik debug ekranında ölçülebilir

### Ne hala doğrulanmadı (RİSK)
- Dart-tarafı MFCC'nin Python librosa ile birebir aynı sayı üretmesi (debug ekran bunu ölçer)
- Saha doğruluğu — kestiğin karpuzlarla işaretle, **Geçmiş ekranındaki "%X" rozeti** gerçek saha sayısını gösterir

---

## watermelon_ai_v3_multimodal_field.apk (önceki)

**Sürüm:** 1.1.0 (multimodal + saha testi modu)
**Boyut:** ~81 MB
**Tarih:** 2026-05-19

### v3'te yeni
- **Persistent history**: Tüm ölçümler SharedPreferences'a kayıt (uygulama kapansa da kalır, 50 ölçüme kadar)
- **Ground truth feedback**: Geçmiş ekranında her ölçümün altında "Kestin mi? Gerçek sonucu işaretle" butonu — kestikten sonra Olgun/Olgunlaşmamış/İçi Geçmiş işaretliyorsun
- **Saha doğruluğu rozeti**: Etiketlenen ölçümlerden otomatik saha doğruluğu hesaplanır (örn. "8/10 = %80")
- **Kalıcı dosya kaydı**: Foto + WAV artık `app documents/samples/` klasörüne yazılıyor (silinmez)
- **Silme / düzenleme**: Hatalı kayıt silinebilir, ground truth değiştirilebilir

---

## watermelon_ai_v2_multimodal.apk (önceki)

**Sürüm:** 1.0.0 (multimodal)
**Boyut:** ~80 MB
**Tarih:** 2026-05-19
**Hedef Android:** SDK 26+ (Android 8.0+)

### Özellikler
- **3-Adım Multimodal Capture Sihirbazı**:
  1. Foto çek (görsel 11-D feature)
  2. Vuruş sesini kaydet (akustik 120-D MFCC + spektral feature)
  3. Telefonu karpuza dokundur (ivmeölçer 7-D haptic feature)
- **Hollow Heart Detection** (8-D) akustik sinyalden
- **Fusion Model** (`fusion_model_fp16.tflite`) — 4 input → 3 sınıf
- **MobileNetV3 Visual Baseline** ile karşılaştırmalı sonuç gösterimi
- Tamamı **cihaz üzerinde**, internet bağlantısı **gerekmiyor**

### Kurulum
1. APK'yı telefona transfer et (USB / Drive / WhatsApp)
2. Ayarlar → Güvenlik → "Bilinmeyen kaynaklara izin ver"
3. APK'ya dokun, kur
4. İlk açılışta **Kamera + Mikrofon** izni ver

### Test Senaryosu
1. Karpuzu masaya/yere yatay koy
2. **Adım 1**: Karpuzu çerçevenin ortasına alıp foto çek
3. **Adım 2**: Telefonun mikrofonunu karpuza yaklaştır, parmak boğumuyla 2-3 kez sertçe vur (3 sn kaydı)
4. **Adım 3**: Telefonun arkasını karpuza yasla, 3 saniye hafif basınç uygula
5. **Sonuç**: Fusion + Visual-only karşılaştırması + f2/HH/Temas diagnosticleri

### Beklenen Doğruluk
- Backend LOWO baseline (4671 örnek, 19 karpuz): **RFC 2-sınıf %61.5**
- Saha doğruluğu: kullanıcı testinde ölçülecek

### Bilinen Sınırlamalar
- MFCC parametreleri librosa ile birebir aynı olmayabilir → akustik feature kalitesi backend'le %85-95 korelasyon (saha testiyle ölçülecek)
- Hollow Heart 8-D feature backend'deki tam pipeline'ın basitleştirilmiş versiyonu (cepstral peak detection daha hafif)
- 19 karpuz dataseti subject leakage limiti — LOWO baseline'ı literatürdeki small-N akustik karpuz çalışmalarıyla tutarlı

---

## watermelon_ai_release.apk (Eski — silinebilir)

İlk sürüm, sadece görsel sınıflandırma yapan ve eski paket yapısına sahip APK.
Önerilmez, yeni sürümü kullan.
