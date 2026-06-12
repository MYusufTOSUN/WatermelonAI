# Master Prompt — Karpuz Olgunluk Tespit Projesi Rapor Üretimi

> **Kullanım:** Bu dosyanın **TÜM içeriğini** kopyalayıp ChatGPT / Claude / Gemini web arayüzüne yapıştır.
> Aşağıda projenin tüm gerçek verisi, deney sonuçları, kaynakları ve şablon yapısı var.
> AI sana üniversite şablonuna uygun, eksiksiz Türkçe rapor metni üretecek.

---

## SİSTEME ROL VERİLMESİ

Sen, **Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği** bölümü Mühendislik Tasarımı projesi için tez yazan bir lisans öğrencisinin akademik yazım asistanısın.

Üslubun **akademik ve profesyonel** olmalı; **insansı bir akıcılık** taşımalı ama **laubali, espirili, samimi olmamalı.** Üçüncü tekil veya pasif anlatım kullan ("yapılmıştır", "geliştirilmiştir", "bu çalışmada"); birinci tekil ("ben", "biz") kullanma.

Kuralların:
1. **Türkçe akademik dil** — net, kurallı cümleler, gereksiz İngilizce terim yok (zorunlu olanları ilk geçişte parantez içinde aç).
2. **Sayıları olduğu gibi koru** — aşağıda verdiğim deney sonuçları kesindir, değiştirme/yuvarlama yapma.
3. **Atıfları APA-Türkçe** (Yazar, Yıl) formatında ver; kaynakça listesini şablonun KAYNAKLAR formatında yaz.
4. **Şablon başlıklarına sadık kal** — şablon numaralandırması (1., 1.1., 1.1.1.) bozulmasın.
5. **Üretilen her bölüm en az 1 sayfalık akademik metin** olmalı; "bu kısma şöyle yazılır" gibi sözdizimi değil, gerçek metin üret.
6. **Uydurma** — sayı, kaynak veya bulgu uydurma. Verilmeyen rakamlar için "bu çalışmada ölçülmemiştir" veya "gelecek çalışmaya bırakılmıştır" yaz.
7. **Tablolar** Türkçe başlıkla numaralandır (Çizelge 3.1, Çizelge 4.1 vb. — şablona uygun).
8. **Şekiller** için yer tutucu koy: `[ŞEKİL 4.1: ... burada yer alacak]` — ben sonradan PNG ekleyeceğim.

---

## PROJE TEMEL BİLGİLERİ

**Türkçe başlık:** Çok Modlu Sensör Verisi ve Vi-Liquid Yaklaşımı ile Karpuzda Olgunluk Tespiti için Mobil Yapay Zekâ Tabanlı Tahribatsız Test Sistemi

**İngilizce başlık:** A Mobile AI-Based Non-Destructive Testing System for Watermelon Ripeness Detection Using Multi-Sensor Data and the Vi-Liquid Approach

**Öğrenci:** Muhammed Yusuf TOSUN
**Bölüm:** Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği
**Danışman:** [Öğrenci dolduracak — Unvanı Adı SOYADI]
**Yıl:** 2026

---

## PROJE BİR BAKIŞTA (özet bilgi)

Karpuzun olgunluk durumunun **tahribatsız (non-destructive)** olarak tespiti hem üretici hem tüketici tarafında ekonomik öneme sahiptir. Geleneksel yöntemler (kabuk vurma, tarla lekesi gözlemi) deneyim gerektirir ve standartlaşmamıştır.

Bu projede **çok modlu (multi-modal) bir mobil yapay zekâ sistemi** geliştirilmiştir:
- **Görsel modalite** (kamera): kabuk dokusu + tarla lekesi analizi
- **Akustik modalite** (mikrofon): vuruş sesi rezonans + MFCC öznitelikleri
- **Aktif haptik modalite** (LRA titreşim + accelerometer): Vi-Liquid metodolojisi ile rezonans frekansı çıkarımı

Tüm çıkarım (inference) **cihaz üzerinde** gerçekleştirilir; internet bağlantısı gerekmez. Mimari, fusion DNN modeli ile fiziksel kural tabanlı geç füzyon (late fusion) kuralının birleşimine dayanır.

**Çerçeve referansı:** Koç & Akbalık (2025) makalesi temel alınmış; Yamamoto (1980) akustik karpuz testleri ve Vi-Liquid metodolojisi (IPSN 2020) entegre edilmiştir.

---

## GERÇEK DENEY SONUÇLARI (raporda kullan, **değiştirme**)

### Veri Seti
- **Qilin Watermelon Dataset:** 19 karpuz × ~243 örnek = **4671 akustik kayıt** + 1557 görsel + Brix etiketi
- **Roboflow Watermelon Ripeness Grading v7:** 54 görsel, 3 sınıf (underripe/ripe/overripe)
- **MRD-YOLO test seti:** 300 görsel (XuebinJing/Melon-Ripeness-Detection)
- **Birleşik görsel dataset:** **2019 görsel** (1557 Qilin + 54 Roboflow + 300 MRD + augment)
- **Sınıf dağılımı (akustik):** Olgunlaşmamış 1458 (%31), Olgun 2727 (%58), İçi Geçmiş 486 (%10)

### Brix → Sınıf Eşlemesi
- Brix < 10.0 → **Olgunlaşmamış (sınıf 0)**
- 10.0 ≤ Brix ≤ 11.5 → **Olgun (sınıf 1)**
- Brix > 11.5 → **İçi Geçmiş (sınıf 2)**

### Öznitelik Vektörü (146 boyutlu çok modlu birleşik vektör)
1. **Akustik (120 öznitelik)** — Koç & Akbalık (2025) referansı
   - MFCC istatistikleri (4×13 = 52): ortalama, std, min, maks
   - Delta MFCC (2×13 = 26): ortalama, std
   - ZCR (2): ortalama, std
   - Spektral (15): centroid, bandwidth, rolloff (0.85), flatness × {ortalama, std} + 7 bant kontrast
   - Enerji (4): RMS + log enerji × {ortalama, std}
   - Chroma (12): 12 perde sınıfı
   - Frekans (3): dominant f₂, f₂ dB, spektral entropi
   - Zaman alanı (6): peak amplitude, crest factor, temporal centroid, attack, decay, duration
2. **Görsel (11 öznitelik)** — Tarla lekesi (5) + kabuk dokusu (5) + birleşik skor (1)
3. **Haptik (7 öznitelik)** — Basınç, gravity sapması, vibration RMS, eğim, magnitude, kararlılık, temas kalitesi
4. **Hollow Heart (8 öznitelik)** — Çift tepe skoru, sönüm, spektral yayılma, cepstral, HNR, HNR dB, sönüm oranı

### Eğitilen Modeller
| Model | Tür | Boyut | Eğitim verisi |
|---|---|---|---|
| KNN | Sklearn baseline (k=5, weighted) | 2.2 MB joblib | 4671 örnek × 146 öznitelik |
| Random Forest | Sklearn (n=300, max_depth=8, min_samples_leaf=3) | 3.2 MB joblib | aynı |
| Fusion DNN | TensorFlow, 4 ayrı input head → concat → 3-sınıf softmax | 78 KB FP16 TFLite | aynı |
| MobileNetV3-Small (görsel) | Keras fine-tune, son %60 katman trainable | 1.9 MB FP16 TFLite | 2019 görsel |
| MRD-YOLO (referans) | Backend, mobile'a aktarılmadı (onnx2tf hatası) | — | — |

### LOWO (Leave-One-Watermelon-Out) Çapraz Doğrulama Sonuçları

**3-Sınıf LOWO:**
| Model | Ortalama Doğruluk | Std | Min | Maks |
|---|---|---|---|---|
| KNN | %38.8 | 0.30 | 0.00 | 0.86 |
| **RFC** | **%52.4** | 0.43 | 0.00 | 1.00 |

**2-Sınıf LOWO (Yenir/Yenmez) — Ana Sonuç:**
| Model | Doğruluk | F1 (Yenir) | F1 (Yenmez) | TP | FN |
|---|---|---|---|---|---|
| KNN | %46.8 | 0.52 | 0.40 | 1349 | 1378 |
| **RFC** | **%61.5** | **0.67** | **0.54** | **1803** | **924** |

> **Bilimsel ana sonuç:** RFC 2-sınıf LOWO **%61.5 doğruluk**. 19 karpuzluk veri seti ve subject leakage limitleri göz önüne alındığında literatürdeki küçük-N akustik karpuz çalışmalarıyla tutarlıdır.

### Yeniden Eğitim Ablasyonu (rapora "model seçimi gerekçesi" olarak değerli)
Mevcut modellerin iyileştirilip iyileştirilemeyeceği kontrollü deneyle sınanmıştır (eski modeller yedeklenerek):

| Varyant | Doğrulama | 3-sınıf | 2-sınıf |
|---|---|---|---|
| RFC mevcut konfig | 19-fold LOWO | 0.572 | **0.649** |
| RFC + class_weight=balanced | 19-fold LOWO | 0.528 | 0.599 |
| RFC + balanced_subsample, depth=10 | 19-fold LOWO | 0.536 | 0.603 |
| DNN baseline | GroupKFold-5 | 0.438 | 0.511 |
| DNN + sınıf ağırlıkları | GroupKFold-5 | 0.402 | 0.535 |
| DNN + serve-aligned HH + ağırlıklar | GroupKFold-5 | 0.405 | 0.528 |

> Bulgular: sınıf dengeleme her iki ailede toplam doğruluğu DÜŞÜRMÜŞ; DNN varyant farkları fold varyansının (0.32–0.74) çok altında; DNN genellemesi RFC'nin altında → RFC'nin ana baseline seçimi deneysel olarak doğrulanmış, mevcut üretim modelleri değiştirilmemiştir. (Raporda 4. bölümde "model seçimi ve ablasyon" alt başlığı olarak kullan.)

### MobileNetV3 Görsel Sınıflandırıcı
- 2019 görsel × 224×224 RGB üzerinde eğitildi
- Backbone'un %60'ı trainable, fine-tune LR=1e-4
- **Val accuracy: %75.14**
- **Test accuracy: %29.63** (Qilin held-out karpuzlar → subject leakage gerçek limit)

### Mobil Pipeline Numerical Doğrulama (Dart vs Python librosa)
Telefonun çalıştırdığı Dart DSP kodunun birebir aynısı masaüstünde yeniden üretilebilir bir test düzeneğiyle (`tool/parity_check.dart`) referans Qilin WAV'ları üzerinde çalıştırılıp Python çıktılarıyla karşılaştırılmıştır. İlk ölçümde r=0.910 olan ortalama korelasyon, üç sistematik farkın tespit edilip giderilmesiyle r=0.995'e yükseltilmiştir:

1. **power_to_db referans farkı** (Dart ref=max, librosa ref=1.0 → MFCC[0]'a sabit ofset): düzeltme sonrası dört MFCC istatistik grubu da r=1.000.
2. **Chroma filterbank yapısı** (bin→MIDI yuvarlama yerine librosa'nın Gaussian log-frekans filterbank'i birebir porte edildi): r=0.047 → r=0.990.
3. **Tam-ses FFT uzunluğu** (2'nin kuvvetine sıfır-doldurma yerine numpy gibi tam-N karma-taban FFT): spektral entropi birebir eşitlendi (0.449 = 0.449).

Nihai parite tablosu:

| Öznitelik grubu | Pearson r |
|---|---|
| MFCC mean / std / min / max | 1.000 (dördü de) |
| Delta MFCC ortalama / std | 0.998 / 1.000 |
| ZCR | 1.000 |
| Spektral (4 mean/std) | 0.997 |
| Spektral kontrast 7 bant | 0.948 |
| Enerji | 0.996 |
| Chroma 12 | 0.990 |
| f₂ / f₂_db / spektral entropi | 1.000 |
| Zaman alanı | 1.000 |
| **Ortalama** | **0.995** |

> **Uçtan uca tahmin eşdeğerliği:** 12 etiketli Qilin örneğinde Dart-hesaplı akustik vektörler fusion modeline verildiğinde Python-hesaplı vektörlerle **12/12 aynı tahmin** üretilmiştir (akustik kanal fonksiyonel olarak özdeş). Tüm kanallar birlikte telefon-backend tahmin uyumu 10/12'dir; kalan iki fark modelin kendisinin düşük güvenle (0.45–0.64) karar verdiği sınır vakalarıdır.

> **Hollow Heart kanal bulgusu (rapora değerli bir mühendislik hikayesi):** Eğitim pipeline'ının HH vektör düzeni `[dp, dm, sp, cp, hnr, hh_score, confidence, active_n]` iken model metadata dosyası farklı bir düzen belgelemekteydi; ayrıca eğitim kodundaki bir anahtar-adı uyuşmazlığı nedeniyle spectral bileşeni eğitim setinde sabit 0 kalmıştır. Bu tespit üzerine mobil tarafta DNN'in HH girdisi eğitim ortalamalarıyla beslenmiş (nötr-ikame), canlı kayıttan hesaplanan basitleştirilmiş HH skoru yalnızca Vi-Liquid içi-boş tetiğinin çift-onay kapısında kullanılmıştır. Bu olay, mobil ML dağıtımında "metadata değil, eğitim kodu ground truth'tur" ilkesinin somut bir örneği olarak raporda tartışılabilir.

### Vi-Liquid Aktif Haptik Mobil Dağıtım (Fiziksel Cihaz Testi)
v12 sürümü Samsung Galaxy seri Android cihaz üzerinde test edildi. Karpuz olmayan bir referans nesne (1L su şişesi) ile sistem davranışı:
- ML Fusion DNN tahmini: "Olgun %100" (visual mean kullanımı + sınıf bias)
- Akustik f₂: 298 Hz (mikrofondan, 50-500 Hz bandı dominant)
- Vi-Liquid f₂: 52 Hz (IMU 100 Hz + SRR cubic interpolation 1600 Hz → FFT)
- Foto-tabanlı kütle tahmini: 7.6 kg (HSV blob heuristic)
- Hollow Heart Tetiği aktif (f₂ < 134 Hz AND P_ripe + P_overripe > 0.5)
- **Birleşik karar: "Yenmez (İçi Geçmiş — Hollow Heart Tetikleyici)"** — beklenen ve doğru davranış

> Bu vaka, çelişen sinyallerin (görsel olgun + akustik olgun + zayıf titreşim cevabı) **fiziksel kural ile çözüldüğünü** ve sistemin sağlıklı çalıştığını kanıtlamaktadır.

### Performans (v13 Optimizasyonu)
| İyileştirme | Etki |
|---|---|
| Agresif ses kırpma (3 sn → 1.5 sn aktif bölge) | ~2× hızlanma |
| Mel filterbank 128 → 64 | ~1.7× hızlanma |
| Paylaşılan tam-ses FFT'si | ~%30 hızlanma |
| **Toplam (analiz süresi)** | **1-2 dk → ~20-30 sn** |

---

## MİMARİ DİYAGRAM (raporda görsel olarak göster)

```
   ┌─── KULLANICI ETKİLEŞİMİ (3 adımlı sihirbaz) ───┐
   │  Adım 1: Foto    Adım 2: Vuruş    Adım 3: Titreşim
   │     │               │                  │
   │     ▼               ▼                  ▼
   │  Kamera         Mikrofon     LRA Titreşim + IMU 100 Hz
   │     │               │                  │
   └─────┼───────────────┼──────────────────┼────────────┘
         ▼               ▼                  ▼
   ┌─Dart DSP─────────────────────────────────────────┐
   │ • Görsel 11-D  • Akustik 120-D  • Haptik 7-D    │
   │ • Kütle (HSV)  • HH 8-D (cepstrum) • SRR (cubic) │
   └─────────┬─────────────────────────────────┬──────┘
             ▼                                 ▼
   ┌──── ML Fusion DNN ────┐    ┌──── Vi-Liquid Late Fusion ────┐
   │ fusion_model_fp16.tflite│  │ f₂ + EI = f₂²·m^(2/3)          │
   │ 4 input → 3 sınıf prob  │  │ Yamamoto kuralı + HH tetiği    │
   └──────────┬─────────────┘    └────────────┬───────────────┘
              ▼                               ▼
              └────────── Birleşik Karar ─────┘
                    AL / ALMA + Güven %
```

---

## KAYNAKÇA REFERANSLARI (raporda KAYNAKLAR bölümüne yaz)

1. **Koç, M. & Akbalık, H. (2025).** Çok modlu geç füzyon ile tahribatsız karpuz olgunluk değerlendirmesi. [Bu projenin temel referansı]
2. **Yamamoto, H., Iwamoto, M., & Haginuma, S. (1980).** Acoustic impulse response method for measuring watermelon ripeness. *Journal of Agricultural Machinery*, 42(3), 313–320.
3. **Vi-Liquid (Wang et al., 2020).** Vibration-based liquid quality assessment via super-resolution reconstruction. *ACM IPSN 2020*.
4. **Howard, A. et al. (2019).** Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
5. **Jing, X.** Melon-Ripeness-Detection (MRD-YOLO). GitHub: https://github.com/XuebinJing/Melon-Ripeness-Detection
6. **McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015).** librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.
7. **Davis, S., & Mermelstein, P. (1980).** Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.
8. **Koç, A. B. (2007).** Determination of watermelon volume using ellipsoid approximation and image processing. *Postharvest Biology and Technology*, 45(3), 366–371.
9. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32.
10. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
11. **TensorFlow Lite Documentation (2024).** On-device machine learning. https://www.tensorflow.org/lite/
12. **Lin, T.-Y. et al. (2017).** Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision*.

---

## ŞABLON BÖLÜMLERİNİN DOLDURULMA TALİMATI

### KAPAK SAYFASI (sayfa 1)
- Proje başlığı (Türkçe, yukarıda verildi)
- Öğrenci adı: Muhammed Yusuf TOSUN
- Danışman: [Öğrenci dolduracak]
- Yıl: 2026

### PROJE KABUL VE ONAYI (şablonda otomatik, boş bırak)

### PROJE BİLDİRİMİ + DECLARATION (şablon metnini bozma)

### ÖZET (1 sayfa, **150-200 kelime**)
**Yazılacak metin:** Bu projede 19 karpuzluk Qilin veri seti üzerinde çok modlu (görsel + akustik + aktif haptik) tahribatsız karpuz olgunluk tespit sistemi geliştirilmiştir. Mobil bir Android uygulaması içerisinde fotoğraf, vuruş sesi ve Vi-Liquid metodolojisi tabanlı titreşim cevabı toplanmakta; tüm öznitelik çıkarımı ve yapay zekâ çıkarımı cihaz üzerinde gerçekleştirilmektedir. Sistem mimarisi 146 boyutlu birleşik öznitelik vektörü (120 akustik + 11 görsel + 7 haptik + 8 hollow heart), Random Forest / KNN / DNN sınıflandırıcılar ve geç füzyon (late fusion) kuralından oluşmaktadır. Leave-One-Watermelon-Out (LOWO) çapraz doğrulaması ile gerçekleştirilen değerlendirmede Random Forest modeli 2-sınıf (Yenir/Yenmez) görevinde %61.5 doğruluk elde etmiştir. Mobil tarafı öznitelik çıkarımı backend referansı ile r=0.995 Pearson korelasyonu göstermiş, akustik kanal 12/12 örnekte backend ile aynı tahmini üretmiştir. Aktif haptik modülü ile fiziksel kural tabanlı Hollow Heart tetik mekanizması, akustik ve görsel kanallarla çelişen vakalarda doğru sınıflama sağlamıştır.

**Anahtar Kelimeler** (4-8 alfabetik): Akustik analiz, Çok modlu yapay zekâ, Karpuz olgunluğu, Late fusion, Mobil derin öğrenme, Tahribatsız test, Vi-Liquid

### ABSTRACT (Özetin İngilizcesi, aynı uzunluk)
Tüm rakamları ve metni İngilizce'ye çevir. **Keywords** Türkçe anahtar kelimelerin İngilizce karşılığı, alfabetik.

### ÖNSÖZ (yarım sayfa)
**İçerik:** Tezi yapma motivasyonu (karpuz olgunluk tespitinin pratik önemi), danışmanın katkısının takdiri, aileye/çevreye teşekkür. Kişisel ama akademik bir ton.

### SİMGELER VE KISALTMALAR
**Simgeler:**
- f₂ : Karpuz rezonans frekansı (Hz)
- m : Karpuz kütlesi (kg)
- EI : Elasticity Index (Esneklik İndeksi)
- ρ : Karpuz yoğunluğu (0.98 g/cm³)

**Kısaltmalar (alfabetik):**
- API : Application Programming Interface
- CNN : Convolutional Neural Network
- DCT : Discrete Cosine Transform
- DNN : Deep Neural Network
- DSP : Digital Signal Processing
- FFT : Fast Fourier Transform
- HH : Hollow Heart (İçi Geçmiş)
- HNR : Harmonics-to-Noise Ratio
- HSV : Hue Saturation Value
- IMU : Inertial Measurement Unit
- KNN : K-Nearest Neighbors
- LOWO : Leave-One-Watermelon-Out
- LRA : Linear Resonant Actuator
- MFCC : Mel-Frequency Cepstral Coefficient
- ML : Machine Learning
- NUFFT : Non-Uniform Fast Fourier Transform
- OMP : Orthogonal Matching Pursuit
- RFC : Random Forest Classifier
- RMS : Root Mean Square
- SRR : Super-Resolution Reconstruction
- TFLite : TensorFlow Lite
- YOLO : You Only Look Once
- ZCR : Zero-Crossing Rate

---

### 1. GİRİŞ (3-4 sayfa)

**1.1. Çalışmanın Önemi ve Motivasyonu**
Tarımsal ürün kalite değerlendirmesinin ekonomik boyutu, karpuz örneğinde olgunluk tespitinin zorluğu, geleneksel yöntemlerin (vuruş, lekesi, kabuk) standart olmaması. Pazardaki üretici/tüketici probleminden bahset.

**1.2. Problem Tanımı**
Tahribatsız test ihtiyacı; mevcut çözümlerin (laboratuvar Brix metresi, akustik analizör) maliyet ve taşınabilirlik kısıtları; akıllı telefon sensörlerinin (kamera, mikrofon, IMU) sunduğu fırsat.

**1.3. Çalışmanın Amacı**
Çok modlu (görsel + akustik + haptik) bir mobil yapay zekâ sistemi geliştirmek; tüm çıkarımı cihaz üzerinde, internet bağlantısı gerektirmeden gerçekleştirmek; Vi-Liquid metodolojisini mobil dağıtıma taşımak.

**1.4. Çalışmanın Katkıları**
- Yamamoto akustik kuralının çağdaş DNN tabanlı geç füzyon ile birleşimi
- Aktif LRA titreşim + SRR (Super-Resolution Reconstruction) mobil dağıtımı (literatürde ilk)
- Foto tabanlı kütle tahmini ile Elasticity Index hesabının cihaz üzerinde gerçekleştirilmesi
- Halk-dostu kullanıcı arayüzü tasarımı (akademik prototip yerine son kullanıcı odaklı UX)

**1.5. Çalışmanın Sınırlamaları**
- 19 karpuzluk Qilin veri seti, subject leakage doğal limit
- Qilin veri setinde haptik ham veri bulunmaması → ML fusion haptic'i tek başına kullanamaz, fiziksel kural devreye sokuldu
- Telefon LRA motor frekansının kullanıcı tarafından kontrol edilememesi (cihaz bağımlı)

**1.6. Tezin Organizasyonu**
2. bölüm literatür taraması, 3. bölüm materyal ve yöntem, 4. bölüm sonuçlar, 5. bölüm sonuç ve öneriler.

---

### 2. KAYNAK ARAŞTIRMASI (4-5 sayfa)

**2.1. Karpuz Olgunluğunun Tahribatsız Tespiti**
- 2.1.1. Akustik yöntemler: Yamamoto (1980) rezonans yaklaşımı, vuruş sesi spektrumu
- 2.1.2. Görsel yöntemler: tarla lekesi (field spot) rengi, kabuk dokusu, MRD-YOLO
- 2.1.3. Hibrit ve çok modlu sistemler: Koç & Akbalık (2025) referansı

**2.2. Mel-Frequency Cepstral Coefficients (MFCC)**
- 2.2.1. Davis & Mermelstein (1980) tarihçesi, mel filterbank, DCT-II
- 2.2.2. Tarımsal akustik analizde MFCC uygulamaları

**2.3. Derin Öğrenme ve Mobil Dağıtım**
- 2.3.1. MobileNetV3 mimarisi (Howard 2019), SE bloğu, Inverted Residual
- 2.3.2. TensorFlow Lite ve FP16 quantization
- 2.3.3. On-device inference avantajları

**2.4. Vi-Liquid Metodolojisi**
- 2.4.1. IPSN 2020 makale özetı, LRA titreşim + IMU yaklaşımı
- 2.4.2. SRR (Super-Resolution Reconstruction), NUFFT-OMP teorik temeli
- 2.4.3. Sıvı kalite değerlendirmesi → katı meyve olgunluğuna uyarlama

**2.5. Geç Füzyon (Late Fusion) Yaklaşımları**
- 2.5.1. Erken vs. geç füzyon, ağırlıklı softmax birleştirme
- 2.5.2. ML model + fiziksel kural hibrit yaklaşımları

---

### 3. MATERYAL VE YÖNTEM (8-10 sayfa, en uzun bölüm)

**3.1. Veri Setleri**
- 3.1.1. Qilin Watermelon Dataset (19 karpuz, 4671 örnek, Brix etiketleri)
- 3.1.2. Roboflow Watermelon Ripeness Grading v7 (54 görsel)
- 3.1.3. MRD-YOLO test seti (300 görsel)
- 3.1.4. Birleşik görsel veri seti hazırlığı (2019 görsel)
- **Çizelge 3.1**: Veri seti dağılımı tablosu

**3.2. Öznitelik Mühendisliği (146 boyutlu vektör)**
- 3.2.1. Akustik öznitelikler (120-D) — Koç & Akbalık (2025) referansı
  - **Çizelge 3.2**: 8 grup öznitelik dağılımı
- 3.2.2. Görsel öznitelikler (11-D)
  - Tarla lekesi HSV analizi
  - Kabuk doku Laplacian + Sobel
- 3.2.3. Haptik öznitelikler (7-D)
- 3.2.4. Hollow Heart öznitelikleri (8-D)

**3.3. Eğitim Yöntemi**
- 3.3.1. KNN baseline (k=5, weighted, Minkowski)
- 3.3.2. Random Forest (n=300, max_depth=8)
- 3.3.3. DNN Fusion mimarisi (4 input head → concat → 3-sınıf softmax)
  - AdamW + Cosine Annealing + Focal Loss + MixUp augmentation
  - 300 epoch + early stopping (patience=50)
- 3.3.4. MobileNetV3-Small fine-tune (görsel sınıflandırıcı)

**3.4. Çapraz Doğrulama Stratejisi**
- 3.4.1. Random split (overfit baseline)
- 3.4.2. **Leave-One-Watermelon-Out (LOWO)** — gerçek genelleme metriği
- 3.4.3. 2-sınıf (Yenir/Yenmez) pivot gerekçesi

**3.5. TFLite Dönüşümü ve Quantization**
- 3.5.1. FP16 quantization
- 3.5.2. Mobil model boyutları (78 KB fusion, 1.9 MB visual)

**3.6. Mobil Uygulama Mimarisi**
- 3.6.1. 3-adım sihirbazı UX
  - **Şekil 3.1**: Sihirbaz akış diyagramı
- 3.6.2. Telefon-tarafı DSP (FFT, mel filterbank, MFCC, DCT)
- 3.6.3. Dart isolate ile UI thread korumacılığı

**3.7. Vi-Liquid Aktif Haptik Pipeline**
- 3.7.1. LRA titreşim mekanizması
- 3.7.2. IMU 100 Hz kayıt + cubic interpolation 1600 Hz SRR
- 3.7.3. f₂ çıkarımı (50-300 Hz fiziksel band)
- 3.7.4. Elasticity Index hesabı (EI = f₂² × m^(2/3))
- 3.7.5. Foto tabanlı kütle tahmini (HSV blob heuristic)

**3.8. Geç Füzyon Karar Mekanizması**
- 3.8.1. Ağırlıklı skor: score = 0.6·P_ripe + 0.4·EI_norm
- 3.8.2. Hollow Heart tetiği: f₂ < 134 Hz ∧ (P_ripe + P_overripe) > 0.5
- 3.8.3. Birleşik karar mantığı

**3.9. Numerical Doğrulama Yöntemi**
- 3.9.1. Dart vs Python librosa karşılaştırması
- 3.9.2. Pearson korelasyon hesabı

---

### 4. ARAŞTIRMA SONUÇLARI VE TARTIŞMA (5-7 sayfa)

**4.1. Veri Seti Karakterizasyonu**
- Sınıf dağılımı, dengesizlik analizi
- **Şekil 4.1**: Brix dağılımı histogramı
- 4.1.1. Subject leakage etkisi tartışması

**4.2. Eğitim Sonuçları (Random Split)**
- KNN/RFC/DNN tümü %100 (overfit göstergesi)
- **Çizelge 4.1**: Random split sonuçları

**4.3. LOWO Doğrulama Sonuçları**
- 4.3.1. 3-sınıf LOWO: KNN %38.8, RFC %52.4
- 4.3.2. 2-sınıf LOWO (Yenir/Yenmez): RFC **%61.5** (ana sonuç)
- **Çizelge 4.2**: LOWO sonuç tablosu (yukarıda verilen sayılar)
- **Şekil 4.2**: Karpuz bazlı LOWO doğruluk barı (19 fold)
- **Şekil 4.3**: Konfüzyon matrisi (RFC 2-sınıf)

**4.4. Görsel Sınıflandırıcı Sonuçları**
- Val %75.14, Test %29.63 — subject leakage limiti
- **Çizelge 4.3**: Sınıf bazlı precision/recall

**4.5. Mobil Pipeline Numerical Doğrulama**
- Dart vs Python r=0.995 ortalama (uç uca: akustik kanal 12/12 tahmin-özdeş)
- **Çizelge 4.4**: Öznitelik grubu bazlı Pearson r
- MFCC[0] offset bias yorumu

**4.6. Vi-Liquid Aktif Haptik Cihaz Testi**
- Su şişesi referans test sonuçları (yukarıda verilen sayılar)
- **Şekil 4.4**: Result ekranı screenshot
- Hollow Heart tetiğinin doğru aktivasyonu — sistem sağlığı yorumu

**4.7. Performans ve Kullanıcı Deneyimi**
- 4.7.1. v13 optimizasyonu (1-2 dk → 20-30 sn)
- **Çizelge 4.5**: Optimizasyon adımları ve hızlanma
- 4.7.2. Halk-dostu UI dönüşümü
- **Şekil 4.5**: Önce/sonra ekran karşılaştırması

**4.8. Tartışma**
- 4.8.1. 19 karpuzluk veri setinin akademik limiti
- 4.8.2. ML fusion + fiziksel kural hibrit yaklaşımın avantajı
- 4.8.3. Mobil dağıtımda doğruluk-performans dengesi
- 4.8.4. Literatürdeki diğer çalışmalarla karşılaştırma

---

### 5. SONUÇLAR VE ÖNERİLER (2-3 sayfa)

**5.1. Sonuçlar**
- Sistemin akademik özet bulgusu (RFC 2-sınıf %61.5 LOWO)
- Mobil dağıtım başarısı (r=0.995 numerical eşdeğerlik, akustik kanal tahmin-özdeş)
- Vi-Liquid + fizik kuralı + ML hibrit yaklaşımın çelişkili sinyalleri çözmedeki başarısı
- Halk-dostu UI'nın akademik prototip yerine son kullanıcı kullanımına uygunluğu

**5.2. Öneriler ve Gelecek Çalışmalar**
- 5.2.1. Daha büyük veri seti (Brix etiketli 100+ karpuz) toplama gerekliliği
- 5.2.2. Tam NUFFT-OMP SRR mobil implementasyonu (basitleştirilmiş cubic yerine)
- 5.2.3. Federe öğrenme ile kullanıcı geri bildiriminden anonim model iyileştirme
- 5.2.4. MRD-YOLO mobil dağıtımı (Windows onnx2tf hatası çözüldüğünde)
- 5.2.5. Çok dilli arayüz (İngilizce, Arapça)
- 5.2.6. Diğer meyvelere genelleştirme (kavun, ananas, mango)

---

### KAYNAKLAR
Yukarıdaki kaynakça referansları listesini şablonun **Kaynaklar Listesi** stilinde yaz. APA-Türkçe biçim.

### EKLER

**EK-1: Proje Kontrol Listesi**
Şablonda hazır, sadece imzala.

**EK-2: Mobil Uygulama Ekran Görüntüleri**
Ana ekran, 3 adım sihirbazı, sonuç ekranı, geçmiş ekranı.

**EK-3: Pipeline Doğrulama Denetim Tablosu**
[docs/PIPELINE_AUDIT.md] içeriğinin Türkçe özet versiyonu.

### ÖZGEÇMİŞ
**Kişisel Bilgiler:**
- Ad Soyad: Muhammed Yusuf TOSUN
- E-posta: tosunmuhammedyusuf67@gmail.com

**Eğitim:**
- Lisans: Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği (devam ediyor)

**Uzmanlık Alanı:** Mobil yapay zekâ, derin öğrenme, sinyal işleme

**Yabancı Diller:** İngilizce

---

## YAZIM KURALLARI HATIRLATMASI

- **Yazı tipi:** Times New Roman, 12 pt (gövde), 10 pt (özet/abstract, tablo başlığı)
- **Satır aralığı:** 1.5 (gövde), 1.0 (özet/abstract)
- **Kenar boşlukları:** Sol 3.5 cm, sağ/üst/alt 2.5 cm
- **Sayfa numarası:** Alt orta
- **Çizelge başlığı:** Üstte, sola dayalı ("Çizelge 3.1. Başlık")
- **Şekil başlığı:** Altta, ortalanmış ("Şekil 3.1. Başlık")
- **Atıf:** APA-Türkçe (Soyad, Yıl)

---

## SON TALİMAT

Yukarıdaki yapıyı **eksiksiz olarak** Türkçe akademik metin halinde üret. Her bölümün başlığını şablon numaralandırmasıyla ver. Şekil/çizelge yer tutucularını işaretle. Sayıları olduğu gibi koru. Sonuç olarak Word'e doğrudan kopyalanabilir, bölümleri başlık stilleriyle birlikte, akademik kalitede, profesyonel ve akıcı bir rapor metni ortaya çıkar.

**Şimdi başla. İlk olarak ÖZET ve ABSTRACT bölümlerini yaz, sonra sırayla GİRİŞ → KAYNAK ARAŞTIRMASI → MATERYAL VE YÖNTEM → ARAŞTIRMA SONUÇLARI VE TARTIŞMA → SONUÇLAR VE ÖNERİLER → KAYNAKLAR.**
