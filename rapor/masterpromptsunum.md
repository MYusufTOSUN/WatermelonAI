# Master Prompt — Karpuz Olgunluk Tespit Projesi SUNUM Üretimi

> **Kullanım:** Bu dosyanın **TÜM içeriğini** kopyalayıp ChatGPT / Claude / Gemini web arayüzüne yapıştır.
> AI sana, üniversite sunum şablonuna uygun, **az yazılı ama anlatılması kolay** bir slayt seti üretecek.
> Her slayt için hem **slayt üstündeki kısa madde metni** hem de altında **🎤 Konuşma Notu** (senin sözlü anlatacağın detay) verilecek.

---

## SİSTEME ROL VERİLMESİ

Sen, **Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği** Mühendislik Tasarımı (proje) sunumu hazırlayan bir lisans öğrencisinin **sunum tasarım asistanısın.**

Görevin: Aşağıdaki şablon iskeletine ve gerçek proje verisine sadık kalarak, **profesyonel, sade, kaliteli görünen** bir slayt seti içeriği üretmek. Çıktın doğrudan PowerPoint'e yazılabilir nitelikte olmalı.

**EN ÖNEMLİ KURAL — AZ YAZI:** Slayt, paragraf değil **konuşma destekçisidir.** Slaytta küçük ve az yazı olacak; detayı ben (öğrenci) sözlü anlatacağım. Bu yüzden her slayt için iki şey üreteceksin:
1. **SLAYT İÇERİĞİ** — başlık + en fazla 4–5 kısa madde (her madde **6–8 kelime**, tam cümle değil, anahtar ifade). Görsel/çizelge için yer tutucu.
2. **🎤 KONUŞMA NOTU** — o slaytı anlatırken söyleyeceğim **akıcı, detaylı** metin (3–6 cümle). Burada süreç, gerekçe ve sayılar açıklanır. Bunu slayta YAZMA; ayrı blok olarak ver.

---

## SLAYT TASARIM VE KALİTE KURALLARI

1. **Az metin:** Bir slaytta en fazla ~25 kelime görünür metin. Cümle değil, madde/anahtar ifade kullan.
2. **6×6 kuralı:** Satır başına ~6 kelime, slayt başına ~6 satırı geçme.
3. **Anahtar sayıları büyük göster:** %61.5, r=0.995, 19 karpuz, 146-D gibi çarpıcı rakamlar slaytta tek başına vurgulu durabilir.
4. **Görseller metnin önünde:** Her içerik slaytında en az bir görsel öneri ver — `[GÖRSEL: ...]` veya `[ÇİZELGE: ...]` yer tutucusu. (Akış diyagramı, ekran görüntüsü, konfüzyon matrisi, mimari şema, vuruş spektrumu.)
5. **Tutarlı dil:** Akademik ama sade Türkçe. Slaytta jargonu minimumda tut; jargonu Konuşma Notu'nda aç.
6. **Numaralandırma:** Şablon footer'ı korunur — her slaytta `BMU` etiketi, sağ üstte/altta `X/N` slayt numarası, alt bilgide "Muhammed Yusuf TOSUN". (N = toplam slayt sayısı.)
7. **Tip/renk önerisi:** Başlıklar koyu yeşil tonu (karpuz teması, ~#1B7F3A), gövde koyu gri/siyah, beyaz zemin. Times New Roman veya temiz bir sans-serif (Calibri). Tutarlı kullan.
8. **Uydurma yok:** Sayı/bulgu uydurma. Aşağıdaki gerçek verileri kullan. Olmayan veri için Konuşma Notu'nda "bu çalışmada ölçülmemiştir / gelecek çalışma" de.

---

## PROJE TEMEL BİLGİLERİ

- **Proje adı (TR):** Çok Modlu Sensör Verisi ve Vi-Liquid Yaklaşımı ile Karpuzda Olgunluk Tespiti için Mobil Yapay Zekâ Tabanlı Tahribatsız Test Sistemi
- **Öğrenci:** Muhammed Yusuf TOSUN
- **Bölüm:** Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği
- **Danışman:** [Öğrenci dolduracak — Unvanı Adı SOYADI]
- **Yıl / Yer:** 2026 / Konya

---

## GERÇEK PROJE VERİSİ (slaytlarda ve konuşma notlarında kullan — DEĞİŞTİRME)

**Ne yaptık (tek cümle):** Telefonla karpuzun fotoğrafını çek, vuruş sesini kaydet, titreşim cevabını ölç → cihaz üzerinde yapay zekâ "al / alma" desin. İnternet gerekmez.

**Çok modlu giriş (3 sensör):**
- Görsel (kamera): kabuk dokusu + tarla lekesi → 11 öznitelik
- Akustik (mikrofon): vuruş sesi MFCC + rezonans → 120 öznitelik
- Aktif haptik (LRA titreşim + IMU): Vi-Liquid metodolojisi → SRR + f₂ + Esneklik İndeksi
- (+ Hollow Heart 8 öznitelik) → **toplam 146 boyutlu birleşik vektör**

**Veri seti:** Qilin Watermelon Dataset — **19 karpuz, 4671 akustik kayıt**, Brix etiketli. Sınıflar: Olgunlaşmamış %31 / Olgun %58 / İçi Geçmiş %10. Görsel için birleşik 2019 görsel.

**Modeller:** KNN, Random Forest (n=300), Fusion DNN (78 KB FP16 TFLite), MobileNetV3-Small görsel (1.9 MB FP16).

**ANA BİLİMSEL SONUÇ:** Random Forest, 2-sınıf (Yenir/Yenmez) **Leave-One-Watermelon-Out (LOWO)** çapraz doğrulamasında **%61.5 doğruluk** (F1 Yenir 0.67). 3-sınıf LOWO: RFC %52.4, KNN %38.8.

**Mobil doğrulama:** Telefonun Dart DSP'si ile Python referansı arasında öznitelik korelasyonu **r=0.995**; akustik kanalda **12/12 aynı tahmin** (telefon = backend).

**Önemli mühendislik bulguları (sunumda "süreç" anlatımı için altın değerinde):**
- **Android mikrofon ön-işlemesi bulgusu:** Android varsayılan kaydı 50–250 Hz bandını (karpuz f₂ rezonansı) kırpıyordu → `AndroidAudioSource.unprocessed` ham mikrofona geçildi. Ders: "giriş sinyal yolu modelden kritik olabilir."
- **Tek-thread CPU çıkarımı:** FP16 model bazı uç değerlerde donanıma bağlı değişkenlik gösteriyordu → `threads=1` ile belirleyici (deterministik) çıkarım sağlandı.
- **FP16 vs FP32 kararı (varsayma, ölç):** FP32'ye geçme hipotezi ölçüldü; depodaki FP32 çökmüş (eğitim doğruluğu %57.6, hep "Olgun"), FP16 doğru (%99.9). FP32'ye GEÇİLMEDİ. Ders: yüksek hassasiyet her zaman iyi değildir.
- **Hollow Heart metadata bulgusu:** Eğitim kodu ile metadata uyuşmazlığı tespit edildi → "metadata değil, eğitim kodu ground truth'tur."
- **Vi-Liquid mobil dağıtımı:** LRA titreşim + IMU 100 Hz + SRR (cubic 1600 Hz) + EI = f₂²·m^(2/3) cihazda çalışıyor; literatürde mobil ilk.

**Saha testi (DÜRÜST çerçeve):** Saha testi yeni başlatıldı, eldeki veri **ön niteliktedir.** Su şişesi (karpuz değil) doğru reddedildi. Gerçek karpuzlar büyük ölçüde iyi/olgun → "iyi karpuzu doğru kabul etme" eğilimi gösterilebilir; **az örnekten genel saha doğruluğu yüzdesi İDDİA EDİLMEZ.** Dengeli ve kesimle doğrulanmış geniş saha kümesi gelecek çalışmadır.

**Performans:** Analiz süresi optimizasyonla 1–2 dk → ~20–30 sn.

**Çerçeve referansı:** Koç & Akbalık (2025); Yamamoto (1980) akustik; Vi-Liquid (IPSN 2020); MobileNetV3 (Howard 2019).

---

## ŞABLON İSKELETİ VE SLAYT PLANI

Şablon 9 bölümlük bir iskelet sunar (Kapak → İçindekiler → Giriş → Kaynak Araştırması → Materyal ve Yöntem → Araştırma Sonuçları ve Tartışma → Sonuçlar ve Öneriler → Kaynaklar → Teşekkürler). İçerik bölümleri gerektiğinde **birden fazla slayda** bölünebilir. Aşağıdaki **önerilen 15 slaytlık plan** kullanılacak; footer numarası buna göre `X/15` olur (öğrenci isterse sıkıştırabilir).

> Her slayt için **(A) SLAYT İÇERİĞİ** ve **(B) 🎤 KONUŞMA NOTU** ayrı ayrı üret.

### Slayt 1 — KAPAK
- (A) T.C. SELÇUK ÜNİVERSİTESİ / TEKNOLOJİ FAKÜLTESİ / BİLGİSAYAR MÜHENDİSLİĞİ · Proje adı · Muhammed Yusuf TOSUN · Danışman [boş] · Konya 2026. `[GÖRSEL: karpuz + telefon teması logo/görsel]`
- (B) Kısa selam + "Karpuzun olgun olup olmadığını telefonla, kesmeden anlayan bir yapay zekâ sistemi geliştirdim" tek cümlelik açılış.

### Slayt 2 — İÇİNDEKİLER
- (A) Giriş · Kaynak Araştırması · Materyal ve Yöntem · Araştırma Sonuçları ve Tartışma · Sonuçlar ve Öneriler · Kaynaklar
- (B) "Sunumumda sırasıyla şu başlıkları ele alacağım…" 1 cümle.

### Slayt 3 — GİRİŞ: Problem ve Motivasyon
- (A) 3–4 madde: Karpuzda olgunluk tahribatsız zor · Geleneksel yöntemler (vuruş/leke) standart değil · Lab Brix metresi pahalı/taşınmaz · Akıllı telefon sensörleri fırsat. `[GÖRSEL: markette karpuz seçen kişi]`
- (B) Üretici/tüketici probleminden başlayıp "herkesin cebinde kamera, mikrofon, ivmeölçer var; bunu kullanabilir miyiz?" sorusuna bağla.

### Slayt 4 — GİRİŞ: Amaç ve Katkılar
- (A) Amaç: Cihaz-üstü çok modlu yapay zekâ · Katkılar: Akustik+DNN geç füzyon · Vi-Liquid mobil dağıtım (ilk) · Foto-tabanlı kütle + EI · Halk-dostu arayüz. `[GÖRSEL: 3 sensör ikonu]`
- (B) "Üç farklı duyuyu (görme, işitme, dokunma) birleştirdim; tüm hesap telefonda, internetsiz. Özgün katkım Vi-Liquid titreşim yöntemini ilk kez mobile taşımak."

### Slayt 5 — KAYNAK ARAŞTIRMASI
- (A) Yamamoto (1980) akustik rezonans · MFCC (Davis & Mermelstein 1980) · MobileNetV3 (Howard 2019) · Vi-Liquid / SRR (IPSN 2020) · Geç füzyon · Koç & Akbalık (2025). `[GÖRSEL: kaynak zaman çizelgesi]`
- (B) Her referansı tek cümleyle bağla: hangisinden neyi aldım. "Yamamoto'dan akustik fikri, Vi-Liquid'den titreşim, MobileNetV3'ten mobil görüntü omurgası."

### Slayt 6 — MATERYAL VE YÖNTEM: Veri Seti
- (A) Qilin: 19 karpuz · 4671 akustik kayıt · Brix etiketli · Sınıf dağılımı %31/%58/%10 · 2019 görsel. `[ÇİZELGE: sınıf dağılımı]`
- (B) Brix → sınıf eşlemesini anlat (<10 ham, 10–11.5 olgun, >11.5 geçmiş). "19 karpuz az; bu sınırı dürüstçe rapora yazdım."

### Slayt 7 — MATERYAL VE YÖNTEM: Öznitelikler ve Mimari
- (A) 146-D birleşik vektör: 120 akustik + 11 görsel + 7 haptik + 8 HH · Modeller: KNN / RFC / Fusion DNN / MobileNetV3. `[GÖRSEL: mimari blok şema]`
- (B) Öznitelik mühendisliğini sade anlat: MFCC = sesin parmak izi, görsel = leke+doku, haptik = titreşim cevabı. DNN dört girişi birleştirip 3 sınıf olasılığı üretir.

### Slayt 8 — MATERYAL VE YÖNTEM: Mobil Pipeline + Vi-Liquid
- (A) 3 adım sihirbazı (Foto→Vuruş→Titreşim) · Dart DSP (FFT/MFCC) · Vi-Liquid: LRA + IMU 100 Hz + SRR 1600 Hz → f₂ → EI=f₂²·m^(2/3) · Geç füzyon + Hollow Heart tetiği. `[GÖRSEL: mimari akış diyagramı]`
- (B) Kullanıcı akışını canlı anlat. Vi-Liquid'i "telefon titrer, ivmeölçer cevabı okur, düşük frekans sinyalini yükseğe çözümleriz" diye aç. Geç füzyon = ML + fizik kuralı birlikte.

### Slayt 9 — ARAŞTIRMA SONUÇLARI: Ana Doğruluk (LOWO)
- (A) **%61.5** büyük punto · RFC 2-sınıf LOWO · F1 0.67 · 3-sınıf RFC %52.4. `[ÇİZELGE: LOWO tablosu]` `[GÖRSEL: konfüzyon matrisi]`
- (B) LOWO'yu neden seçtiğimi vurgula: aynı karpuzun verisi hem eğitim hem testte olmasın diye (subject leakage). "Bu yüzden %61.5 dürüst genelleme sayısı; random split %100 çıkıyor ama o yanıltıcı."

### Slayt 10 — ARAŞTIRMA SONUÇLARI: Mobil Doğrulama + Mühendislik Bulguları
- (A) Telefon = backend: **r=0.995** · akustik 12/12 aynı tahmin · Android mikrofon kırpma düzeltmesi · tek-thread CPU · FP16 vs FP32 kararı. `[ÇİZELGE: parite tablosu]`
- (B) Süreç hikâyesini anlat (en etkileyici kısım): "Saha testinde iyi karpuza 'ham' dedi; matematik kusursuzdu, suçlu Android'in mikrofon ön-işlemesiydi. Ham mikrofona geçtim. Ayrıca FP32'ye geçecektim, ölçtüm — bozuk çıktı, geçmedim. Her kararı varsayımla değil ölçümle aldım."

### Slayt 11 — ARAŞTIRMA SONUÇLARI: Vi-Liquid Cihaz Testi + Saha Ön Bulguları
- (A) Su şişesi → doğru "alma" (fizik kuralı çözdü) · Gerçek karpuz ön testleri sürüyor · Saha verisi ön nitelikte. `[GÖRSEL: sonuç ekranı + PDF rapor]`
- (B) DÜRÜST çerçeve: "Saha testine yeni başladım. Su şişesini doğru reddetti. Karpuzları kesip doğruluyorum; örneklem henüz küçük ve çoğu iyi karpuz, bu yüzden genel saha doğruluğu yüzdesi iddia etmiyorum — dengeli ve geniş saha testi gelecek çalışmam."

### Slayt 12 — ARAŞTIRMA SONUÇLARI: Performans ve Arayüz
- (A) Analiz 1–2 dk → ~20–30 sn · Halk-dostu UI · PDF ölçüm raporu + paylaşım. `[GÖRSEL: önce/sonra ekran]`
- (B) Optimizasyon adımlarını (agresif ses kırpma, mel 128→64, paylaşılan FFT) kısaca anlat. "Jargonu arayüzden temizledim; herkes 'al/alma' görsün diye."

### Slayt 13 — SONUÇLAR VE ÖNERİLER
- (A) Sonuç: Cihaz-üstü çok modlu sistem çalışıyor (%61.5 LOWO, r=0.995) · ML+fizik hibriti çelişkiyi çözüyor · Öneriler: dengeli geniş saha verisi · 100+ Brix etiketli karpuz · tam NUFFT-OMP SRR · diğer meyveler. `[GÖRSEL: özet ikonları]`
- (B) Başardıklarını + dürüst sınırları + gelecek yol haritasını topla. "En büyük ihtiyaç daha çok ve dengeli gerçek karpuz verisi."

### Slayt 14 — KAYNAKLAR
- (A) APA-Türkçe listesi (Koç & Akbalık 2025, Yamamoto 1980, Vi-Liquid 2020, Howard 2019, Davis & Mermelstein 1980, librosa 2015, Breiman 2001 …).
- (B) Sözlü anlatım yok; "ana referanslarım şunlar" tek cümle.

### Slayt 15 — TEŞEKKÜRLER
- (A) "Teşekkürler" · Muhammed Yusuf TOSUN · Soru-cevaba hazır. T.C. Selçuk Üniversitesi başlığı.
- (B) Kısa kapanış + soruları davet eden tek cümle.

---

## ÇIKTI BİÇİMİ (AI bu formatta üretsin)

Her slaytı şu blok yapısında ver:

```
────────────────────────────────
SLAYT X / 15 — [BÖLÜM] : [Slayt Başlığı]
Footer: BMU · X/15 · Muhammed Yusuf TOSUN

📑 SLAYT İÇERİĞİ
• madde 1 (kısa)
• madde 2 (kısa)
• madde 3 (kısa)
[GÖRSEL/ÇİZELGE: ...]

🎤 KONUŞMA NOTU
(öğrencinin sözlü anlatacağı 3–6 cümlelik akıcı metin)
────────────────────────────────
```

---

## SON TALİMAT

Yukarıdaki 15 slaytı **sırayla, eksiksiz** üret. Kurallar:
- Slaytlarda **az ve büyük yazı**, paragraf yok, madde/anahtar ifade.
- Her slayt için ayrı **🎤 Konuşma Notu** ver (detay burada, slaytta değil).
- Gerçek sayıları koru (%61.5, r=0.995, 19 karpuz/4671, 146-D, ~20–30 sn).
- Saha testini **dürüst/ön nitelikli** anlat; uydurma doğruluk yüzdesi verme.
- Görsel yer tutucularını işaretle.
- Akademik, sade, tutarlı, profesyonel ton.

**Şimdi başla. Slayt 1 (Kapak) ile başla, sırayla 15. slayda (Teşekkürler) kadar üret.**
