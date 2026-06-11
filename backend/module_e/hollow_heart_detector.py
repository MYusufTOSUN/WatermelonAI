"""
MODULE_E: Hollow Heart (İçi Geçmiş) Dedektörü

Karpuzlarda hollow heart (iç boşluk) durumunu çoklu akustik
gösterge analizi ile tespit eder.

Fiziksel Arka Plan:
  - Hollow heart karpuzlarda iç boşluk rezonans davranışını değiştirir:
    1) Ana rezonans tepesi bölünür (dual-peak / split resonance)
    2) Sönüm hızlanır ve düzensizleşir (kavite kayıpları)
    3) Spektral enerji daha geniş banda yayılır
    4) Harmonik yapı bozulur (HNR düşer)
    5) Cepstral periyodiklik zayıflar

Tespit Stratejisi:
  5 bağımsız gösterge → ağırlıklı toplam → hollow heart skoru

  HH_score = w1*dual_peak + w2*damping + w3*spectral + w4*cepstral + w5*hnr

  HH_score > 0.55  →  İçi Geçmiş (Hollow Heart)
  HH_score > 0.75  →  Yüksek Güvenli Hollow Heart

Referans:
  - Yamamoto et al. (2008): İç boşluk akustik tespiti
  - Koç & Akbalık (2025): Karpuz olgunluk sınıflandırma
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from typing import Dict, Tuple, Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    AUDIO_SAMPLE_RATE,
    FREQ_BAND_LOW,
    FREQ_BAND_HIGH,
    F2_RIPE_THRESHOLD,
    # Hollow Heart parametreleri
    HH_DUAL_PEAK_MIN_DISTANCE_HZ,
    HH_DUAL_PEAK_MAX_DISTANCE_HZ,
    HH_DUAL_PEAK_PROMINENCE_DB,
    HH_DUAL_PEAK_VALLEY_DEPTH_DB,
    HH_DAMPING_RATIO_THRESHOLD,
    HH_DECAY_RATE_FAST_THRESHOLD,
    HH_DECAY_IRREGULARITY_THRESHOLD,
    HH_SPECTRAL_SPREAD_THRESHOLD,
    HH_SPECTRAL_ENTROPY_HIGH,
    HH_SPECTRAL_FLATNESS_HIGH,
    HH_CEPSTRAL_PEAK_PROMINENCE,
    HH_QUEFRENCY_RANGE,
    HH_HNR_LOW_THRESHOLD,
    HH_WEIGHT_DUAL_PEAK,
    HH_WEIGHT_DAMPING,
    HH_WEIGHT_SPECTRAL,
    HH_WEIGHT_CEPSTRAL,
    HH_WEIGHT_HNR,
    HH_DETECTION_THRESHOLD,
    HH_HIGH_CONFIDENCE_THRESHOLD,
)


class HollowHeartDetector:
    """
    Karpuz hollow heart (iç boşluk) dedektörü.

    Çoklu akustik gösterge analizi ile yüksek doğrulukta
    hollow heart tespiti yapar.

    Kullanım:
        detector = HollowHeartDetector()
        result = detector.detect(audio_signal, sample_rate)
        if result["is_hollow"]:
            print(f"Hollow Heart! Güven: {result['confidence']:.2f}")
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        freq_range: Tuple[float, float] = (FREQ_BAND_LOW, FREQ_BAND_HIGH)
    ):
        self.sample_rate = sample_rate
        self.freq_range = freq_range

    # =================================================================
    # ANA TESPİT METODU
    # =================================================================

    def detect(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        f2: Optional[float] = None,
        f2_magnitude_db: Optional[float] = None,
        ei: Optional[float] = None,
        verbose: bool = True
    ) -> Dict[str, object]:
        """
        Hollow heart tespiti yapar.

        5 bağımsız göstergeyi analiz eder ve ağırlıklı skora göre
        hollow heart kararı verir.

        Args:
            audio: Ses sinyali (vuruş kaydı)
            sr: Örnekleme hızı (None ise varsayılan)
            f2: Önceden hesaplanmış dominant frekans (Hz) - opsiyonel
            f2_magnitude_db: Önceden hesaplanmış f2 genliği (dB) - opsiyonel
            ei: Önceden hesaplanmış Elasticity Index - opsiyonel
            verbose: Detaylı çıktı yazdır

        Returns:
            Hollow heart tespit sonuçları:
              - is_hollow: bool - Hollow heart var mı?
              - hh_score: float - Hollow heart skoru (0-1)
              - confidence: float - Güven skoru
              - indicators: dict - Her göstergenin detaylı sonuçları
              - recommendation: str - Öneri metni
        """
        if sr is None:
            sr = self.sample_rate

        if verbose:
            print("\n" + "=" * 60)
            print("  HOLLOW HEART DEDEKTORU")
            print(f"  Sinyal: {len(audio)} ornek @ {sr}Hz")
            print("=" * 60)

        # --- Gösterge 1: Dual-peak (Çift Tepe) Rezonans ---
        dual_peak_result = self.analyze_dual_peak_resonance(audio, sr)
        dual_peak_score = dual_peak_result["score"]

        # --- Gösterge 2: Sönüm Analizi ---
        damping_result = self.analyze_damping_characteristics(audio, sr)
        damping_score = damping_result["score"]

        # --- Gösterge 3: Spektral Yayılma ---
        spectral_result = self.analyze_spectral_spread(audio, sr)
        spectral_score = spectral_result["score"]

        # --- Gösterge 4: Cepstral Analiz ---
        cepstral_result = self.analyze_cepstral_features(audio, sr)
        cepstral_score = cepstral_result["score"]

        # --- Gösterge 5: Harmonik-Gürültü Oranı (HNR) ---
        hnr_result = self.analyze_harmonic_noise_ratio(audio, sr)
        hnr_score = hnr_result["score"]

        # --- Ağırlıklı Toplam ---
        hh_score = (
            HH_WEIGHT_DUAL_PEAK * dual_peak_score +
            HH_WEIGHT_DAMPING * damping_score +
            HH_WEIGHT_SPECTRAL * spectral_score +
            HH_WEIGHT_CEPSTRAL * cepstral_score +
            HH_WEIGHT_HNR * hnr_score
        )

        # --- EI ve f2 ile fiziksel doğrulama (varsa) ---
        physical_boost = 0.0
        if f2 is not None and f2 < F2_RIPE_THRESHOLD * 0.6:
            # Çok düşük f2: hollow heart göstergesi
            physical_boost += 0.05
        if ei is not None and ei < 1e4:
            # Çok düşük EI: aşırı yumuşak → hollow heart olabilir
            physical_boost += 0.03

        hh_score = min(1.0, hh_score + physical_boost)

        # --- Karar ---
        is_hollow = hh_score >= HH_DETECTION_THRESHOLD
        is_high_confidence = hh_score >= HH_HIGH_CONFIDENCE_THRESHOLD

        # Güven skoru: eşikten uzaklığa göre
        if is_hollow:
            confidence = min(1.0, 0.6 + (hh_score - HH_DETECTION_THRESHOLD) * 2.0)
        else:
            confidence = min(1.0, 0.6 + (HH_DETECTION_THRESHOLD - hh_score) * 2.0)

        # Aktif gösterge sayısı
        active_indicators = sum([
            dual_peak_score > 0.5,
            damping_score > 0.5,
            spectral_score > 0.5,
            cepstral_score > 0.5,
            hnr_score > 0.5
        ])

        # Çok-göstergeli teyit: 3+ gösterge aktifse güveni artır
        if is_hollow and active_indicators >= 3:
            confidence = min(1.0, confidence + 0.1)
        elif is_hollow and active_indicators <= 1:
            confidence = max(0.5, confidence - 0.15)

        # Öneri metni
        if is_high_confidence:
            recommendation = (
                "YÜKSEK GÜVENLE İÇİ GEÇMİŞ: Bu karpuzda belirgin "
                "hollow heart (iç boşluk) tespit edildi. Tüketim için "
                "uygun değildir."
            )
        elif is_hollow:
            recommendation = (
                "OLASI İÇİ GEÇMİŞ: Bu karpuzda hollow heart belirtileri "
                "var. Tüketim kalitesi düşük olabilir."
            )
        else:
            recommendation = (
                "HOLLOW HEART YOK: Bu karpuzda iç boşluk belirtisi "
                "tespit edilmedi."
            )

        if verbose:
            print(f"\n  [Gosterge 1] Dual-Peak Rezonans : {dual_peak_score:.3f}")
            print(f"  [Gosterge 2] Sonum Analizi      : {damping_score:.3f}")
            print(f"  [Gosterge 3] Spektral Yayilma   : {spectral_score:.3f}")
            print(f"  [Gosterge 4] Cepstral Analiz    : {cepstral_score:.3f}")
            print(f"  [Gosterge 5] HNR Analizi        : {hnr_score:.3f}")
            if physical_boost > 0:
                print(f"  [Fiziksel]   EI/f2 Boost        : +{physical_boost:.3f}")
            print(f"\n  TOPLAM HH SKORU: {hh_score:.3f} (esik: {HH_DETECTION_THRESHOLD})")
            print(f"  KARAR: {'ICI GECMIS [!]' if is_hollow else 'Normal [OK]'}")
            print(f"  GUVEN: {confidence:.2f} ({active_indicators}/5 gosterge aktif)")
            print(f"{'=' * 60}")

        return {
            "is_hollow": is_hollow,
            "is_high_confidence": is_high_confidence,
            "hh_score": float(hh_score),
            "confidence": float(confidence),
            "active_indicators": active_indicators,
            "physical_boost": float(physical_boost),
            "indicators": {
                "dual_peak": dual_peak_result,
                "damping": damping_result,
                "spectral_spread": spectral_result,
                "cepstral": cepstral_result,
                "hnr": hnr_result,
            },
            "recommendation": recommendation,
        }

    # =================================================================
    # GÖSTERGe 1: DUAL-PEAK REZONANS ANALİZİ
    # =================================================================

    def analyze_dual_peak_resonance(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, object]:
        """
        Çift tepe (dual-peak / split) rezonans analizi.

        Hollow heart karpuzlarda iç kavite, ana rezonans tepesinin
        iki ayrı tepeye bölünmesine neden olur. Bu bölünme
        boşluk boyutu ile orantılıdır.

        Fiziksel Model:
          Normal karpuz:  Tek baskın tepe (f2)
          Hollow heart:   İki tepe (f2a, f2b) — aralarında belirgin vadi

        Ölçülen:
          - Tepe sayısı ve pozisyonları
          - İki tepe arası frekans farkı (Hz)
          - Vadi derinliği (dB)
          - Tepe oranı (asimetri)

        Returns:
            score (0-1): 1.0 = belirgin dual-peak → hollow heart göstergesi
        """
        N = len(audio)
        if N < 64:
            return {"score": 0.0, "peaks_found": 0, "detail": "Sinyal çok kısa"}

        # FFT
        windowed = audio * np.hanning(N)
        fft_vals = rfft(windowed)
        freqs = rfftfreq(N, d=1.0 / sr)
        magnitudes = np.abs(fft_vals) * 2.0 / N
        mag_db = 20.0 * np.log10(magnitudes + 1e-10)

        # İlgi bandına filtrele
        mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        band_freqs = freqs[mask]
        band_db = mag_db[mask]

        if len(band_db) < 10:
            return {"score": 0.0, "peaks_found": 0, "detail": "Bant çok dar"}

        # Tepe tespiti (scipy find_peaks)
        # Minimum belirginlik: HH_DUAL_PEAK_PROMINENCE_DB
        freq_resolution = sr / N
        min_distance_samples = max(1, int(HH_DUAL_PEAK_MIN_DISTANCE_HZ / freq_resolution))

        peaks, properties = scipy_signal.find_peaks(
            band_db,
            prominence=HH_DUAL_PEAK_PROMINENCE_DB,
            distance=min_distance_samples,
            height=-60  # Minimum genlik eşiği (dB)
        )

        if len(peaks) < 2:
            return {
                "score": 0.0,
                "peaks_found": len(peaks),
                "detail": "Çift tepe bulunamadı (tek tepe veya tepe yok)"
            }

        # En güçlü 2 tepeyi al
        peak_amplitudes = band_db[peaks]
        top2_idx = np.argsort(peak_amplitudes)[-2:]
        peak1_idx, peak2_idx = sorted(top2_idx)

        f_peak1 = float(band_freqs[peaks[peak1_idx]])
        f_peak2 = float(band_freqs[peaks[peak2_idx]])
        amp_peak1 = float(band_db[peaks[peak1_idx]])
        amp_peak2 = float(band_db[peaks[peak2_idx]])
        freq_diff = abs(f_peak2 - f_peak1)

        # Frekans farkı hollow heart aralığında mı?
        if not (HH_DUAL_PEAK_MIN_DISTANCE_HZ <= freq_diff <= HH_DUAL_PEAK_MAX_DISTANCE_HZ):
            return {
                "score": 0.1,
                "peaks_found": len(peaks),
                "freq_diff_hz": freq_diff,
                "detail": f"Frekans farkı aralık dışı: {freq_diff:.1f}Hz"
            }

        # İki tepe arasındaki vadi derinliği
        idx_start = peaks[peak1_idx]
        idx_end = peaks[peak2_idx]
        if idx_start > idx_end:
            idx_start, idx_end = idx_end, idx_start

        valley_region = band_db[idx_start:idx_end + 1]
        valley_min = float(np.min(valley_region))
        peak_avg = (amp_peak1 + amp_peak2) / 2.0
        valley_depth = peak_avg - valley_min

        # Tepe asimetrisi (genlik oranı)
        amp_ratio = min(amp_peak1, amp_peak2) / (max(amp_peak1, amp_peak2) + 1e-10)

        # Skor hesapla
        score = 0.0

        # Vadi derinliği skoru
        if valley_depth >= HH_DUAL_PEAK_VALLEY_DEPTH_DB:
            depth_score = min(1.0, valley_depth / (HH_DUAL_PEAK_VALLEY_DEPTH_DB * 3))
            score += 0.4 * depth_score

        # Frekans farkı skoru (orta mesafe en belirgin)
        optimal_diff = (HH_DUAL_PEAK_MIN_DISTANCE_HZ + HH_DUAL_PEAK_MAX_DISTANCE_HZ) / 2
        diff_score = 1.0 - abs(freq_diff - optimal_diff) / (optimal_diff + 1e-10)
        diff_score = max(0.0, diff_score)
        score += 0.3 * diff_score

        # Asimetri skoru (simetrik tepeler daha güçlü gösterge)
        score += 0.3 * amp_ratio

        score = min(1.0, score)

        return {
            "score": float(score),
            "peaks_found": len(peaks),
            "peak1_hz": f_peak1,
            "peak2_hz": f_peak2,
            "peak1_db": amp_peak1,
            "peak2_db": amp_peak2,
            "freq_diff_hz": freq_diff,
            "valley_depth_db": valley_depth,
            "amplitude_ratio": float(amp_ratio),
            "detail": f"Dual-peak: {f_peak1:.1f}Hz & {f_peak2:.1f}Hz, Δf={freq_diff:.1f}Hz, vadi={valley_depth:.1f}dB"
        }

    # =================================================================
    # GÖSTERGE 2: SÖNÜM (DAMPING) ANALİZİ
    # =================================================================

    def analyze_damping_characteristics(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, object]:
        """
        Sönüm (damping) karakteristik analizi.

        Hollow heart karpuzlarda iç kavite ek enerji kaybına
        neden olur:
          - Daha hızlı sönüm (decay rate artışı)
          - Düzensiz sönüm zarfı (envelope irregularity)
          - Yüksek sönüm oranı (damping ratio)

        Zarf modeli: A(t) = A₀ · exp(-ζ·ω₀·t)
          Normal:  ζ ≈ 0.03–0.08, düzgün zarf
          Hollow:  ζ > 0.12, düzensiz zarf

        Returns:
            score (0-1): 1.0 = hızlı/düzensiz sönüm → hollow heart göstergesi
        """
        N = len(audio)
        if N < 64:
            return {"score": 0.0, "detail": "Sinyal çok kısa"}

        abs_audio = np.abs(audio)

        # Hilbert zarfı (envelope)
        analytic_signal = scipy_signal.hilbert(audio)
        envelope = np.abs(analytic_signal)

        # Zarfı yumuşat (hızlı dalgalanmaları kaldır)
        smooth_window = min(N // 8, max(5, int(sr * 0.005)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        if smooth_window >= 3 and smooth_window <= N:
            envelope_smooth = scipy_signal.savgol_filter(
                envelope, smooth_window, min(2, smooth_window - 1)
            )
        else:
            envelope_smooth = envelope.copy()

        # Tepe noktasını bul
        peak_idx = np.argmax(envelope_smooth)
        peak_amp = envelope_smooth[peak_idx]

        if peak_amp < 1e-10 or peak_idx >= N - 10:
            return {"score": 0.0, "detail": "Tepe bulunamadı"}

        # Sönüm bölgesi: tepe sonrası
        decay_region = envelope_smooth[peak_idx:]
        decay_time = np.arange(len(decay_region)) / sr

        if len(decay_region) < 10:
            return {"score": 0.0, "detail": "Sönüm bölgesi çok kısa"}

        # --- 1) Sönüm oranı (ζ) tahmini ---
        # Üstel fit: A(t) = A₀ · exp(-α·t), ζ ≈ α / (2π·f2)
        # Log-lineer regresyon
        log_decay = np.log(np.maximum(decay_region, 1e-10))
        t = decay_time

        # En küçük kareler ile α tahmini
        valid_mask = decay_region > peak_amp * 0.01  # %1 üstü geçerli
        if np.sum(valid_mask) < 5:
            valid_mask = np.ones(len(decay_region), dtype=bool)

        t_valid = t[valid_mask]
        log_valid = log_decay[valid_mask]

        if len(t_valid) > 2:
            coeffs = np.polyfit(t_valid, log_valid, 1)
            alpha = -coeffs[0]  # Sönüm katsayısı (pozitif)
        else:
            alpha = 0.0

        # Sönüm oranı tahmini (f2 ≈ 120Hz varsayımı)
        f2_est = 120.0  # Ortalama karpuz rezonansı
        damping_ratio = alpha / (2 * np.pi * f2_est) if f2_est > 0 else 0.0

        # --- 2) Sönüm hızı (Decay Rate) ---
        # %10 seviyesine düşme süresi
        threshold_10 = 0.1 * peak_amp
        below_10 = np.where(decay_region <= threshold_10)[0]
        if len(below_10) > 0:
            decay_samples = below_10[0]
            decay_rate = sr / (decay_samples + 1e-10)
        else:
            decay_rate = 0.0

        # --- 3) Zarf düzensizliği (Envelope Irregularity) ---
        # Zarfın monoton azalıştan sapma derecesi
        if len(decay_region) > 5:
            # Birinci fark (türev)
            diff = np.diff(decay_region)
            # Pozitif türev oranı (artış olmaması beklenir)
            positive_ratio = np.sum(diff > 0) / (len(diff) + 1e-10)
            # Zarf roughness: normalize standart sapma
            residual = decay_region - np.linspace(
                decay_region[0], decay_region[-1], len(decay_region)
            )
            roughness = np.std(residual) / (peak_amp + 1e-10)
            irregularity = 0.6 * positive_ratio + 0.4 * min(1.0, roughness * 5)
        else:
            irregularity = 0.0

        # --- Skor Hesapla ---
        score = 0.0

        # Sönüm oranı skoru
        if damping_ratio > HH_DAMPING_RATIO_THRESHOLD:
            zeta_score = min(1.0, (damping_ratio - HH_DAMPING_RATIO_THRESHOLD) /
                            HH_DAMPING_RATIO_THRESHOLD)
            score += 0.35 * zeta_score

        # Hızlı sönüm skoru
        if decay_rate > HH_DECAY_RATE_FAST_THRESHOLD:
            rate_score = min(1.0, (decay_rate - HH_DECAY_RATE_FAST_THRESHOLD) /
                           HH_DECAY_RATE_FAST_THRESHOLD)
            score += 0.30 * rate_score

        # Düzensizlik skoru
        if irregularity > HH_DECAY_IRREGULARITY_THRESHOLD:
            irreg_score = min(1.0, (irregularity - HH_DECAY_IRREGULARITY_THRESHOLD) /
                             (1.0 - HH_DECAY_IRREGULARITY_THRESHOLD + 1e-10))
            score += 0.35 * irreg_score

        score = min(1.0, score)

        return {
            "score": float(score),
            "damping_ratio": float(damping_ratio),
            "decay_rate": float(decay_rate),
            "alpha": float(alpha),
            "irregularity": float(irregularity),
            "peak_amplitude": float(peak_amp),
            "detail": (
                f"ζ={damping_ratio:.4f} (eşik:{HH_DAMPING_RATIO_THRESHOLD}), "
                f"decay_rate={decay_rate:.1f}/s, "
                f"düzensizlik={irregularity:.3f}"
            )
        }

    # =================================================================
    # GÖSTERGE 3: SPEKTRAL YAYILMA ANALİZİ
    # =================================================================

    def analyze_spectral_spread(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, object]:
        """
        Spektral yayılma (spread) ve entropi analizi.

        Hollow heart karpuzlarda:
          - Rezonans enerjisi daha geniş bir frekans bandına yayılır
          - Spektral entropi artar (düzensiz dağılım)
          - Spektral düzlük (flatness) artar (gürültü benzeri)

        Ölçülen:
          - Normalized spectral spread: σ_f / f_c
          - Shannon spectral entropy (normalize)
          - Wiener spectral flatness

        Returns:
            score (0-1): 1.0 = çok yayılmış spektrum → hollow heart göstergesi
        """
        N = len(audio)
        if N < 64:
            return {"score": 0.0, "detail": "Sinyal çok kısa"}

        # FFT
        windowed = audio * np.hanning(N)
        fft_vals = rfft(windowed)
        freqs = rfftfreq(N, d=1.0 / sr)
        magnitudes = np.abs(fft_vals)

        # İlgi bandı
        mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        band_freqs = freqs[mask]
        band_mags = magnitudes[mask]

        if len(band_mags) < 5:
            return {"score": 0.0, "detail": "Bant çok dar"}

        # Power spectrum
        psd = band_mags ** 2
        psd_norm = psd / (np.sum(psd) + 1e-10)

        # --- 1) Spektral Centroid ve Spread ---
        spectral_centroid = np.sum(band_freqs * psd_norm)
        spectral_spread = np.sqrt(
            np.sum(((band_freqs - spectral_centroid) ** 2) * psd_norm)
        )
        # Normalize: spread / centroid
        normalized_spread = spectral_spread / (spectral_centroid + 1e-10)

        # --- 2) Shannon Spektral Entropi ---
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        max_entropy = np.log2(len(psd_norm))
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # --- 3) Wiener Spektral Flatness ---
        # GM / AM oranı: 1.0 = tamamen düz (beyaz gürültü)
        log_mean = np.mean(np.log(band_mags + 1e-10))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = np.mean(band_mags)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # --- 4) Enerji yoğunlaşma oranı (Concentration Ratio) ---
        # En güçlü %10'luk frekans bölgesindeki enerji oranı
        sorted_psd = np.sort(psd)[::-1]
        top_10_pct = max(1, int(0.10 * len(sorted_psd)))
        concentration = np.sum(sorted_psd[:top_10_pct]) / (np.sum(psd) + 1e-10)
        # Düşük yoğunlaşma → dağılmış enerji → hollow heart
        dispersion = 1.0 - concentration

        # --- Skor Hesapla ---
        score = 0.0

        # Yayılma skoru
        if normalized_spread > HH_SPECTRAL_SPREAD_THRESHOLD:
            spread_score = min(1.0, (normalized_spread - HH_SPECTRAL_SPREAD_THRESHOLD) /
                              (1.0 - HH_SPECTRAL_SPREAD_THRESHOLD + 1e-10))
            score += 0.30 * spread_score

        # Entropi skoru
        if normalized_entropy > HH_SPECTRAL_ENTROPY_HIGH:
            entropy_score = min(1.0, (normalized_entropy - HH_SPECTRAL_ENTROPY_HIGH) /
                               (1.0 - HH_SPECTRAL_ENTROPY_HIGH + 1e-10))
            score += 0.30 * entropy_score

        # Düzlük skoru
        if spectral_flatness > HH_SPECTRAL_FLATNESS_HIGH:
            flat_score = min(1.0, (spectral_flatness - HH_SPECTRAL_FLATNESS_HIGH) /
                            (1.0 - HH_SPECTRAL_FLATNESS_HIGH + 1e-10))
            score += 0.20 * flat_score

        # Dağılma skoru
        score += 0.20 * min(1.0, dispersion * 1.5)

        score = min(1.0, score)

        return {
            "score": float(score),
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_spread_hz": float(spectral_spread),
            "normalized_spread": float(normalized_spread),
            "spectral_entropy": float(normalized_entropy),
            "spectral_flatness": float(spectral_flatness),
            "energy_concentration": float(concentration),
            "energy_dispersion": float(dispersion),
            "detail": (
                f"spread={normalized_spread:.3f} (eşik:{HH_SPECTRAL_SPREAD_THRESHOLD}), "
                f"entropy={normalized_entropy:.3f}, "
                f"flatness={spectral_flatness:.3f}"
            )
        }

    # =================================================================
    # GÖSTERGE 4: CEPSTRAL ANALİZ
    # =================================================================

    def analyze_cepstral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, object]:
        """
        Cepstral (quefrency domain) analiz.

        Cepstrum: log-power spektrumunun ters FFT'si.
        Quefrency alanında periyodik yapıları tespit eder.

        Hollow heart karpuzlarda:
          - Rezonans periyodikliği zayıflar (cepstral tepe düşer)
          - Birden fazla zayıf cepstral tepe oluşur
          - Rahmeniya (cepstral) düzlüğü artar

        Ölçülen:
          - Quefrency tepe belirginliği (prominence)
          - Cepstral düzlük (flatness)
          - Rahmeniya varyansı

        Returns:
            score (0-1): 1.0 = bozuk cepstral yapı → hollow heart göstergesi
        """
        N = len(audio)
        if N < 128:
            return {"score": 0.0, "detail": "Sinyal çok kısa (cepstrum için)"}

        # Power cepstrum: IFFT(log(|FFT(x)|^2))
        fft_vals = rfft(audio * np.hanning(N))
        power_spectrum = np.abs(fft_vals) ** 2
        log_spectrum = np.log(power_spectrum + 1e-10)

        # Real cepstrum (kullan: IFFT yerine IRFFT)
        cepstrum = np.fft.irfft(log_spectrum)
        cepstrum_abs = np.abs(cepstrum)

        # Quefrency ekseni (saniye)
        quefrency = np.arange(len(cepstrum)) / sr

        # İlgi quefrency aralığı (karpuz rezonansına uygun)
        q_low, q_high = HH_QUEFRENCY_RANGE
        q_mask = (quefrency >= q_low) & (quefrency <= q_high)
        q_band = cepstrum_abs[q_mask]
        q_freqs = quefrency[q_mask]

        if len(q_band) < 5:
            return {"score": 0.0, "detail": "Quefrency bandı çok dar"}

        # --- 1) Cepstral Tepe Belirginliği ---
        # Normal karpuz: belirgin tek tepe (güçlü periyodiklik)
        # Hollow: zayıf/dağınık tepeler
        peak_val = np.max(q_band)
        mean_val = np.mean(q_band)
        prominence = (peak_val - mean_val) / (mean_val + 1e-10)

        # Düşük belirginlik → bozulmuş periyodiklik → hollow
        prominence_deficit = max(0.0, HH_CEPSTRAL_PEAK_PROMINENCE - prominence)
        prominence_score = min(1.0, prominence_deficit / (HH_CEPSTRAL_PEAK_PROMINENCE + 1e-10))

        # --- 2) Cepstral Düzlük (Flatness) ---
        log_mean_ceps = np.mean(np.log(q_band + 1e-10))
        geom_mean_ceps = np.exp(log_mean_ceps)
        arith_mean_ceps = np.mean(q_band)
        cepstral_flatness = geom_mean_ceps / (arith_mean_ceps + 1e-10)
        # Yüksek düzlük → düzensiz → hollow heart
        flatness_score = min(1.0, max(0.0, cepstral_flatness * 2.0 - 0.5))

        # --- 3) Rahmeniya Varyansı ---
        # Yüksek varyans: düzensiz cepstral yapı
        cepstral_variance = float(np.var(q_band))
        variance_norm = min(1.0, cepstral_variance / (mean_val ** 2 + 1e-10))

        # --- Skor Hesapla ---
        score = (
            0.45 * prominence_score +
            0.30 * flatness_score +
            0.25 * (1.0 - min(1.0, prominence * 5))  # Düşük prominence → yüksek skor
        )
        score = min(1.0, max(0.0, score))

        return {
            "score": float(score),
            "peak_prominence": float(prominence),
            "cepstral_flatness": float(cepstral_flatness),
            "cepstral_variance": float(cepstral_variance),
            "quefrency_range": HH_QUEFRENCY_RANGE,
            "detail": (
                f"prominence={prominence:.4f} (eşik:{HH_CEPSTRAL_PEAK_PROMINENCE}), "
                f"flatness={cepstral_flatness:.4f}"
            )
        }

    # =================================================================
    # GÖSTERGE 5: HARMONİK-GÜRÜLTÜ ORANI (HNR)
    # =================================================================

    def analyze_harmonic_noise_ratio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, object]:
        """
        Harmonik-Gürültü Oranı (HNR) analizi.

        HNR: Sinyalin harmonik bileşenlerinin gürültüye oranı.

        Normal karpuz:
          - Net rezonans → yüksek HNR (>10dB)
          - Baskın harmonikler (f2, 2·f2, 3·f2)

        Hollow heart karpuz:
          - Dağınık rezonans → düşük HNR (<5dB)
          - Harmonik yapı bozuk

        Yöntem:
          1. Otokorelasyon ile periyodiklik tespit
          2. Harmonik enerji / (toplam enerji - harmonik enerji)
          3. HNR = 10·log10(harmonik / gürültü)

        Returns:
            score (0-1): 1.0 = düşük HNR → hollow heart göstergesi
        """
        N = len(audio)
        if N < 128:
            return {"score": 0.0, "hnr_db": 0.0, "detail": "Sinyal çok kısa"}

        # --- 1) Otokorelasyon tabanlı HNR ---
        # Normalize otokorelasyon
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[N - 1:]  # Sadece pozitif lag
        autocorr_norm = autocorr / (autocorr[0] + 1e-10)

        # f2 periyoduna uygun lag aralığı
        # f2 ≈ 50–500Hz → lag ≈ sr/500 .. sr/50
        min_lag = max(1, int(sr / self.freq_range[1]))
        max_lag = min(N // 2, int(sr / self.freq_range[0]))

        if max_lag <= min_lag:
            return {"score": 0.0, "hnr_db": 0.0, "detail": "Lag aralığı geçersiz"}

        lag_region = autocorr_norm[min_lag:max_lag + 1]
        if len(lag_region) < 3:
            return {"score": 0.0, "hnr_db": 0.0, "detail": "Lag bölgesi çok kısa"}

        # Otokorelasyondaki en yüksek tepe
        peak_autocorr = float(np.max(lag_region))
        peak_lag = int(np.argmax(lag_region) + min_lag)
        estimated_f0 = sr / peak_lag if peak_lag > 0 else 0.0

        # HNR hesabı: r_peak / (1 - r_peak)
        peak_autocorr = np.clip(peak_autocorr, 0.01, 0.999)
        hnr_linear = peak_autocorr / (1.0 - peak_autocorr)
        hnr_db = 10.0 * np.log10(hnr_linear + 1e-10)

        # --- 2) Spektral tabanlı HNR doğrulama ---
        fft_vals = rfft(audio * np.hanning(N))
        freqs = rfftfreq(N, d=1.0 / sr)
        magnitudes = np.abs(fft_vals) ** 2

        total_energy = np.sum(magnitudes)

        # Harmonik enerji: f0 ve harmoniklerinin ±Δf çevresindeki enerji
        harmonic_energy = 0.0
        if estimated_f0 > 0:
            delta_f = sr / N * 2  # ±2 bin genişliği
            for h in range(1, 6):  # İlk 5 harmonik
                h_freq = estimated_f0 * h
                h_mask = np.abs(freqs - h_freq) <= delta_f
                harmonic_energy += np.sum(magnitudes[h_mask])

        noise_energy = total_energy - harmonic_energy
        if noise_energy < 1e-10:
            noise_energy = 1e-10

        hnr_spectral_db = 10.0 * np.log10(
            (harmonic_energy + 1e-10) / noise_energy
        )

        # İki HNR ölçümünün ortalaması
        hnr_combined = (hnr_db + hnr_spectral_db) / 2.0

        # --- Skor Hesapla ---
        # Düşük HNR → yüksek hollow heart skoru
        if hnr_combined < HH_HNR_LOW_THRESHOLD:
            score = min(1.0, (HH_HNR_LOW_THRESHOLD - hnr_combined) /
                       (HH_HNR_LOW_THRESHOLD + 1e-10))
        else:
            score = max(0.0, 1.0 - (hnr_combined - HH_HNR_LOW_THRESHOLD) /
                       (HH_HNR_LOW_THRESHOLD + 1e-10))

        score = min(1.0, max(0.0, score))

        return {
            "score": float(score),
            "hnr_autocorr_db": float(hnr_db),
            "hnr_spectral_db": float(hnr_spectral_db),
            "hnr_combined_db": float(hnr_combined),
            "estimated_f0_hz": float(estimated_f0),
            "peak_autocorrelation": float(peak_autocorr),
            "detail": (
                f"HNR={hnr_combined:.1f}dB (eşik:{HH_HNR_LOW_THRESHOLD}dB), "
                f"f0≈{estimated_f0:.1f}Hz"
            )
        }

    # =================================================================
    # YARDIMCI: HOLLOW HEART ÖZELLİK VEKTÖRÜ
    # =================================================================

    def extract_hollow_heart_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Hollow heart tespiti için özellik vektörü üretir.

        Late Fusion ve ML modelleri için ek girdi olarak kullanılabilir.

        Returns:
            8 boyutlu HH özellik vektörü:
              [0] hh_score          - Toplam hollow heart skoru
              [1] dual_peak_score   - Çift tepe skoru
              [2] damping_score     - Sönüm skoru
              [3] spectral_score    - Spektral yayılma skoru
              [4] cepstral_score    - Cepstral skor
              [5] hnr_score         - HNR skoru
              [6] hnr_db            - HNR değeri (dB)
              [7] damping_ratio     - Sönüm oranı
        """
        if sr is None:
            sr = self.sample_rate

        result = self.detect(audio, sr, verbose=False)

        indicators = result["indicators"]
        return np.array([
            result["hh_score"],
            indicators["dual_peak"]["score"],
            indicators["damping"]["score"],
            indicators["spectral_spread"]["score"],
            indicators["cepstral"]["score"],
            indicators["hnr"]["score"],
            indicators["hnr"].get("hnr_combined_db", 0.0),
            indicators["damping"].get("damping_ratio", 0.0),
        ])

    # =================================================================
    # SİMÜLATÖR: TEST VERİ ÜRETİCİ
    # =================================================================

    @staticmethod
    def simulate_hollow_heart_signal(
        sr: int = 44100,
        duration: float = 0.5,
        f2: float = 90.0,
        is_hollow: bool = True,
        severity: float = 0.7,
        noise_level: float = 0.02
    ) -> np.ndarray:
        """
        Hollow heart test sinyali üretir.

        Args:
            sr: Örnekleme hızı
            duration: Süre (saniye)
            f2: Ana rezonans frekansı (Hz)
            is_hollow: True ise hollow heart sinyali
            severity: Hollow heart şiddeti (0-1)
            noise_level: Gürültü seviyesi

        Returns:
            Sentetik vuruş sinyali
        """
        N = int(sr * duration)
        t = np.arange(N) / sr

        if is_hollow:
            # Hollow heart: çift tepe, hızlı/düzensiz sönüm, geniş spektrum
            split = severity * 25.0  # Frekans ayrışması (Hz)
            f1 = f2 - split / 2
            f2b = f2 + split / 2

            # Hızlı ve düzensiz sönüm
            decay_fast = 8.0 + severity * 15.0
            envelope = np.exp(-decay_fast * t)
            # Düzensizlik ekle
            irregularity = severity * 0.3 * np.sin(2 * np.pi * 7 * t) * np.exp(-5 * t)
            envelope = np.maximum(0, envelope + irregularity)

            # Dual-peak sinyal
            signal = (
                0.6 * np.sin(2 * np.pi * f1 * t) * envelope +
                0.4 * np.sin(2 * np.pi * f2b * t) * envelope
            )

            # Ek gürültümsü harmonikler (bozulmuş yapı)
            for h in range(3, 7):
                amp = severity * 0.05 * np.random.uniform(0.5, 1.5)
                phase = np.random.uniform(0, 2 * np.pi)
                signal += amp * np.sin(2 * np.pi * f2 * h * t + phase) * envelope

        else:
            # Normal karpuz: tek baskın tepe, yavaş/düzgün sönüm
            decay_slow = 3.0
            envelope = np.exp(-decay_slow * t)

            signal = np.sin(2 * np.pi * f2 * t) * envelope
            # Temiz harmonikler
            signal += 0.15 * np.sin(2 * np.pi * f2 * 2 * t) * envelope
            signal += 0.05 * np.sin(2 * np.pi * f2 * 3 * t) * envelope

        # Gürültü ekle
        signal += noise_level * np.random.randn(N)

        # Attack (vuruş başlangıcı)
        attack_samples = int(0.002 * sr)
        if attack_samples > 0 and attack_samples < N:
            attack_window = np.linspace(0, 1, attack_samples)
            signal[:attack_samples] *= attack_window

        return signal

