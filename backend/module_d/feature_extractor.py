"""
MODULE_D: Özellik Mühendisliği Modülü (120 Öznitelik)

Koç & Akbalık (2025) referanslı 120 boyutlu akustik özellik vektörü.

Hibrit Mikrofon/İvmeölçer verisinden çıkarılan özellik grupları:

  Grup 1: MFCC İstatistikleri          -> 52 özellik (13 × 4 stat)
  Grup 2: Delta MFCC İstatistikleri    -> 26 özellik (13 × 2 stat)
  Grup 3: ZCR (Sıfır Geçiş Oranı)     ->  2 özellik
  Grup 4: Spektral Özellikler          -> 15 özellik
  Grup 5: Enerji Özellikleri           ->  4 özellik
  Grup 6: Chroma (Ton Sınıfı)         -> 12 özellik
  Grup 7: Frekans & Rezonans          ->  3 özellik
  Grup 8: Zaman Alanı Özellikleri      ->  6 özellik
  ─────────────────────────────────────────────
  TOPLAM                               -> 120 özellik

Qilin Watermelon Dataset WAV dosyalarını işler.
"""

import numpy as np
import librosa
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    AUDIO_SAMPLE_RATE,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    FREQ_BAND_LOW,
    FREQ_BAND_HIGH
)

# =====================================================================
# Sabitler
# =====================================================================
N_FEATURES = 120                 # Toplam özellik sayısı
N_CHROMA = 12                    # Chroma pitch sınıfı
N_SPECTRAL_CONTRAST_BANDS = 7   # Spektral kontrast bant sayısı


# =====================================================================
# FEATURE NAMES (120 boyut sırasıyla)
# =====================================================================

def get_feature_names(n_mfcc: int = N_MFCC) -> List[str]:
    """
    120 özelliğin isim listesini döndürür.
    Model yorumlanabilirliği ve hata ayıklama için kullanılır.
    """
    names = []

    # Grup 1: MFCC İstatistikleri (52)
    for stat in ["mean", "std", "min", "max"]:
        for i in range(1, n_mfcc + 1):
            names.append(f"mfcc_{i}_{stat}")

    # Grup 2: Delta MFCC İstatistikleri (26)
    for stat in ["mean", "std"]:
        for i in range(1, n_mfcc + 1):
            names.append(f"delta_mfcc_{i}_{stat}")

    # Grup 3: ZCR (2)
    names += ["zcr_mean", "zcr_std"]

    # Grup 4: Spektral Özellikler (15)
    names += [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
    ]
    for i in range(1, N_SPECTRAL_CONTRAST_BANDS + 1):
        names.append(f"spectral_contrast_band{i}_mean")

    # Grup 5: Enerji Özellikleri (4)
    names += ["rms_mean", "rms_std", "log_energy_mean", "log_energy_std"]

    # Grup 6: Chroma (12)
    for i in range(1, N_CHROMA + 1):
        names.append(f"chroma_{i}_mean")

    # Grup 7: Frekans & Rezonans (3)
    names += ["f2", "f2_magnitude_db", "spectral_entropy"]

    # Grup 8: Zaman Alanı (6)
    names += [
        "peak_amplitude", "crest_factor", "temporal_centroid",
        "attack_time", "decay_rate", "signal_duration_norm"
    ]

    assert len(names) == N_FEATURES, \
        f"Özellik sayısı uyuşmuyor: {len(names)} != {N_FEATURES}"
    return names


class AcousticFeatureExtractor:
    """
    120 boyutlu akustik özellik çıkarıcı.

    Koç & Akbalık (2025) referanslı kapsamlı özellik seti.
    Karpuz vuruş seslerinden olgunluk göstergesi olan
    akustik parametreleri hesaplar.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_names = get_feature_names(n_mfcc)

    # =================================================================
    # SES DOSYASI YUKLEME
    # =================================================================

    def load_audio(
        self,
        file_path: str,
        target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükler ve isteğe bağlı yeniden örnekler.

        Args:
            file_path: WAV dosyasının yolu
            target_sr: Hedef örnekleme hızı (None ise orijinal)

        Returns:
            (audio_data, sample_rate): Ses verisi ve örnekleme hızı
        """
        sr = target_sr if target_sr else self.sample_rate
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        return y, sr

    # =================================================================
    # GRUP 1: MFCC İSTATİSTİKLERİ (52 özellik)
    # =================================================================

    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) çıkarır.

        Args:
            audio: Ses sinyali (1D numpy array)
            sr: Örnekleme hızı

        Returns:
            MFCC matrisi (n_mfcc, T)
        """
        if sr is None:
            sr = self.sample_rate
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc

    def extract_mfcc_full_stats(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        MFCC'nin genişletilmiş istatistiksel özetini çıkarır.

        4 istatistik × 13 katsayı = 52 özellik
        [mean_1..13, std_1..13, min_1..13, max_1..13]

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı

        Returns:
            (4 * n_mfcc,) = (52,) boyutunda özellik vektörü
        """
        mfcc = self.extract_mfcc(audio, sr)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_min = np.min(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])

    # =================================================================
    # GRUP 2: DELTA MFCC İSTATİSTİKLERİ (26 özellik)
    # =================================================================

    def extract_delta_mfcc_stats(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Delta MFCC (birinci türev) istatistikleri.

        ΔMFCC: Zamansal değişim bilgisi - geçici olaylarda
        (vuruş sesi gibi) kritik önem taşır.

        2 istatistik × 13 katsayı = 26 özellik
        [delta_mean_1..13, delta_std_1..13]

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı

        Returns:
            (2 * n_mfcc,) = (26,) boyutunda özellik vektörü
        """
        mfcc = self.extract_mfcc(audio, sr)
        n_frames = mfcc.shape[1]

        if n_frames < 3:
            # Çok kısa sinyal: delta hesaplanamaz, sıfır döndür
            return np.zeros(2 * self.n_mfcc)

        # Delta width'i çerçeve sayısına göre ayarla
        # librosa delta default width=9, ancak n_frames < width ise hata verir
        delta_width = min(9, n_frames)
        if delta_width % 2 == 0:
            delta_width -= 1  # Tek sayı olmalı
        delta_width = max(3, delta_width)

        try:
            delta_mfcc = librosa.feature.delta(mfcc, order=1, width=delta_width)
        except Exception:
            # Fallback: basit fark hesabı
            delta_mfcc = np.diff(mfcc, axis=1)
            # Boyut eşitle (ilk çerçeveyi sıfır ile doldur)
            delta_mfcc = np.hstack([np.zeros((self.n_mfcc, 1)), delta_mfcc])

        delta_mean = np.mean(delta_mfcc, axis=1)
        delta_std = np.std(delta_mfcc, axis=1)
        return np.concatenate([delta_mean, delta_std])

    # Geriye uyumluluk: Eski 26 boyutlu MFCC stats metodu
    def extract_mfcc_stats(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Geriye uyumlu: MFCC ortalama + std (26 boyut).
        Yeni kod extract_mfcc_full_stats() kullanmalıdır.
        """
        mfcc = self.extract_mfcc(audio, sr)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    # =================================================================
    # GRUP 3: ZCR - SIFIR GEÇİŞ ORANI (2 özellik)
    # =================================================================

    def extract_zcr(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Zero-Crossing Rate (ZCR) hesaplar.

        Olgun karpuzlarda ZCR düşer; içi geçmiş karpuzlarda
        daha da düşer (boşluk nedeniyle sönümleme artar).

        Args:
            audio: Ses sinyali

        Returns:
            (zcr_frames, zcr_mean): Çerçeve bazlı ZCR ve ortalama ZCR
        """
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return zcr[0], float(np.mean(zcr))

    def extract_zcr_stats(self, audio: np.ndarray) -> np.ndarray:
        """
        ZCR mean ve std döndürür (2 özellik).

        Returns:
            [zcr_mean, zcr_std]
        """
        zcr_frames, _ = self.extract_zcr(audio)
        return np.array([float(np.mean(zcr_frames)), float(np.std(zcr_frames))])

    # =================================================================
    # GRUP 4: SPEKTRAL ÖZELLİKLER (15 özellik)
    # =================================================================

    def extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Genişletilmiş spektral özellikler.

        15 özellik:
          - Spectral Centroid (mean, std)      = 2
          - Spectral Bandwidth (mean, std)     = 2
          - Spectral Rolloff (mean, std)       = 2
          - Spectral Flatness (mean, std)      = 2
          - Spectral Contrast (7 bands × mean) = 7

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı

        Returns:
            15 spektral özellik içeren sözlük
        """
        if sr is None:
            sr = self.sample_rate

        # Spektral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Spektral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Spektral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Spektral flatness (Wiener entropy)
        # Düz spektrum = gürültü benzeri, sivri = tonal
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Spektral kontrast (7 frekans bandı)
        # Her bant için tepe ve vadi arasındaki fark
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_bands=N_SPECTRAL_CONTRAST_BANDS - 1  # +1 valley = 7 toplam
        )
        # contrast shape: (n_bands+1, T) = (7, T)

        result = {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
        }

        # 7 bant kontrast ortalamaları
        contrast_means = np.mean(contrast, axis=1)  # (7,)
        for i in range(N_SPECTRAL_CONTRAST_BANDS):
            result[f"spectral_contrast_band{i+1}_mean"] = float(contrast_means[i])

        return result

    def extract_spectral_vector(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Spektral özellikleri düz vektör olarak döndürür (15 özellik).
        """
        feats = self.extract_spectral_features(audio, sr)
        vec = [
            feats["spectral_centroid_mean"], feats["spectral_centroid_std"],
            feats["spectral_bandwidth_mean"], feats["spectral_bandwidth_std"],
            feats["spectral_rolloff_mean"], feats["spectral_rolloff_std"],
            feats["spectral_flatness_mean"], feats["spectral_flatness_std"],
        ]
        for i in range(1, N_SPECTRAL_CONTRAST_BANDS + 1):
            vec.append(feats[f"spectral_contrast_band{i}_mean"])
        return np.array(vec)

    # =================================================================
    # GRUP 5: ENERJİ ÖZELLİKLERİ (4 özellik)
    # =================================================================

    def extract_energy_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        RMS enerji ve log-enerji istatistikleri (4 özellik).

        [rms_mean, rms_std, log_energy_mean, log_energy_std]

        Log enerji vuruş seslerinde sönüm karakteristiğini yakalar.
        """
        if sr is None:
            sr = self.sample_rate

        # RMS
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]  # (1, T) -> (T,)

        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Log enerji (çerçeve bazlı)
        log_energy = np.log(rms + 1e-10)
        log_energy_mean = float(np.mean(log_energy))
        log_energy_std = float(np.std(log_energy))

        return np.array([rms_mean, rms_std, log_energy_mean, log_energy_std])

    # =================================================================
    # GRUP 6: CHROMA ÖZELLİKLERİ (12 özellik)
    # =================================================================

    def extract_chroma_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Chroma (12 pitch sınıfı) ortalamaları (12 özellik).

        Vuruş sesinin harmonik yapısını C, C#, D, ..., B
        nota sınıflarında yakalar.

        Karpuz rezonansının tonal profili olgunluğa göre değişir.
        """
        if sr is None:
            sr = self.sample_rate

        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=N_CHROMA
        )
        # chroma shape: (12, T)
        chroma_means = np.mean(chroma, axis=1)  # (12,)
        return chroma_means

    # =================================================================
    # GRUP 7: FREKANS & REZONANS (3 özellik)
    # =================================================================

    def extract_dominant_frequency(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        freq_range: Tuple[float, float] = (FREQ_BAND_LOW, FREQ_BAND_HIGH)
    ) -> Tuple[float, float]:
        """
        Dominant rezonans frekansını (f2) çıkarır.

        Fiziksel prensip:
        - f2 < 150Hz ve magnitude > 25dB → olgun karpuz
        - f2 çok düşük + ZCR düşük → içi geçmiş karpuz

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı
            freq_range: Arama frekans aralığı

        Returns:
            (f2, magnitude_db): Dominant frekans (Hz) ve genliği (dB)
        """
        if sr is None:
            sr = self.sample_rate

        N = len(audio)
        windowed = audio * np.hanning(N)
        fft_vals = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(N, d=1.0 / sr)
        magnitudes = np.abs(fft_vals) * 2.0 / N

        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        band_freqs = frequencies[mask]
        band_mags = magnitudes[mask]

        if len(band_mags) == 0:
            return 0.0, -np.inf

        peak_idx = np.argmax(band_mags)
        f2 = float(band_freqs[peak_idx])
        magnitude_db = float(20 * np.log10(band_mags[peak_idx] + 1e-10))

        return f2, magnitude_db

    def extract_spectral_entropy(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> float:
        """
        Spektral entropi hesaplar.

        Düşük entropi → tonal/harmonik (tek frekans baskın)
        Yüksek entropi → gürültü benzeri (yaygın spektrum)

        Olgun karpuz: Düşük-orta entropi (net rezonans)
        İçi geçmiş:  Yüksek entropi (dağılmış rezonans)
        """
        if sr is None:
            sr = self.sample_rate

        N = len(audio)
        fft_vals = np.abs(np.fft.rfft(audio * np.hanning(N)))
        psd = fft_vals ** 2
        psd_norm = psd / (np.sum(psd) + 1e-10)

        # Shannon entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

        # Normalize (0-1 arası): log2(N/2) max entropy
        max_entropy = np.log2(len(fft_vals))
        normalized_entropy = entropy / (max_entropy + 1e-10)

        return float(normalized_entropy)

    def extract_frequency_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Frekans & rezonans özellikleri (3 özellik).
        [f2, f2_magnitude_db, spectral_entropy]
        """
        f2, f2_db = self.extract_dominant_frequency(audio, sr)
        entropy = self.extract_spectral_entropy(audio, sr)
        return np.array([f2, f2_db, entropy])

    # =================================================================
    # GRUP 8: ZAMAN ALANI ÖZELLİKLERİ (6 özellik)
    # =================================================================

    def extract_time_domain_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Zaman alanı özellikleri (6 özellik).

        Vuruş sesi dinamiğini yakalar:
          - Peak Amplitude: Maksimum genlik
          - Crest Factor: Tepe/RMS oranı (impulsiveness)
          - Temporal Centroid: Enerjinin ağırlıklı zaman merkezi
          - Attack Time: Sinyalin tepe noktasına ulaşma süresi
          - Decay Rate: Tepe sonrası sönüm hızı
          - Signal Duration: Normalize edilmiş süre

        Olgun karpuz: Hızlı attack, yavaş decay
        Olgunlaşmamış: Daha hızlı decay (sert yüzey)
        İçi geçmiş: Yavaş attack, çok hızlı decay
        """
        if sr is None:
            sr = self.sample_rate

        N = len(audio)
        abs_audio = np.abs(audio)

        # 1. Peak Amplitude
        peak_amp = float(np.max(abs_audio))

        # 2. Crest Factor = Peak / RMS
        rms = float(np.sqrt(np.mean(audio ** 2)))
        crest_factor = peak_amp / (rms + 1e-10)

        # 3. Temporal Centroid (enerjinin ağırlıklı zaman merkezi)
        energy = audio ** 2
        total_energy = np.sum(energy) + 1e-10
        time_axis = np.arange(N) / sr
        temporal_centroid = float(np.sum(time_axis * energy) / total_energy)

        # 4. Attack Time (sinyalin %90 tepe genliğine ulaşma süresi)
        threshold_90 = 0.9 * peak_amp
        attack_samples = np.argmax(abs_audio >= threshold_90)
        attack_time = float(attack_samples / sr) if peak_amp > 1e-10 else 0.0

        # 5. Decay Rate (tepe sonrası üstel sönüm oranı tahmini)
        peak_idx = np.argmax(abs_audio)
        if peak_idx < N - 1 and peak_amp > 1e-10:
            decay_portion = abs_audio[peak_idx:]
            # %10 seviyesine düşme süresi
            threshold_10 = 0.1 * peak_amp
            below_10 = np.where(decay_portion <= threshold_10)[0]
            if len(below_10) > 0:
                decay_samples = below_10[0]
                decay_time = decay_samples / sr
                # decay_rate = 1 / decay_time (hızlı sönüm = yüksek oran)
                decay_rate = 1.0 / (decay_time + 1e-10)
            else:
                # Hiç %10'a düşmemişse yavaş sönüm
                remaining = len(decay_portion) / sr
                decay_rate = 1.0 / (remaining + 1e-10) * 0.1
        else:
            decay_rate = 0.0

        # 6. Signal Duration (normalize) = etkin sinyal süresi / toplam süre
        # Etkin sinyal: enerjinin %95'inin yer aldığı aralık
        cumulative_energy = np.cumsum(energy)
        total = cumulative_energy[-1] + 1e-10
        start_idx = np.searchsorted(cumulative_energy, 0.025 * total)
        end_idx = np.searchsorted(cumulative_energy, 0.975 * total)
        effective_duration = (end_idx - start_idx) / sr
        total_duration = N / sr
        duration_norm = float(effective_duration / (total_duration + 1e-10))

        return np.array([
            peak_amp, crest_factor, temporal_centroid,
            attack_time, decay_rate, duration_norm
        ])

    # =================================================================
    # BİRLEŞİK ÖZELLİK ÇIKARIMI (120 boyut)
    # =================================================================

    def extract_all_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> Dict[str, object]:
        """
        Tüm 120 akustik özelliği tek seferde çıkarır.

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı

        Returns:
            Tüm özellikleri içeren sözlük
        """
        if sr is None:
            sr = self.sample_rate

        # Grup 1: MFCC İstatistikleri (52)
        mfcc_full_stats = self.extract_mfcc_full_stats(audio, sr)

        # Grup 2: Delta MFCC (26)
        delta_mfcc_stats = self.extract_delta_mfcc_stats(audio, sr)

        # Grup 3: ZCR (2)
        zcr_stats = self.extract_zcr_stats(audio)

        # Grup 4: Spektral (15)
        spectral = self.extract_spectral_features(audio, sr)

        # Grup 5: Enerji (4)
        energy = self.extract_energy_features(audio, sr)

        # Grup 6: Chroma (12)
        chroma = self.extract_chroma_features(audio, sr)

        # Grup 7: Frekans & Rezonans (3)
        f2, f2_db = self.extract_dominant_frequency(audio, sr)
        spectral_entropy = self.extract_spectral_entropy(audio, sr)

        # Grup 8: Zaman alanı (6)
        time_domain = self.extract_time_domain_features(audio, sr)

        features = {
            # Grup 1
            "mfcc_full_stats": mfcc_full_stats,
            # Grup 2
            "delta_mfcc_stats": delta_mfcc_stats,
            # Grup 3
            "zcr_mean": float(zcr_stats[0]),
            "zcr_std": float(zcr_stats[1]),
            # Grup 4
            **spectral,
            # Grup 5
            "rms_mean": float(energy[0]),
            "rms_std": float(energy[1]),
            "log_energy_mean": float(energy[2]),
            "log_energy_std": float(energy[3]),
            # Grup 6
            "chroma_means": chroma,
            # Grup 7
            "f2": f2,
            "f2_magnitude_db": f2_db,
            "spectral_entropy": spectral_entropy,
            # Grup 8
            "peak_amplitude": float(time_domain[0]),
            "crest_factor": float(time_domain[1]),
            "temporal_centroid": float(time_domain[2]),
            "attack_time": float(time_domain[3]),
            "decay_rate": float(time_domain[4]),
            "signal_duration_norm": float(time_domain[5]),
        }

        return features

    def extract_feature_vector(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        120 boyutlu düz özellik vektörünü üretir.

        Koç & Akbalık (2025) referanslı tam özellik seti.
        Makine öğrenmesi modeline doğrudan girilebilir.

        Grup yapısı:
          [0:52]    MFCC istatistikleri (mean/std/min/max × 13)
          [52:78]   Delta MFCC istatistikleri (mean/std × 13)
          [78:80]   ZCR (mean, std)
          [80:95]   Spektral özellikler (centroid/bw/rolloff/flatness/contrast)
          [95:99]   Enerji özellikleri (rms/log_energy)
          [99:111]  Chroma özelikleri (12 pitch sınıfı)
          [111:114] Frekans & rezonans (f2, f2_db, entropy)
          [114:120] Zaman alanı (peak/crest/centroid/attack/decay/duration)

        Args:
            audio: Ses sinyali
            sr: Örnekleme hızı

        Returns:
            120 boyutlu 1D özellik vektörü (numpy array)
        """
        if sr is None:
            sr = self.sample_rate

        # Grup 1: MFCC İstatistikleri (52)
        mfcc_stats = self.extract_mfcc_full_stats(audio, sr)

        # Grup 2: Delta MFCC (26)
        delta_stats = self.extract_delta_mfcc_stats(audio, sr)

        # Grup 3: ZCR (2)
        zcr_stats = self.extract_zcr_stats(audio)

        # Grup 4: Spektral (15)
        spectral_vec = self.extract_spectral_vector(audio, sr)

        # Grup 5: Enerji (4)
        energy_vec = self.extract_energy_features(audio, sr)

        # Grup 6: Chroma (12)
        chroma_vec = self.extract_chroma_features(audio, sr)

        # Grup 7: Frekans & Rezonans (3)
        freq_vec = self.extract_frequency_features(audio, sr)

        # Grup 8: Zaman alanı (6)
        time_vec = self.extract_time_domain_features(audio, sr)

        # Birleştir: 52 + 26 + 2 + 15 + 4 + 12 + 3 + 6 = 120
        vector = np.concatenate([
            mfcc_stats,     # 52
            delta_stats,    # 26
            zcr_stats,      #  2
            spectral_vec,   # 15
            energy_vec,     #  4
            chroma_vec,     # 12
            freq_vec,       #  3
            time_vec,       #  6
        ])

        assert len(vector) == N_FEATURES, \
            f"Özellik vektörü boyut hatası: {len(vector)} != {N_FEATURES}"

        return vector

    # =================================================================
    # GERİYE UYUMLU: 37 BOYUTLU VEKTOR
    # =================================================================

    def extract_feature_vector_v1(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Geriye uyumlu 37 boyutlu özellik vektörü (v1).

        Mevcut eğitilmiş modeller bu boyutu bekler.
        Yeni modeller extract_feature_vector() kullanmalıdır.

        Returns:
            37 boyutlu 1D özellik vektörü
        """
        if sr is None:
            sr = self.sample_rate

        features = self.extract_all_features(audio, sr)

        mfcc_stats_v1 = self.extract_mfcc_stats(audio, sr)  # 26

        vector_parts = [
            mfcc_stats_v1,                                               # 26
            np.array([features["zcr_mean"]]),                            #  1
            np.array([features["f2"]]),                                  #  1
            np.array([features["f2_magnitude_db"]]),                     #  1
            np.array([features["spectral_centroid_mean"]]),              #  1
            np.array([features["spectral_centroid_std"]]),               #  1
            np.array([features["spectral_bandwidth_mean"]]),             #  1
            np.array([features["spectral_bandwidth_std"]]),              #  1
            np.array([features["spectral_rolloff_mean"]]),               #  1
            np.array([features["spectral_rolloff_std"]]),                #  1
            np.array([features["rms_mean"]]),                            #  1
            np.array([features["rms_std"]]),                             #  1
        ]
        return np.concatenate(vector_parts)  # 37


class AccelerometerFeatureExtractor:
    """
    İvmeölçer (accelerometer) verisinden mekanik özellik çıkarıcı.

    Haptik motor + IMU verisi üzerinden çalışır.
    SRR rekonstrükte edilmiş sinyal girdisi de kabul eder.
    """

    def __init__(self, sample_rate: float = 1600.0):
        """
        Args:
            sample_rate: Giriş sinyalinin örnekleme hızı
                         (SRR sonrası 1600Hz veya ham 100Hz)
        """
        self.sample_rate = sample_rate

    def extract_dominant_frequency(
        self,
        accel_data: np.ndarray,
        freq_range: Tuple[float, float] = (FREQ_BAND_LOW, FREQ_BAND_HIGH)
    ) -> Tuple[float, float]:
        """
        İvmeölçer verisinden dominant frekansı çıkarır.

        Args:
            accel_data: İvmeölçer sinyal verisi (1D)
            freq_range: Arama frekans aralığı

        Returns:
            (f2, magnitude_db): Dominant frekans ve dB genliği
        """
        N = len(accel_data)
        windowed = accel_data * np.hanning(N)
        fft_vals = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)
        magnitudes = np.abs(fft_vals) * 2.0 / N

        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        band_freqs = frequencies[mask]
        band_mags = magnitudes[mask]

        if len(band_mags) == 0:
            return 0.0, -np.inf

        peak_idx = np.argmax(band_mags)
        f2 = float(band_freqs[peak_idx])
        mag_db = float(20 * np.log10(band_mags[peak_idx] + 1e-10))

        return f2, mag_db

    def extract_zcr(self, accel_data: np.ndarray) -> float:
        """İvmeölçer sinyalinin sıfır geçiş oranını hesaplar."""
        zero_crossings = np.where(np.diff(np.sign(accel_data)))[0]
        duration = len(accel_data) / self.sample_rate
        return len(zero_crossings) / duration if duration > 0 else 0.0

    def extract_features(self, accel_data: np.ndarray) -> Dict[str, float]:
        """
        İvmeölçer verisinden tüm özellikleri çıkarır.

        Returns:
            Özellik sözlüğü
        """
        f2, f2_db = self.extract_dominant_frequency(accel_data)
        zcr = self.extract_zcr(accel_data)

        # İstatistiksel özellikler
        return {
            "accel_f2": f2,
            "accel_f2_db": f2_db,
            "accel_zcr": zcr,
            "accel_rms": float(np.sqrt(np.mean(accel_data ** 2))),
            "accel_peak": float(np.max(np.abs(accel_data))),
            "accel_std": float(np.std(accel_data)),
            "accel_kurtosis": float(
                np.mean((accel_data - np.mean(accel_data)) ** 4) /
                (np.std(accel_data) ** 4 + 1e-10) - 3
            )
        }
