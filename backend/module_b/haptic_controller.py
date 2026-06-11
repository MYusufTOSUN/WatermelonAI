"""
MODULE_B: Aktif Haptik Algılama Modülü

Akıllı telefonun LRA (Linear Resonant Actuator) titreşim motorunu
kullanarak kontrollü frekans taraması gerçekleştirir.

İşleyiş:
1. Telefon karpuz yüzeyine temas ettirilir
2. LRA motor 100-400Hz arası frekans taraması yapar
3. Eşzamanlı olarak IMU (ivmeölçer) ve mikrofon kaydı alınır
4. Yanıt sinyali analiz edilerek mekanik özellikler çıkarılır

Not: Bu modül Python tarafında sinyal tasarımı ve simülasyonunu sağlar.
Gerçek haptik kontrol Flutter tarafında (Android/iOS native) yapılır.
"""

import numpy as np
from typing import Tuple, Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    HAPTIC_FREQ_START,
    HAPTIC_FREQ_END,
    HAPTIC_SWEEP_DURATION,
    HAPTIC_STEP_MS,
    AUDIO_SAMPLE_RATE
)


class HapticSweepDesigner:
    """
    Haptik frekans taraması sinyal tasarımcısı.
    
    LRA motor için optimum frekans taraması parametrelerini
    hesaplar ve referans sinyali üretir.
    """

    def __init__(
        self,
        freq_start: float = HAPTIC_FREQ_START,
        freq_end: float = HAPTIC_FREQ_END,
        duration: float = HAPTIC_SWEEP_DURATION,
        step_ms: float = HAPTIC_STEP_MS
    ):
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.duration = duration
        self.step_ms = step_ms

    def generate_chirp_signal(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chirp (frekans taraması) referans sinyali üretir.
        
        Args:
            sample_rate: Çıkış örnekleme hızı
            method: Tarama yöntemi ('linear', 'logarithmic', 'quadratic')
            
        Returns:
            (time_axis, chirp_signal): Zaman ekseni ve chirp sinyali
        """
        from scipy.signal import chirp

        t = np.linspace(0, self.duration, int(sample_rate * self.duration))
        signal = chirp(
            t,
            f0=self.freq_start,
            f1=self.freq_end,
            t1=self.duration,
            method=method
        )

        return t, signal

    def generate_stepped_frequencies(self) -> List[Dict[str, float]]:
        """
        Adım adım frekans listesi oluşturur (haptik motor kontrolü için).
        
        Her adımda motor belirli bir frekansta çalıştırılır.
        Flutter tarafına gönderilecek frekans planı.
        
        Returns:
            Her adım için frekans ve süre bilgisi
        """
        n_steps = int(self.duration * 1000 / self.step_ms)
        freq_step = (self.freq_end - self.freq_start) / max(n_steps - 1, 1)

        steps = []
        for i in range(n_steps):
            freq = self.freq_start + i * freq_step
            start_time_ms = i * self.step_ms
            steps.append({
                "step_index": i,
                "frequency_hz": freq,
                "start_time_ms": start_time_ms,
                "duration_ms": self.step_ms,
                "amplitude": 1.0  # Normalize genlik
            })

        return steps

    def generate_phase_shifted_protocol(
        self,
        n_sweeps: int = 16
    ) -> List[Dict]:
        """
        SRR için faz kaydırmalı çoklu tarama protokolü oluşturur.
        
        Her taramada farklı bir faz kayması kullanılarak
        stroboskopik örnekleme prensibi uygulanır.
        
        Args:
            n_sweeps: Toplam tarama sayısı (= SRR upsample faktörü)
            
        Returns:
            Her tarama için faz kayması ve zamanlama bilgisi
        """
        protocol = []
        phase_step = 1.0 / n_sweeps  # Her taramada faz kayması (periyot oranı)

        for sweep_idx in range(n_sweeps):
            phase_offset_ratio = sweep_idx * phase_step
            phase_offset_ms = phase_offset_ratio * self.step_ms

            protocol.append({
                "sweep_index": sweep_idx,
                "phase_offset_ratio": phase_offset_ratio,
                "phase_offset_ms": phase_offset_ms,
                "freq_start": self.freq_start,
                "freq_end": self.freq_end,
                "duration": self.duration
            })

        return protocol


class IdleProfileCalibrator:
    """
    Girişim iptali için boşta titreşim profili kalibrasyoncusu.
    
    Haptik motor açılmadan önce ortam titreşim profili alınır.
    Bu profil daha sonra aktif algılama verisinden çıkarılır.
    """

    def __init__(self, calibration_duration: float = 1.0):
        """
        Args:
            calibration_duration: Kalibrasyon süresi (saniye)
        """
        self.calibration_duration = calibration_duration
        self.idle_profile = None
        self.idle_spectrum = None

    def record_idle_profile(self, imu_data: np.ndarray, sample_rate: float):
        """
        Boşta titreşim profilini kaydeder.
        
        Args:
            imu_data: Haptik motor kapalıyken alınan IMU verisi
            sample_rate: Örnekleme hızı
        """
        self.idle_profile = imu_data.copy()
        self.idle_spectrum = np.fft.rfft(imu_data)
        print(f"[MODULE_B] Boşta profili kaydedildi: {len(imu_data)} örnek @ {sample_rate}Hz")

    def get_idle_profile(self) -> np.ndarray:
        """Kaydedilmiş boşta profilini döndürür."""
        if self.idle_profile is None:
            raise ValueError("Boşta profili henüz kaydedilmedi. Önce record_idle_profile() çağırın.")
        return self.idle_profile

    def subtract_idle(
        self,
        active_signal: np.ndarray
    ) -> np.ndarray:
        """
        Aktif sinyalden boşta profilini çıkarır (zaman domeni).
        
        Args:
            active_signal: Haptik motor aktifken alınan sinyal
            
        Returns:
            Temizlenmiş sinyal
        """
        idle = self.get_idle_profile()

        # Uzunluk eşitle
        min_len = min(len(active_signal), len(idle))
        clean = active_signal[:min_len] - idle[:min_len]

        return clean


class HapticSimulator:
    """
    Test ve geliştirme için haptik yanıt simülatörü.
    
    Gerçek donanım olmadan sinyal işleme pipeline'ını
    test etmek için kullanılır.
    """

    @staticmethod
    def simulate_watermelon_response(
        chirp_signal: np.ndarray,
        sample_rate: int,
        ripeness: str = "ripe",
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Karpuz yanıt sinyalini simüle eder.
        
        Args:
            chirp_signal: Giriş chirp sinyali
            sample_rate: Örnekleme hızı
            ripeness: Olgunluk durumu ('immature', 'ripe', 'overripe')
            noise_level: Gürültü seviyesi
            
        Returns:
            Simüle edilmiş yanıt sinyali
        """
        t = np.arange(len(chirp_signal)) / sample_rate

        # Olgunluk durumuna göre rezonans parametreleri
        params = {
            "immature": {"f_resonance": 200, "damping": 0.05, "amplitude": 0.3},
            "ripe": {"f_resonance": 120, "damping": 0.08, "amplitude": 0.6},
            "overripe": {"f_resonance": 80, "damping": 0.15, "amplitude": 0.4}
        }

        p = params.get(ripeness, params["ripe"])

        # Sönümlenmiş sinüzoidal yanıt
        response = p["amplitude"] * np.exp(-p["damping"] * t * p["f_resonance"]) * \
                   np.sin(2 * np.pi * p["f_resonance"] * t)

        # Giriş sinyali ile konvolüsyon
        convolved = np.convolve(chirp_signal, response[:min(len(response), 1000)], mode='same')
        convolved = convolved / (np.max(np.abs(convolved)) + 1e-10)

        # Gürültü ekle
        noise = noise_level * np.random.randn(len(convolved))
        return convolved + noise

