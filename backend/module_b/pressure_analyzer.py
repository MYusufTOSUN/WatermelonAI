"""
MODULE_B: Basinc Analizi ve Temas Denetimi (Python Backend)

Ivmeolcer verisinden basinc profili cikarir ve temas kalitesini
degerlendirir. Flutter tarafindaki ContactChecker ile paralel
calisir.

Isleyis:
  1. Ham ivmeolcer verisini al (x, y, z, timestamp)
  2. Yercekimi bilesenini ayir (low-pass filtre)
  3. Dinamik basinc bilesenini hesapla (high-pass)
  4. Temas kalitesini degerlendir
  5. Basinc profilini Late Fusion'a girdi olarak ver

Bilimsel Referans:
  - Yercekimi: g ~ 9.81 m/s^2 (z-ekseni, telefon duz yatarken)
  - Temas basinc kuvveti: F = m * (a_z - g)
  - Normalize basinc: P = (a_z - g_baseline) / threshold
  
Kullanim:
  analyzer = PressureAnalyzer()
  result = analyzer.analyze(accel_data, timestamps, sample_rate=100)
  print(result["contact_quality"])
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    AUDIO_SAMPLE_RATE,
    PRESSURE_LIGHT_THRESHOLD,
    PRESSURE_GOOD_THRESHOLD,
    PRESSURE_STRONG_THRESHOLD,
    PRESSURE_STABILITY_THRESHOLD,
    PRESSURE_WINDOW_SIZE,
    PRESSURE_LP_CUTOFF,
    PRESSURE_GRAVITY_DEFAULT,
)


class PressureAnalyzer:
    """
    Ivmeolcer tabanli basinc analizi.
    
    Flutter'dan gelen ham ivmeolcer verisini isler ve
    temas kalitesini degerlendirir.
    """

    def __init__(
        self,
        gravity_default: float = PRESSURE_GRAVITY_DEFAULT,
        light_threshold: float = PRESSURE_LIGHT_THRESHOLD,
        good_threshold: float = PRESSURE_GOOD_THRESHOLD,
        strong_threshold: float = PRESSURE_STRONG_THRESHOLD,
        stability_threshold: float = PRESSURE_STABILITY_THRESHOLD,
        window_size: int = PRESSURE_WINDOW_SIZE,
    ):
        self.gravity_default = gravity_default
        self.light_threshold = light_threshold
        self.good_threshold = good_threshold
        self.strong_threshold = strong_threshold
        self.stability_threshold = stability_threshold
        self.window_size = window_size

        # Kalibrasyon durumu
        self._baseline_gravity = gravity_default
        self._is_calibrated = False

    # =================================================================
    # KALIBRASYON
    # =================================================================

    def calibrate(
        self,
        idle_accel_z: np.ndarray,
        sample_rate: float = 100.0
    ) -> Dict[str, float]:
        """
        Bosta yercekimi referansini kalibre eder.
        
        Telefon duz yatarken alinan ivmeolcer z-ekseni verisinden
        yercekimi bazini hesaplar.
        
        Args:
            idle_accel_z: Bosta z-ekseni ivmeolcer verisi (m/s^2)
            sample_rate: Ornekleme hizi (Hz)
            
        Returns:
            Kalibrasyon sonuclari
        """
        if len(idle_accel_z) < 10:
            raise ValueError("Kalibrasyon icin en az 10 ornek gerekli.")

        self._baseline_gravity = np.mean(idle_accel_z)
        gravity_std = np.std(idle_accel_z)
        self._is_calibrated = True

        return {
            "baseline_gravity": float(self._baseline_gravity),
            "gravity_std": float(gravity_std),
            "n_samples": len(idle_accel_z),
            "sample_rate": sample_rate,
            "duration_s": len(idle_accel_z) / sample_rate,
            "is_calibrated": True,
        }

    # =================================================================
    # ANA ANALIZ
    # =================================================================

    def analyze(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        timestamps_ns: Optional[np.ndarray] = None,
        sample_rate: float = 100.0,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """
        Tam basinc analizi yapar.
        
        Args:
            accel_x: X-ekseni ivme verisi (m/s^2)
            accel_y: Y-ekseni ivme verisi (m/s^2)
            accel_z: Z-ekseni ivme verisi (m/s^2)
            timestamps_ns: Zaman damgalari (nanosaniye) - opsiyonel
            sample_rate: Ornekleme hizi (Hz)
            verbose: Detayli cikti
            
        Returns:
            Basinc analiz sonuclari
        """
        n_samples = len(accel_z)
        if n_samples < self.window_size:
            return self._empty_result("Yetersiz veri")

        # 1) Buyukluk (magnitude) hesapla
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # 2) Yercekimi sapmasini hesapla
        gravity_deviation = np.abs(accel_z - self._baseline_gravity)
        mean_deviation = np.mean(gravity_deviation)

        # 3) Kararlilik (standart sapma)
        mag_std = np.std(magnitude)
        is_stable = mag_std < self.stability_threshold

        # 4) Titresim RMS (yuksek geciren filtre)
        mean_z = np.mean(accel_z)
        hp_z = accel_z - mean_z  # Basit yuksek geciren
        vibration_rms = np.sqrt(np.mean(hp_z**2))

        # 5) Egim acisi
        mean_z_val = np.mean(accel_z)
        tilt_angle_rad = np.arccos(np.clip(mean_z_val / 9.81, -1.0, 1.0))
        tilt_angle_deg = np.degrees(tilt_angle_rad)

        # 6) Normalize basinc (0-1)
        contact_pressure = np.clip(mean_deviation / self.good_threshold, 0.0, 1.0)

        # 7) Temas kalitesi
        quality, message = self._evaluate_quality(
            mean_deviation, is_stable
        )

        # 8) Pencereli analiz (zaman serisi)
        pressure_profile = self._compute_pressure_profile(
            accel_z, sample_rate
        )

        # 9) Temas baslangic/bitis tespiti
        contact_events = self._detect_contact_events(
            gravity_deviation, sample_rate
        )

        if verbose:
            print(f"\n{'='*50}")
            print(f"  BASINC ANALIZI")
            print(f"  Ornekler: {n_samples} @ {sample_rate}Hz")
            print(f"{'='*50}")
            print(f"  Yercekimi Sapma : {mean_deviation:.3f} m/s^2")
            print(f"  Basinc (norm)   : {contact_pressure:.2f}")
            print(f"  Titresim RMS    : {vibration_rms:.4f}")
            print(f"  Egim Acisi      : {tilt_angle_deg:.1f} derece")
            print(f"  Kararlilik      : {'Evet' if is_stable else 'Hayir'} (std={mag_std:.3f})")
            print(f"  Kalite          : {quality}")
            print(f"  Mesaj           : {message}")
            print(f"  Temas Olaylari  : {len(contact_events)} adet")
            print(f"{'='*50}")

        return {
            "contact_pressure": float(contact_pressure),
            "gravity_deviation": float(mean_deviation),
            "vibration_rms": float(vibration_rms),
            "tilt_angle_deg": float(tilt_angle_deg),
            "magnitude_std": float(mag_std),
            "is_stable": bool(is_stable),
            "contact_quality": quality,
            "message": message,
            "pressure_profile": pressure_profile,
            "contact_events": contact_events,
            "n_samples": n_samples,
            "sample_rate": sample_rate,
            "is_calibrated": self._is_calibrated,
        }

    # =================================================================
    # BASINC PROFILI (Late Fusion girdi vektoru)
    # =================================================================

    def extract_pressure_features(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        sample_rate: float = 100.0,
    ) -> np.ndarray:
        """
        Basinc ozellik vektoru cikarir (Late Fusion icin).
        
        7 boyutlu haptik ozellik vektoru:
          [0] contact_pressure   - Normalize basinc (0-1)
          [1] gravity_deviation  - Yercekimi sapmasi (m/s^2)
          [2] vibration_rms      - Titresim RMS
          [3] tilt_angle_norm    - Normalize egim acisi (0-1)
          [4] magnitude_mean     - Ortalama buyukluk
          [5] stability_score    - Kararlilik skoru (0-1)
          [6] contact_quality_score - Temas kalite skoru (0-1)
          
        Args:
            accel_x, accel_y, accel_z: 3-eksen ivmeolcer verisi
            sample_rate: Ornekleme hizi
            
        Returns:
            7 boyutlu ozellik vektoru
        """
        result = self.analyze(accel_x, accel_y, accel_z,
                              sample_rate=sample_rate, verbose=False)

        # Egim normalizasyonu (0-90 derece -> 0-1)
        tilt_norm = np.clip(result["tilt_angle_deg"] / 90.0, 0.0, 1.0)

        # Magnitude ortalaması
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        mag_mean = np.mean(magnitude)

        # Kararlilik skoru (dusuk std = yuksek skor)
        stability_score = np.clip(
            1.0 - result["magnitude_std"] / self.stability_threshold, 0.0, 1.0
        )

        # Temas kalite skoru
        quality_map = {
            "no_contact": 0.0,
            "light_contact": 0.3,
            "good_contact": 0.8,
            "too_strong": 0.5,  # Cok guclu de ideal degil
        }
        quality_score = quality_map.get(result["contact_quality"], 0.0)

        features = np.array([
            result["contact_pressure"],
            result["gravity_deviation"],
            result["vibration_rms"],
            tilt_norm,
            mag_mean,
            stability_score,
            quality_score,
        ], dtype=np.float64)

        return features

    # =================================================================
    # PENCERELI ANALIZ (Basinc Profili)
    # =================================================================

    def _compute_pressure_profile(
        self,
        accel_z: np.ndarray,
        sample_rate: float,
    ) -> Dict[str, object]:
        """
        Pencereli basinc profili hesaplar.
        
        Returns:
            Basinc profili metrikleri
        """
        n = len(accel_z)
        window = self.window_size
        n_windows = max(1, n // window)

        pressures = []
        timestamps = []

        for i in range(n_windows):
            start = i * window
            end = min(start + window, n)
            segment = accel_z[start:end]

            dev = np.abs(np.mean(segment) - self._baseline_gravity)
            pressure = np.clip(dev / self.good_threshold, 0.0, 1.0)
            pressures.append(float(pressure))
            timestamps.append(float(start / sample_rate))

        pressures = np.array(pressures)

        return {
            "values": pressures.tolist(),
            "timestamps": timestamps,
            "mean_pressure": float(np.mean(pressures)),
            "max_pressure": float(np.max(pressures)),
            "min_pressure": float(np.min(pressures)),
            "std_pressure": float(np.std(pressures)),
            "n_windows": n_windows,
        }

    # =================================================================
    # TEMAS OLAYI TESPITI
    # =================================================================

    def _detect_contact_events(
        self,
        gravity_deviation: np.ndarray,
        sample_rate: float,
    ) -> List[Dict[str, float]]:
        """
        Temas baslangic ve bitis zamanlarini tespit eder.
        
        Esik gecis analizi ile temas olaylarini bulur.
        """
        # Esik: light_threshold'un uzerinde = temas var
        in_contact = gravity_deviation > self.light_threshold

        events = []
        contact_start = None

        for i in range(len(in_contact)):
            if in_contact[i] and contact_start is None:
                contact_start = i
            elif not in_contact[i] and contact_start is not None:
                # Temas sona erdi
                duration = (i - contact_start) / sample_rate
                peak_dev = float(np.max(gravity_deviation[contact_start:i]))
                events.append({
                    "start_time_s": float(contact_start / sample_rate),
                    "end_time_s": float(i / sample_rate),
                    "duration_s": float(duration),
                    "peak_deviation": peak_dev,
                })
                contact_start = None

        # Hala temasda ise
        if contact_start is not None:
            duration = (len(in_contact) - contact_start) / sample_rate
            peak_dev = float(np.max(gravity_deviation[contact_start:]))
            events.append({
                "start_time_s": float(contact_start / sample_rate),
                "end_time_s": float(len(in_contact) / sample_rate),
                "duration_s": float(duration),
                "peak_deviation": peak_dev,
            })

        return events

    # =================================================================
    # KALITE DEGERLENDIRME
    # =================================================================

    def _evaluate_quality(
        self,
        mean_deviation: float,
        is_stable: bool,
    ) -> Tuple[str, str]:
        """
        Temas kalitesini degerlendirir.
        
        Returns:
            (quality_key, message)
        """
        if mean_deviation > self.strong_threshold:
            return "too_strong", "Cok fazla basinc! Hafifce basin."
        elif mean_deviation > self.good_threshold:
            if is_stable:
                return "good_contact", "Mukemmel temas! Olcume hazir."
            else:
                return "good_contact", "Iyi temas. Sabit tutun."
        elif mean_deviation > self.light_threshold:
            return "light_contact", "Biraz daha bastirin."
        else:
            return "no_contact", "Telefonu karpuza bastirin."

    def _empty_result(self, reason: str) -> Dict[str, object]:
        """Bos sonuc olustur."""
        return {
            "contact_pressure": 0.0,
            "gravity_deviation": 0.0,
            "vibration_rms": 0.0,
            "tilt_angle_deg": 0.0,
            "magnitude_std": 0.0,
            "is_stable": False,
            "contact_quality": "no_contact",
            "message": reason,
            "pressure_profile": {"values": [], "timestamps": []},
            "contact_events": [],
            "n_samples": 0,
            "sample_rate": 0.0,
            "is_calibrated": self._is_calibrated,
        }

    # =================================================================
    # SIMULASYON (Test icin)
    # =================================================================

    @staticmethod
    def simulate_contact_data(
        duration_s: float = 5.0,
        sample_rate: float = 100.0,
        contact_type: str = "good",
        noise_level: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Test icin simule ivmeolcer verisi olusturur.
        
        Args:
            duration_s: Sure (saniye)
            sample_rate: Ornekleme hizi (Hz)
            contact_type: Temas tipi ('none', 'light', 'good', 'strong')
            noise_level: Gurultu seviyesi
            
        Returns:
            (accel_x, accel_y, accel_z) tuple
        """
        n_samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, n_samples)

        # Yercekimi bazı (z-ekseni)
        g = 9.81

        # Temas tipine gore sapma
        deviation_map = {
            "none": 0.05,      # ~0 sapma
            "light": 0.5,      # Hafif
            "good": 1.2,       # Ideal
            "strong": 3.0,     # Cok guclu
            "variable": None,  # Degisken
        }

        deviation = deviation_map.get(contact_type, 1.0)

        if contact_type == "variable":
            # Degisken basinc profili (baslangicta yok, ortada iyi, sonda azaliyor)
            envelope = np.zeros(n_samples)
            ramp_up = int(0.2 * n_samples)
            hold = int(0.5 * n_samples)
            ramp_down = n_samples - ramp_up - hold

            envelope[:ramp_up] = np.linspace(0, 1.2, ramp_up)
            envelope[ramp_up:ramp_up + hold] = 1.2
            envelope[ramp_up + hold:] = np.linspace(1.2, 0.3, ramp_down)
            deviation = envelope
        else:
            deviation = np.full(n_samples, deviation)

        # Z-ekseni: yercekimi + basinc sapmasi
        accel_z = g + deviation + noise_level * np.random.randn(n_samples)

        # X ve Y: genelde kucuk (yanal titresim)
        accel_x = noise_level * 0.5 * np.random.randn(n_samples)
        accel_y = noise_level * 0.5 * np.random.randn(n_samples)

        return accel_x, accel_y, accel_z

