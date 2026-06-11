"""
MODULE_E: Sinyal Kalitesi Değerlendirici (Signal Quality Estimator)

Her modaliteden gelen sinyalin kalitesini 0-1 arasında değerlendirir.
Dynamic Late Fusion ağırlıklarını belirlemek için kullanılır.

Modaliteler:
  - Görüntü: parlaklık, blur (Laplacian varyansı), model güveni
  - Ses: SNR, clipping oranı, RMS enerji, spektral düzlük
  - Haptik/IMU: temas basıncı, faz dağılım kalitesi, RMS titreşim

Çıktı:
  visual_quality   : float [0, 1]
  acoustic_quality : float [0, 1]
  haptic_quality   : float [0, 1]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalQuality:
    """Sinyal kalitesi sonucu"""
    visual_quality: float = 0.5
    acoustic_quality: float = 0.5
    haptic_quality: float = 0.5

    def __repr__(self):
        return (
            f"SignalQuality(visual={self.visual_quality:.2f}, "
            f"acoustic={self.acoustic_quality:.2f}, "
            f"haptic={self.haptic_quality:.2f})"
        )


class SignalQualityEstimator:
    """
    Çok-modaliteli sinyal kalitesi değerlendirici.

    Dynamic Late Fusion'da ağırlık hesaplaması:
        w_i(t) = w_i_base * q_i / Σ(w_j_base * q_j)

    Burada q_i = SignalQuality skoru.
    """

    # ─────────────────────────────────────
    # GÖRÜNTÜ KALİTESİ
    # ─────────────────────────────────────

    @staticmethod
    def estimate_visual_quality(
        image: Optional[np.ndarray] = None,
        model_confidence: float = 0.5,
        xigua_confidence: float = 0.5,
    ) -> float:
        """
        Görüntü sinyal kalitesini hesapla.

        Args:
            image: Görüntü (NumPy array, H×W×3, uint8 veya float32)
            model_confidence: MRD-YOLO tespit güveni
            xigua_confidence: Xigua fusion güveni

        Returns:
            Görüntü kalite skoru [0, 1]
        """
        scores = []
        weights = []

        # 1) Model güveni (en ağırlıklı)
        scores.append(float(model_confidence))
        weights.append(2.0)

        # 2) Xigua güveni
        scores.append(float(xigua_confidence))
        weights.append(1.5)

        # 3) Piksel bazlı kalite (varsa)
        if image is not None and image.size > 0:
            # Parlaklık
            brightness = np.mean(image) / 255.0 if image.max() > 1 else np.mean(image)
            brightness_score = 1.0 - abs(brightness - 0.5) * 2.0
            scores.append(max(0.0, min(1.0, brightness_score)))
            weights.append(1.0)

            # Blur tespiti (Laplacian varyansı)
            try:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                # Yüksek varyans = net görüntü
                blur_score = min(1.0, laplacian_var / 500.0)
                scores.append(blur_score)
                weights.append(1.0)
            except ImportError:
                pass  # cv2 yoksa atla

        # Ağırlıklı ortalama
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            return max(0.0, min(1.0, weighted_sum / total_weight))
        return 0.5

    # ─────────────────────────────────────
    # SES KALİTESİ
    # ─────────────────────────────────────

    @staticmethod
    def estimate_acoustic_quality(
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> float:
        """
        Ses sinyal kalitesini hesapla.

        Args:
            audio: Ses sinyali (1D NumPy array, normalize [-1, 1])
            sample_rate: Örnekleme hızı (Hz)

        Returns:
            Ses kalite skoru [0, 1]
        """
        if audio is None or len(audio) == 0:
            return 0.0

        scores = []

        # 1) SNR tahmini
        noise_len = max(1, int(len(audio) * 0.1))
        noise_rms = np.sqrt(np.mean(audio[:noise_len] ** 2)) + 1e-10
        signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
        snr_db = 20 * np.log10(signal_rms / noise_rms)
        snr_score = max(0.0, min(1.0, snr_db / 30.0))
        scores.append(snr_score)

        # 2) Clipping oranı
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        clipping_score = 1.0 - min(1.0, clipping_ratio * 10.0)
        scores.append(clipping_score)

        # 3) Enerji
        rms = signal_rms
        if rms < 0.005:
            energy_score = 0.1
        elif rms > 0.8:
            energy_score = 0.3
        else:
            energy_score = min(1.0, rms / 0.3)
        scores.append(energy_score)

        # 4) Spektral düzlük
        try:
            fft_mag = np.abs(np.fft.rfft(audio))
            fft_mag = fft_mag[fft_mag > 0]
            if len(fft_mag) > 0:
                geo_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
                ari_mean = np.mean(fft_mag)
                flatness = geo_mean / (ari_mean + 1e-10)
                flatness_score = 1.0 - min(1.0, flatness)
                scores.append(flatness_score)
        except Exception:
            pass

        return float(np.mean(scores)) if scores else 0.5

    # ─────────────────────────────────────
    # HAPTİK / IMU KALİTESİ
    # ─────────────────────────────────────

    @staticmethod
    def estimate_haptic_quality(
        accel_data: Optional[np.ndarray] = None,
        contact_pressure: float = 0.5,
        is_contact_stable: bool = False,
        phase_distribution_quality: float = 0.5,
    ) -> float:
        """
        Haptik sinyal kalitesini hesapla.

        Args:
            accel_data: İvmeölçer z-ekseni verisi
            contact_pressure: Temas basıncı [0, 1]
            is_contact_stable: Temas kararlı mı?
            phase_distribution_quality: SRR faz dağılım kalitesi [0, 1]

        Returns:
            Haptik kalite skoru [0, 1]
        """
        scores = []

        # 1) Temas basıncı
        if contact_pressure < 0.3:
            pressure_score = contact_pressure
        elif contact_pressure > 0.95:
            pressure_score = 0.6
        else:
            pressure_score = 0.8 + 0.2 * (1.0 - abs(contact_pressure - 0.7))
        scores.append(max(0.0, min(1.0, pressure_score)))

        # 2) Kararlılık
        scores.append(1.0 if is_contact_stable else 0.3)

        # 3) Faz dağılımı
        scores.append(max(0.0, min(1.0, phase_distribution_quality)))

        # 4) IMU RMS
        if accel_data is not None and len(accel_data) > 0:
            rms = np.sqrt(np.mean(accel_data ** 2))
            imu_score = min(1.0, rms / 1.0)
            scores.append(imu_score)

        return float(np.mean(scores)) if scores else 0.5

    # ─────────────────────────────────────
    # TAM DEĞERLENDİRME
    # ─────────────────────────────────────

    def evaluate_all(
        self,
        image: Optional[np.ndarray] = None,
        model_confidence: float = 0.5,
        xigua_confidence: float = 0.5,
        audio: Optional[np.ndarray] = None,
        audio_sample_rate: int = 44100,
        accel_data: Optional[np.ndarray] = None,
        contact_pressure: float = 0.5,
        is_contact_stable: bool = False,
        phase_distribution_quality: float = 0.5,
    ) -> SignalQuality:
        """Tüm modaliteler için sinyal kalitesi değerlendirmesi."""
        return SignalQuality(
            visual_quality=self.estimate_visual_quality(
                image=image,
                model_confidence=model_confidence,
                xigua_confidence=xigua_confidence,
            ),
            acoustic_quality=self.estimate_acoustic_quality(
                audio=audio if audio is not None else np.zeros(1),
                sample_rate=audio_sample_rate,
            ),
            haptic_quality=self.estimate_haptic_quality(
                accel_data=accel_data,
                contact_pressure=contact_pressure,
                is_contact_stable=is_contact_stable,
                phase_distribution_quality=phase_distribution_quality,
            ),
        )

