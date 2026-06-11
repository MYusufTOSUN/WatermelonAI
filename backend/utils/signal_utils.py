"""
Sinyal İşleme Yardımcı Araçları

Genel sinyal işleme fonksiyonları:
- Filtreleme
- Pencereli FFT
- Sinyal kalite kontrol
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


def bandpass_filter(
    data: np.ndarray,
    sample_rate: float,
    low_freq: float,
    high_freq: float,
    order: int = 4
) -> np.ndarray:
    """Butterworth bant geçiren filtre."""
    nyq = sample_rate / 2.0
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data)


def lowpass_filter(
    data: np.ndarray,
    sample_rate: float,
    cutoff_freq: float,
    order: int = 4
) -> np.ndarray:
    """Butterworth alçak geçiren filtre."""
    nyq = sample_rate / 2.0
    cutoff = cutoff_freq / nyq
    b, a = scipy_signal.butter(order, cutoff, btype='low')
    return scipy_signal.filtfilt(b, a, data)


def highpass_filter(
    data: np.ndarray,
    sample_rate: float,
    cutoff_freq: float,
    order: int = 4
) -> np.ndarray:
    """Butterworth yüksek geçiren filtre."""
    nyq = sample_rate / 2.0
    cutoff = cutoff_freq / nyq
    b, a = scipy_signal.butter(order, cutoff, btype='high')
    return scipy_signal.filtfilt(b, a, data)


def compute_fft(
    data: np.ndarray,
    sample_rate: float,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pencereli FFT hesaplar.
    
    Args:
        data: Zaman domeni sinyali
        sample_rate: Örnekleme hızı
        window: Pencere tipi ('hann', 'hamming', 'blackman', vb.)
        
    Returns:
        (frequencies, magnitudes): Frekans ve genlik dizileri
    """
    N = len(data)

    # Pencere uygula
    win = scipy_signal.get_window(window, N)
    windowed = data * win

    # FFT
    fft_vals = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    magnitudes = np.abs(fft_vals) * 2.0 / N

    return frequencies, magnitudes


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spektrogram hesaplar.
    
    Returns:
        (frequencies, times, Sxx): Frekans, zaman ve güç spektral yoğunluğu
    """
    f, t, Sxx = scipy_signal.spectrogram(
        data, fs=sample_rate,
        nperseg=nperseg, noverlap=noverlap
    )
    return f, t, Sxx


def signal_to_noise_ratio(
    signal_data: np.ndarray,
    noise_data: np.ndarray
) -> float:
    """
    SNR (Signal-to-Noise Ratio) hesaplar (dB).
    
    Args:
        signal_data: Temiz sinyal
        noise_data: Gürültü sinyali
        
    Returns:
        SNR (dB)
    """
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def find_peaks_in_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    n_peaks: int = 5,
    min_distance_hz: float = 10.0,
    sample_rate: float = 44100.0
) -> list:
    """
    Frekans spektrumunda tepe noktalarını bulur.
    
    Returns:
        [(freq, magnitude), ...] listesi
    """
    # Minimum mesafeyi örnek sayısına çevir
    freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    min_distance = max(1, int(min_distance_hz / freq_resolution))

    peak_indices, properties = scipy_signal.find_peaks(
        magnitudes, distance=min_distance, height=0
    )

    if len(peak_indices) == 0:
        return []

    # En yüksek tepeleri seç
    heights = magnitudes[peak_indices]
    sorted_idx = np.argsort(heights)[::-1][:n_peaks]

    peaks = []
    for idx in sorted_idx:
        pi = peak_indices[idx]
        peaks.append({
            "frequency": float(frequencies[pi]),
            "magnitude": float(magnitudes[pi]),
            "magnitude_db": float(20 * np.log10(magnitudes[pi] + 1e-10))
        })

    return peaks

