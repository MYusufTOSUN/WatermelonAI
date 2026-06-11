"""
Ses İşleme Yardımcı Araçları

Qilin Watermelon Dataset WAV dosyalarını yükleme,
ön işleme ve .npy formatına dönüştürme.
"""

import numpy as np
import librosa
import os
from pathlib import Path
from typing import Tuple, Optional


def load_wav(
    file_path: str,
    target_sr: int = 44100,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    WAV dosyasini yukler.
    
    Qilin dataset: 2 kanalli WAV dosyalari icerir.
    mono=False ile yuklendiginde shape: (channels, samples)
    mono=True ile yuklendiginde shape: (samples,)
    
    Args:
        file_path: WAV dosya yolu
        target_sr: Hedef ornekleme hizi
        mono: Mono'ya donustur (False: tum kanallari koru)
        
    Returns:
        (audio, sr): Ses verisi ve ornekleme hizi
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return y, sr


def load_wav_right_channel(
    file_path: str,
    target_sr: int = 44100
) -> Tuple[np.ndarray, int]:
    """
    Qilin WAV dosyasindan sag kanali yukler.
    
    Qilin dataset 2 kanalli kayit yapar:
      - Sol kanal (0): Referans
      - Sag kanal (1): Vurus sesi -> analiz icin kullanilir
    
    Args:
        file_path: WAV dosya yolu
        target_sr: Hedef ornekleme hizi
        
    Returns:
        (audio_right, sr): Sag kanal ses verisi ve ornekleme hizi
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=False)
    
    if len(y.shape) > 1 and y.shape[0] >= 2:
        # 2+ kanal -> sag kanal (index 1)
        return y[1], sr
    elif len(y.shape) > 1:
        # Tek kanal ama 2D array
        return y[0], sr
    else:
        # Zaten mono
        return y, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Ses sinyalini [-1, 1] aralığına normalize eder."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = 20
) -> np.ndarray:
    """Sesin başındaki ve sonundaki sessizliği kırpar."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def segment_knock(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.1,
    min_duration: float = 0.05
) -> list:
    """
    Vuruş seslerini segmentlere ayırır.
    
    Karpuz vuruş kayıtlarında birden fazla vuruş olabilir.
    Her vuruşu ayrı ayrı tespit eder.
    
    Args:
        audio: Ses sinyali
        sr: Örnekleme hızı
        threshold: Enerji eşiği
        min_duration: Minimum vuruş süresi (saniye)
        
    Returns:
        Vuruş segmentlerinin listesi [(start_sample, end_sample), ...]
    """
    # Zarf (envelope) hesapla
    envelope = np.abs(audio)
    # Yumuşatma
    kernel_size = int(sr * 0.01)  # 10ms pencere
    if kernel_size > 0:
        kernel = np.ones(kernel_size) / kernel_size
        envelope = np.convolve(envelope, kernel, mode='same')

    # Eşik üzerindeki bölgeleri bul
    above_threshold = envelope > threshold * np.max(envelope)

    segments = []
    in_segment = False
    start = 0

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_segment:
            start = i
            in_segment = True
        elif not above_threshold[i] and in_segment:
            if (i - start) / sr >= min_duration:
                segments.append((start, i))
            in_segment = False

    if in_segment and (len(above_threshold) - start) / sr >= min_duration:
        segments.append((start, len(above_threshold)))

    return segments


def wav_to_npy(
    wav_path: str,
    output_dir: str,
    sr: int = 44100
) -> str:
    """
    WAV dosyasını .npy özellik matrisine dönüştürür.
    
    Qilin veri setindeki her WAV dosyası için:
    - Ses yükleme ve normalize etme
    - Sessizlik kırpma
    - .npy olarak kaydetme
    
    Args:
        wav_path: Giriş WAV dosya yolu
        output_dir: Çıkış dizini
        sr: Örnekleme hızı
        
    Returns:
        Çıkış .npy dosya yolu
    """
    audio, sr = load_wav(wav_path, target_sr=sr)
    audio = normalize_audio(audio)
    audio = trim_silence(audio, sr)

    os.makedirs(output_dir, exist_ok=True)

    basename = Path(wav_path).stem
    npy_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(npy_path, audio)

    return npy_path

