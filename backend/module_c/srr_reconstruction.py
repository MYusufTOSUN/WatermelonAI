"""
MODULE_C: Supersampling Rate Reconstruction (SRR) Algoritmasi
Vi-Liquid Metodolojisi Implementasyonu

Akilli telefon ivmeolcer limiti: ~100Hz (Android/iOS API siniri)
Hedef titresim frekansi: 300-500Hz (karpuz rezonansi)
SRR cikisi: 1600Hz sanal ornekleme hizi

=================================================================
TEMEL PRENSIP (Vi-Liquid):
=================================================================
1. LRA motor bilinen bir frekansta (167Hz) titresim uretir
   - Periyot T_vib = 1/167 = ~5.988ms
   
2. IMU 100Hz'de ornekler (T_imu = 10ms)
   - T_imu > T_vib oldugu icin her ornek farkli fazda duser

3. Cift Yollu Geri Kazanim:
   A) Periyot Katlama: LRA frekansindaki zorunlu yanitı geri kazanir
      (sertlik olcumu icin)
   B) NUFFT-OMP: Tum frekans bilesenlerini geri kazanir
      (dogal frekans f2 tespiti icin)

4. Faz Hesabi:
   phi_k = (t_k mod T_vib) / T_vib   -> [0, 1)
   
5. OMP (Orthogonal Matching Pursuit):
   - Frekans domeninde seyrek (sparse) geri kazanim
   - Ham orneklerin zamanlama bilgisiyle dogal frekans tespiti

6. SPI (Straight Path Interference) Giderimi:
   - Cihaz-kaynak dogru yol girisimini frekans domeninde cikar

Referanslar:
  - Vi-Liquid: Vibration-based Liquid Quality Assessment (IPSN 2020)
  - SystemClock.elapsedRealtimeNanos() tabanli faz indeksleme
=================================================================
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from scipy.fft import rfft, irfft, rfftfreq
from typing import Tuple, Optional, Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    IMU_NATIVE_RATE,
    SRR_TARGET_RATE,
    SRR_UPSAMPLE_FACTOR
)


# =================================================================
# ANA SINIF: SRR Rekonstruktor
# =================================================================

class SRRReconstructor:
    """
    Supersampling Rate Reconstruction (SRR) Motoru.
    Vi-Liquid metodolojisi ile 100Hz -> 1600Hz sinyal rekonstruksiyonu.
    
    Akis:
      1. phase_offset_sampling()     -> Faz ofset orneklemesi
      2. period_folding()            -> Periyot katlama (zorunlu yanit)
      3. nufft_omp_recovery()        -> NUFFT-OMP f2 geri kazanimi
      4. spi_noise_cancellation()    -> SPI gurultu giderimi
      5. extract_dominant_f2()       -> Dominant frekans cikarimi
    """

    def __init__(
        self,
        native_rate: int = IMU_NATIVE_RATE,
        target_rate: int = SRR_TARGET_RATE,
        upsample_factor: int = SRR_UPSAMPLE_FACTOR,
        lra_frequency: float = 167.0
    ):
        self.native_rate = native_rate
        self.target_rate = target_rate
        self.upsample_factor = upsample_factor
        self.lra_frequency = lra_frequency
        self.lra_period_ns = int(1e9 / lra_frequency)
        self.lra_period_s = 1.0 / lra_frequency

    # =============================================================
    # PHASE 1: FAZ OFSET ORNEKLEMESI
    # =============================================================

    def phase_offset_sampling(
        self,
        timestamps_ns: np.ndarray,
        imu_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Her IMU orneginin LRA titresim periyodu icindeki faz ofsetini hesaplar.
        
        Vi-Liquid faz hesabi:
            phi_k = (t_k mod T_vib) / T_vib
        
        Args:
            timestamps_ns: IMU orneklerinin nanosaniye zaman damgalari (N,)
            imu_data: Ivmeolcer verileri (N,) veya (N,3)
            
        Returns:
            phases: [0, 1) araliginda faz degerleri (N,)
            cycle_indices: Her ornegin ait oldugu periyot indeksi (N,)
            magnitudes: Ivme degerleri (N,)
        """
        t_relative_ns = timestamps_ns - timestamps_ns[0]
        phases = (t_relative_ns % self.lra_period_ns).astype(np.float64) / self.lra_period_ns
        cycle_indices = (t_relative_ns // self.lra_period_ns).astype(np.int64)
        
        if imu_data.ndim == 2 and imu_data.shape[1] == 3:
            magnitudes = np.sqrt(np.sum(imu_data ** 2, axis=1))
        elif imu_data.ndim == 2 and imu_data.shape[0] == 3:
            magnitudes = np.sqrt(np.sum(imu_data ** 2, axis=0))
        else:
            magnitudes = imu_data.astype(np.float64).copy()
        
        n_cycles = int(cycle_indices[-1]) + 1 if len(cycle_indices) > 0 else 0
        unique_phases = len(np.unique(np.round(phases, 3)))
        
        print(f"[SRR-Phase] {len(timestamps_ns)} ornek, {n_cycles} cevrim")
        print(f"[SRR-Phase] Benzersiz faz pozisyonu: ~{unique_phases}")
        print(f"[SRR-Phase] LRA: {self.lra_frequency}Hz, Periyot: {self.lra_period_ns/1e6:.3f}ms")
        
        return phases, cycle_indices, magnitudes

    # =============================================================
    # PHASE 2: PERIYOT KATLAMA (Zorunlu Yanit icin)
    # =============================================================

    def period_folding(
        self,
        magnitudes: np.ndarray,
        phases: np.ndarray,
        n_bins: int = None,
        outlier_rejection: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Periyot Katlama: LRA frekansindaki zorunlu yaniti geri kazanir.
        
        NOT: Bu yontem SADECE LRA frekansi ve harmoniklerini korur.
        Dogal frekans (f2) icin nufft_omp_recovery() kullanin.
        
        Args:
            magnitudes: Ivme degerleri (N,)
            phases: Normalize faz degerleri [0,1) (N,)
            n_bins: Bin sayisi (varsayilan: upsample_factor)
            outlier_rejection: Outlier reddi uygula
            
        Returns:
            bin_centers: Faz bin merkezleri (n_bins,)
            folded_signal: Katlanmis sinyal (n_bins,)
        """
        if n_bins is None:
            n_bins = self.upsample_factor

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_samples: List[List[float]] = [[] for _ in range(n_bins)]
        
        bin_indices = np.digitize(phases, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        for i in range(len(magnitudes)):
            bin_samples[bin_indices[i]].append(float(magnitudes[i]))
        
        folded_signal = np.zeros(n_bins)
        valid_mask = np.zeros(n_bins, dtype=bool)
        
        for b in range(n_bins):
            samples = np.array(bin_samples[b])
            if len(samples) < 1:
                continue
            if outlier_rejection and len(samples) >= 5:
                q1, q3 = np.percentile(samples, [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                clean = samples[(samples >= lower) & (samples <= upper)]
                folded_signal[b] = np.median(clean) if len(clean) > 0 else np.median(samples)
            else:
                folded_signal[b] = np.median(samples)
            valid_mask[b] = True
        
        n_valid = np.sum(valid_mask)
        n_empty = n_bins - n_valid
        
        if n_empty > 0 and n_valid >= 4:
            vc = bin_centers[valid_mask]
            vv = folded_signal[valid_mask]
            try:
                cs = CubicSpline(vc, vv, bc_type='periodic')
                folded_signal = cs(bin_centers)
            except Exception:
                folded_signal = np.interp(bin_centers, vc, vv)
        elif n_empty > 0 and n_valid >= 2:
            folded_signal = np.interp(bin_centers, bin_centers[valid_mask], folded_signal[valid_mask])
        
        return bin_centers, folded_signal

    # =============================================================
    # PHASE 3: NUFFT-OMP SEYREK GERI KAZANIM
    # =============================================================

    def nufft_omp_recovery(
        self,
        imu_data: np.ndarray,
        timestamps_ns: np.ndarray,
        n_atoms: int = 8,
        freq_range: Tuple[float, float] = (50.0, 800.0),
        dictionary_size: int = 512,
        target_n_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        NUFFT-OMP: Ham orneklerden tum frekans bilesenlerini geri kazanir.
        
        Periyot katlama YERINE dogrudan ham ornekleri kullanir.
        Bu sayede f2 (dogal frekans) dahil tum bilesenler korunur.
        
        Algoritma:
          1. Ham orneklerin gercek zamanlarini kullan (non-uniform)
          2. Sozluk D: sin(2*pi*f*t_k) ve cos(2*pi*f*t_k) her aday f icin
          3. OMP ile en uygun frekanslari sec
          4. Secilen frekanslari kullanarak hedef hizda (1600Hz) yeniden sentezle
        
        Args:
            imu_data: Ham IMU verisi (N,)
            timestamps_ns: Nanosaniye zaman damgalari (N,)
            n_atoms: Secilecek frekans atom sayisi
            freq_range: Arama araligi (Hz)
            dictionary_size: Sozluk boyutu
            target_n_samples: Cikis ornek sayisi (None: otomatik)
            
        Returns:
            time_axis: Cikis zaman ekseni (saniye)
            reconstructed: 1600Hz rekonstrukte sinyal
            frequency_atoms: Secilen frekanslar
            omp_details: Detaylar
        """
        N = len(imu_data)
        
        # Gercek zamanlar (saniye)
        t_samples = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
        total_duration = t_samples[-1]
        
        # Aday frekanslar
        candidate_freqs = np.linspace(freq_range[0], freq_range[1], dictionary_size)
        
        # Sozluk matrisi D: (N, 2*dictionary_size)
        # Her aday frekans icin sin ve cos atomlari
        n_dict = 2 * dictionary_size
        D = np.zeros((N, n_dict))
        
        for i, f in enumerate(candidate_freqs):
            D[:, 2 * i] = np.sin(2 * np.pi * f * t_samples)
            D[:, 2 * i + 1] = np.cos(2 * np.pi * f * t_samples)
        
        # Normalize et
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        D_norm = D / norms
        norms_flat = norms[0]

        # OMP Algoritmasi (Gram matrix cached, incremental support)
        y = imu_data.astype(np.float64).copy()
        support = []

        max_iters = min(n_atoms * 2, n_dict, N // 2)

        # Precompute D_norm^T y once (correlations basislari)
        Dty_norm = D_norm.T @ y  # (n_dict,)
        y_energy_sq = float(np.dot(y, y))

        # Incremental Gram sutunlari: her destek atomu icin (n_dict,) vektor
        gram_cols = []  # list of np.ndarray (n_dict,)
        residual_corr = Dty_norm.copy()  # D_norm^T @ residual, incrementally updated
        x_norm = None  # normalized-space coefficients on support

        for iteration in range(max_iters):
            # En yuksek korelasyonlu atomu bul (destekleri sifirla)
            correlations_abs = np.abs(residual_corr)
            if support:
                correlations_abs[np.asarray(support)] = 0.0

            best = int(np.argmax(correlations_abs))
            if correlations_abs[best] < 1e-10:
                break

            support.append(best)

            # Yeni Gram sutunu (cache): D_norm^T @ D_norm[:, best]
            gram_cols.append(D_norm.T @ D_norm[:, best])

            # Kucuk Gram alt-matrisi G_s = (k,k) ve RHS
            G_support_cols = np.column_stack(gram_cols)  # (n_dict, k)
            support_arr = np.asarray(support)
            G_s = G_support_cols[support_arr]            # (k, k) simetrik PSD
            rhs = Dty_norm[support_arr]                  # (k,)

            try:
                x_norm = np.linalg.solve(G_s, rhs)
            except np.linalg.LinAlgError:
                # Tekil sistem: kucuk bir regularizasyonla cek
                try:
                    x_norm = np.linalg.solve(
                        G_s + 1e-8 * np.eye(len(support)), rhs
                    )
                except np.linalg.LinAlgError:
                    support.pop()
                    gram_cols.pop()
                    break

            # Rezidual korelasyonlarini guncelle: Dty_norm - G[:, support] @ x_norm
            residual_corr = Dty_norm - G_support_cols @ x_norm

            # Erken durdurma (artik norm^2 = ||y||^2 - x^T * rhs)
            r_energy_sq = max(y_energy_sq - float(np.dot(x_norm, rhs)), 0.0)
            if y_energy_sq > 0 and (r_energy_sq / y_energy_sq) < 1e-4:
                break

        # Normalize edilmis cozumu orijinal olcege cevir ve katsayilari yerlestir
        coefficients = np.zeros(n_dict)
        if support and x_norm is not None:
            for idx, s in enumerate(support):
                coefficients[s] = x_norm[idx] / norms_flat[s]
        
        # Frekanslari ve genlikleri cikar
        freq_amp_pairs = []
        seen_freqs = set()
        for s in support:
            freq_idx = s // 2
            f = candidate_freqs[freq_idx]
            if freq_idx in seen_freqs:
                continue
            seen_freqs.add(freq_idx)
            
            a_sin = coefficients[2 * freq_idx]
            a_cos = coefficients[2 * freq_idx + 1]
            amp = np.sqrt(a_sin ** 2 + a_cos ** 2)
            phase = np.arctan2(a_cos, a_sin)
            freq_amp_pairs.append((f, amp, phase, a_sin, a_cos))
        
        # Genlige gore sirala
        freq_amp_pairs.sort(key=lambda x: x[1], reverse=True)
        freq_amp_pairs = freq_amp_pairs[:n_atoms]
        
        frequency_atoms = np.array([p[0] for p in freq_amp_pairs])
        
        # Hedef hizda (1600Hz) sinyal sentezle
        if target_n_samples is None:
            target_n_samples = int(total_duration * self.target_rate)
        
        t_out = np.linspace(0, total_duration, target_n_samples, endpoint=False)
        reconstructed = np.zeros(target_n_samples)
        
        for f, amp, ph, a_sin, a_cos in freq_amp_pairs:
            reconstructed += a_sin * np.sin(2 * np.pi * f * t_out) + \
                             a_cos * np.cos(2 * np.pi * f * t_out)
        
        # SNR
        signal_on_samples = np.zeros(N)
        for f, amp, ph, a_sin, a_cos in freq_amp_pairs:
            signal_on_samples += a_sin * np.sin(2 * np.pi * f * t_samples) + \
                                 a_cos * np.cos(2 * np.pi * f * t_samples)
        
        residual_final = y - signal_on_samples
        snr = 10 * np.log10(np.var(signal_on_samples) / (np.var(residual_final) + 1e-10))
        
        print(f"[SRR-OMP] {len(freq_amp_pairs)} frekans geri kazanildi, SNR: {snr:.1f} dB")
        for i, (f, a, _, _, _) in enumerate(freq_amp_pairs[:5]):
            a_db = 20 * np.log10(a + 1e-10)
            print(f"  [{i+1}] {f:.1f} Hz -> {a_db:.1f} dB (amp={a:.4f})")
        
        omp_details = {
            "n_atoms_found": len(freq_amp_pairs),
            "freq_amp_pairs": freq_amp_pairs,
            "snr_db": float(snr),
            "residual_energy": float(np.linalg.norm(residual_final)),
            "signal_energy": float(np.linalg.norm(y)),
            "total_duration_s": total_duration,
            "target_samples": target_n_samples
        }
        
        return t_out, reconstructed, frequency_atoms, omp_details

    # =============================================================
    # PHASE 4: SPI GURULTU GIDERIMI
    # =============================================================

    def spi_noise_cancellation(
        self,
        active_signal: np.ndarray,
        baseline_signal: np.ndarray,
        sample_rate: float = None,
        spectral_floor_db: float = -60.0,
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Straight Path Interference (SPI) Giderimi.
        
        SPI Modeli:
            Y_clean(f) = Y_measured(f) - alpha * Y_baseline(f)
            |Y_clean(f)| = max(|Y_measured(f)| - alpha * |Y_baseline(f)|, floor)
            angle(Y_clean(f)) = angle(Y_measured(f))
        
        Args:
            active_signal: Haptik motor aktifken alinan sinyal (N,)
            baseline_signal: Motor kapaliyken alinan referans sinyal (M,)
            sample_rate: Ornekleme hizi
            spectral_floor_db: Minimum spektral zemin (dB)
            alpha: Cikartma katsayisi
            
        Returns:
            clean_signal: Temizlenmis sinyal (N,)
            spi_details: Detaylar
        """
        if sample_rate is None:
            sample_rate = float(self.target_rate)
        
        N = len(active_signal)
        
        # Baseline uzunluk ayarlama
        if len(baseline_signal) < N:
            reps = int(np.ceil(N / len(baseline_signal)))
            bl = np.tile(baseline_signal, reps)[:N]
        elif len(baseline_signal) > N:
            bl = baseline_signal[:N]
        else:
            bl = baseline_signal.copy()
        
        # FFT
        Y_active = rfft(active_signal)
        Y_baseline = rfft(bl)
        freqs = rfftfreq(N, d=1.0 / sample_rate)
        
        mag_active = np.abs(Y_active)
        mag_baseline = np.abs(Y_baseline)
        phase_active = np.angle(Y_active)
        
        # Spektral zemin
        spectral_floor = 10 ** (spectral_floor_db / 20.0) * np.max(mag_active)
        
        # SPI cikartma
        mag_clean = np.maximum(mag_active - alpha * mag_baseline, spectral_floor)
        Y_clean = mag_clean * np.exp(1j * phase_active)
        clean_signal = irfft(Y_clean, n=N)
        
        # Metrikler
        power_before = np.mean(active_signal ** 2)
        power_after = np.mean(clean_signal ** 2)
        noise_reduction = 10 * np.log10((power_before + 1e-10) / (power_after + 1e-10))
        
        print(f"[SRR-SPI] Gurultu azaltma: {noise_reduction:.1f} dB")
        
        spi_details = {
            "noise_reduction_db": float(noise_reduction),
            "alpha": alpha,
            "spectral_floor_db": spectral_floor_db
        }
        
        return clean_signal, spi_details

    # =============================================================
    # TAM PIPELINE: reconstruct()
    # =============================================================

    def reconstruct(
        self,
        imu_data: np.ndarray,
        timestamps_ns: np.ndarray,
        baseline_signal: np.ndarray = None,
        n_omp_atoms: int = 8,
        omp_freq_range: Tuple[float, float] = (50.0, 800.0)
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Tam SRR rekonstruksiyon pipeline'i.
        
        100Hz IMU verisinden 1600Hz sanal sinyal uretir.
        NUFFT-OMP yaklasimi ile f2 dahil tum frekanslari geri kazanir.
        
        Args:
            imu_data: Ham IMU verisi (N,) veya (N,3)
            timestamps_ns: Nanosaniye zaman damgalari (N,)
            baseline_signal: Bosta profil sinyali (opsiyonel)
            n_omp_atoms: OMP atom sayisi
            omp_freq_range: OMP frekans arama araligi
            
        Returns:
            time_axis: Zaman ekseni (saniye)
            reconstructed: 1600Hz rekonstrukte sinyal (M,)
            details: Tum ara sonuclar
        """
        print("\n" + "=" * 60)
        print("  SRR REKONSTRUKSIYON PIPELINE (Vi-Liquid)")
        print(f"  Giris: {self.native_rate}Hz, {len(imu_data)} ornek")
        print(f"  Hedef: {self.target_rate}Hz ({self.upsample_factor}x)")
        print(f"  LRA: {self.lra_frequency}Hz")
        print("=" * 60)
        
        details = {}
        
        # Cok boyutlu veriyi tek boyuta indir
        if imu_data.ndim == 2:
            if imu_data.shape[1] == 3:
                data_1d = imu_data[:, 2]  # z-ekseni (yuzey dik)
            else:
                data_1d = imu_data[:, 0]
        else:
            data_1d = imu_data.copy()
        
        # 1) Faz ofset orneklemesi
        print("\n[1/4] Faz Ofset Orneklemesi...")
        phases, cycle_indices, _ = self.phase_offset_sampling(timestamps_ns, data_1d)
        details["n_cycles"] = int(cycle_indices[-1]) + 1 if len(cycle_indices) > 0 else 0
        
        # 2) NUFFT-OMP ile f2 dahil tum frekans geri kazanimi
        print("\n[2/4] NUFFT-OMP Seyrek Geri Kazanim...")
        time_axis, reconstructed, frequency_atoms, omp_details = self.nufft_omp_recovery(
            data_1d, timestamps_ns,
            n_atoms=n_omp_atoms,
            freq_range=omp_freq_range,
            dictionary_size=512
        )
        details["omp"] = omp_details
        details["frequency_atoms"] = frequency_atoms
        print(f"[SRR] Cikis: {len(reconstructed)} ornek @ {self.target_rate}Hz")
        
        # 3) SPI gurultu giderimi
        if baseline_signal is not None:
            print("\n[3/4] SPI Gurultu Giderimi...")
            # Baseline'i hedef hiza yukselt (basit tekrar)
            bl_upsampled = np.repeat(baseline_signal, self.upsample_factor)
            if len(bl_upsampled) > len(reconstructed):
                bl_upsampled = bl_upsampled[:len(reconstructed)]
            elif len(bl_upsampled) < len(reconstructed):
                bl_upsampled = np.tile(bl_upsampled,
                    int(np.ceil(len(reconstructed) / len(bl_upsampled)))
                )[:len(reconstructed)]
            
            reconstructed, spi_details = self.spi_noise_cancellation(
                reconstructed, bl_upsampled,
                sample_rate=float(self.target_rate)
            )
            details["spi"] = spi_details
        else:
            print("\n[3/4] SPI Atlandi (baseline yok)")
        
        # 4) f2 cikarimi (OMP + FFT hibrit yaklasim)
        print("\n[4/4] Dominant Frekans (f2) Cikarimi...")
        
        # OMP'den gelen frekans atomlarini fiziksel aralikla filtrele
        # Karpuz rezonansi: 50-300Hz (fiziksel sinir)
        f2_physical_range = (50.0, 300.0)
        omp_pairs = omp_details.get("freq_amp_pairs", [])
        
        f2_omp = 0.0
        f2_omp_db = -np.inf
        for freq, amp, _, _, _ in omp_pairs:
            if f2_physical_range[0] <= freq <= f2_physical_range[1]:
                f2_omp = freq
                f2_omp_db = 20.0 * np.log10(amp + 1e-10)
                break  # Genlige gore siralanmis, ilk uygun
        
        # FFT tabanli f2 (dogrulama)
        f2_fft, f2_fft_db, fft_details = self.extract_dominant_f2(
            reconstructed, float(self.target_rate)
        )
        
        # Hibrit karar: OMP fiziksel aralikta bulduysa onu kullan
        if f2_omp > 0:
            f2 = f2_omp
            f2_db = f2_omp_db
            print(f"  [OMP] f2 = {f2:.1f} Hz (fiziksel aralikta, genlik: {f2_db:.1f} dB)")
        else:
            f2 = f2_fft
            f2_db = f2_fft_db
            print(f"  [FFT] f2 = {f2:.1f} Hz (OMP aralikta bulamadi)")
        
        if f2_fft != f2:
            print(f"  [FFT dogrulama] FFT f2 = {f2_fft:.1f} Hz ({f2_fft_db:.1f} dB)")
        
        details["f2"] = f2
        details["f2_db"] = f2_db
        details["f2_omp"] = f2_omp
        details["f2_fft"] = f2_fft
        details["fft"] = fft_details
        
        print(f"\n{'='*60}")
        print(f"  SONUC: f2 = {f2:.1f} Hz, Genlik = {f2_db:.1f} dB")
        print(f"  Cikis: {self.target_rate}Hz, {len(reconstructed)} ornek")
        print(f"{'='*60}")
        
        return time_axis, reconstructed, details

    # =============================================================
    # f2 DOMINANT FREKANS CIKARIMI
    # =============================================================

    def extract_dominant_f2(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        freq_range: Tuple[float, float] = (50.0, 500.0)
    ) -> Tuple[float, float, Dict]:
        """
        Rekonstrukte edilmis sinyalden dominant rezonans frekansi (f2) cikarir.
        
        f2, Elasticity Index (EI = f2^2 * m^(2/3)) formulunde kullanilir.
        
        Fiziksel sinirlar:
          - f2 < 150Hz ve magnitude > 25dB -> olgun karpuz
        """
        N = len(signal_data)
        if N < 4:
            return 0.0, -np.inf, {}
        
        window = np.hanning(N)
        windowed = signal_data * window
        
        fft_vals = rfft(windowed)
        freqs = rfftfreq(N, d=1.0 / sample_rate)
        amplitudes = np.abs(fft_vals) * 2.0 / N
        
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        band_freqs = freqs[freq_mask]
        band_amps = amplitudes[freq_mask]
        
        if len(band_amps) == 0:
            return 0.0, -np.inf, {"freqs": freqs, "amplitudes": amplitudes}
        
        peak_idx = np.argmax(band_amps)
        f2 = float(band_freqs[peak_idx])
        f2_amplitude = float(band_amps[peak_idx])
        f2_db = 20.0 * np.log10(f2_amplitude + 1e-10)
        
        # Harmonik analiz
        harmonics = {}
        for h in [2, 3, 4]:
            h_freq = f2 * h
            if h_freq <= freq_range[1]:
                h_mask = np.abs(band_freqs - h_freq) < (sample_rate / N * 2)
                if np.any(h_mask):
                    h_amp = float(np.max(band_amps[h_mask]))
                    harmonics[f"f2x{h}"] = {
                        "freq": float(h_freq),
                        "amplitude_db": 20.0 * np.log10(h_amp + 1e-10)
                    }
        
        fft_details = {
            "freqs": freqs,
            "amplitudes": amplitudes,
            "f2": f2,
            "f2_db": f2_db,
            "harmonics": harmonics,
            "freq_resolution": float(sample_rate / N)
        }
        
        return f2, f2_db, fft_details

    # =============================================================
    # CHIRP SWEEP + SRR: f2 DOGAL FREKANS TESPITI
    # =============================================================

    def chirp_sweep_f2_detection(
        self,
        sweep_data: List[Dict],
        freq_range: Tuple[float, float] = (80.0, 400.0)
    ) -> Tuple[float, float, Dict]:
        """
        Chirp Sweep + SRR ile dogal frekans (f2) tespiti.
        
        Vi-Liquid Yaklasimi:
          1. LRA'yi f_start -> f_end arasi adim adim surur
          2. Her adimda SRR ile yanitın genligini olcer
          3. Transfer fonksiyonu H(f) = |yanit(f)| / |uyarim(f)| olusturur
          4. H(f)'nin tepesi = dogal frekans f2
        
        Bu yontem, 100Hz IMU limitine ragmen 50-500Hz araliginda
        dogru f2 tespiti saglar. Cunku her adimda SRR, bilinen
        LRA frekansindaki yaniti geri kazanir (super-Nyquist degil,
        sadece bilinen frekanstaki genlik olcumu).
        
        Args:
            sweep_data: Her frekans adimi icin olcum verileri:
                [{
                    'freq_hz': float,     # LRA frekansi
                    'imu_data': ndarray,  # IMU verisi (N,)
                    'timestamps_ns': ndarray,  # Zaman damgalari (N,)
                    'excitation_amp': float  # Uyarim genligi
                }, ...]
            freq_range: f2 arama araligi (Hz)
            
        Returns:
            f2: Dogal frekans (Hz)
            f2_magnitude: f2'deki yanitın genligi
            sweep_result: Tam tarama sonuclari
        """
        transfer_function = []
        
        for step in sweep_data:
            f_step = step['freq_hz']
            imu = step['imu_data']
            ts_ns = step['timestamps_ns']
            exc_amp = step.get('excitation_amp', 1.0)
            
            # Bu frekansta SRR periyot katlama
            step_period_ns = int(1e9 / f_step)
            phases = ((ts_ns - ts_ns[0]) % step_period_ns).astype(np.float64) / step_period_ns
            
            _, folded = self.period_folding(
                imu.astype(np.float64), phases,
                n_bins=self.upsample_factor,
                outlier_rejection=True
            )
            
            # Yanitın genligi (RMS)
            response_amp = np.sqrt(np.mean(folded ** 2))
            
            # Transfer fonksiyonu: H(f) = yanitın genligi / uyarim genligi
            h_f = response_amp / (exc_amp + 1e-10)
            h_f_db = 20.0 * np.log10(h_f + 1e-10)
            
            transfer_function.append({
                'freq_hz': f_step,
                'response_amp': float(response_amp),
                'transfer_h': float(h_f),
                'transfer_h_db': float(h_f_db)
            })
        
        # f2 = transfer fonksiyonunun tepesi (fiziksel aralikta)
        valid_tf = [
            tf for tf in transfer_function
            if freq_range[0] <= tf['freq_hz'] <= freq_range[1]
        ]
        
        if not valid_tf:
            return 0.0, -np.inf, {"transfer_function": transfer_function}
        
        best = max(valid_tf, key=lambda x: x['transfer_h'])
        f2 = best['freq_hz']
        f2_magnitude = best['transfer_h_db']
        
        # Parabolik interpolasyon ile f2'yi ince-ayar
        freqs_arr = np.array([tf['freq_hz'] for tf in valid_tf])
        h_arr = np.array([tf['transfer_h'] for tf in valid_tf])
        peak_idx = np.argmax(h_arr)
        
        if 0 < peak_idx < len(h_arr) - 1:
            # 3-noktali parabolik tepe interpolasyonu
            f0, f1, f2_ = freqs_arr[peak_idx - 1], freqs_arr[peak_idx], freqs_arr[peak_idx + 1]
            h0, h1, h2_ = h_arr[peak_idx - 1], h_arr[peak_idx], h_arr[peak_idx + 1]
            
            denom = 2 * (2 * h1 - h0 - h2_)
            if abs(denom) > 1e-10:
                f2_refined = f1 + (f2_ - f0) * (h0 - h2_) / denom
                if freq_range[0] <= f2_refined <= freq_range[1]:
                    f2 = float(f2_refined)
        
        print(f"[SRR-Sweep] f2 = {f2:.1f} Hz (H={f2_magnitude:.1f} dB)")
        print(f"[SRR-Sweep] {len(transfer_function)} frekans adimi taranmis")
        
        sweep_result = {
            "transfer_function": transfer_function,
            "f2": f2,
            "f2_magnitude_db": f2_magnitude,
            "n_steps": len(transfer_function),
            "freq_range": freq_range
        }
        
        return f2, f2_magnitude, sweep_result

    def simulate_chirp_sweep(
        self,
        true_f2: float = 120.0,
        freq_start: float = 80.0,
        freq_end: float = 400.0,
        n_steps: int = 32,
        step_duration: float = 0.3,
        damping: float = 0.08,
        noise_level: float = 0.02
    ) -> List[Dict]:
        """
        Chirp sweep test verisi simulator.
        
        Bilinen dogal frekansta (true_f2) rezonans tepesi olan
        bir karpuz yanitini simule eder.
        
        Transfer fonksiyonu modeli (2. derece sönümlü sistem):
            H(f) = 1 / sqrt((1-(f/f2)^2)^2 + (2*zeta*f/f2)^2)
        
        Args:
            true_f2: Gercek dogal frekans (Hz)
            freq_start: Tarama baslangici
            freq_end: Tarama bitisi
            n_steps: Frekans adim sayisi
            step_duration: Her adim suresi (saniye)
            damping: Sonum orani (zeta)
            noise_level: Gurultu seviyesi
            
        Returns:
            sweep_data: Chirp sweep + SRR icin hazir veri
        """
        sweep_data = []
        sweep_freqs = np.linspace(freq_start, freq_end, n_steps)
        
        for f_step in sweep_freqs:
            # 2. derece sönümlü sistem transfer fonksiyonu
            r = f_step / true_f2
            h_mag = 1.0 / np.sqrt((1 - r**2)**2 + (2 * damping * r)**2)
            
            # IMU ornekleri uret
            n_samples = int(self.native_rate * step_duration)
            t_imu = np.arange(n_samples) / self.native_rate
            
            # Jitter
            jitter_ns = np.random.randint(-3000000, 3000000, n_samples)
            ts_ns = (t_imu * 1e9).astype(np.int64) + jitter_ns
            ts_ns = np.sort(np.maximum(ts_ns, 0))
            t_actual = ts_ns / 1e9
            
            # Yanitın sinyali (uyarim frekansinda, genlik H(f))
            response = h_mag * np.sin(2 * np.pi * f_step * t_actual)
            response += noise_level * np.random.randn(n_samples)
            
            sweep_data.append({
                'freq_hz': float(f_step),
                'imu_data': response,
                'timestamps_ns': ts_ns,
                'excitation_amp': 1.0
            })
        
        return sweep_data

    # =============================================================
    # BANT GECIREN FILTRE
    # =============================================================

    def apply_bandpass_filter(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        low_freq: float = 50.0,
        high_freq: float = 500.0,
        order: int = 4
    ) -> np.ndarray:
        """Butterworth bant geciren filtre."""
        nyq = sample_rate / 2.0
        low = low_freq / nyq
        high = min(high_freq / nyq, 0.99)
        
        if low >= high or low <= 0:
            return signal_data
        
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        
        if len(signal_data) > 3 * max(len(b), len(a)):
            return scipy_signal.filtfilt(b, a, signal_data)
        else:
            return scipy_signal.lfilter(b, a, signal_data)


# =================================================================
# YARDIMCI FONKSIYONLAR
# =================================================================

def extract_fft_amplitude(
    signal_data: np.ndarray,
    sample_rate: float,
    freq_range: Tuple[float, float] = (50.0, 500.0)
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """FFT genlik spektrumu + dominant frekans."""
    N = len(signal_data)
    if N < 2:
        return np.array([]), np.array([]), 0.0, -np.inf
    
    windowed = signal_data * np.hanning(N)
    fft_vals = rfft(windowed)
    frequencies = rfftfreq(N, d=1.0 / sample_rate)
    amplitudes = np.abs(fft_vals) * 2.0 / N
    
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    band_freqs = frequencies[freq_mask]
    band_amps = amplitudes[freq_mask]
    
    if len(band_amps) == 0:
        return frequencies, amplitudes, 0.0, -np.inf
    
    peak_idx = np.argmax(band_amps)
    dominant_freq = float(band_freqs[peak_idx])
    dominant_amplitude = float(band_amps[peak_idx])
    dominant_magnitude_db = 20.0 * np.log10(dominant_amplitude + 1e-10)
    
    return frequencies, amplitudes, dominant_freq, dominant_magnitude_db


def baseline_interference_cancellation(
    active_signal: np.ndarray,
    idle_profile: np.ndarray,
    alpha: float = 1.0
) -> np.ndarray:
    """SPI tabanli girisim iptali (geriye uyumlu)."""
    srr = SRRReconstructor()
    clean, _ = srr.spi_noise_cancellation(
        active_signal, idle_profile,
        sample_rate=float(IMU_NATIVE_RATE),
        alpha=alpha
    )
    return clean


# =================================================================
# SRR SIMULATORU (Test & Gelistirme)
# =================================================================

class SRRSimulator:
    """
    SRR algoritmasini test etmek icin gercekci sentetik veri uretici.
    """
    
    @staticmethod
    def generate_test_data(
        true_frequency: float = 120.0,
        lra_frequency: float = 167.0,
        imu_rate: float = 100.0,
        duration: float = 2.0,
        noise_level: float = 0.05,
        harmonics: list = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SRR test verisi uretir.
        
        Bilinen frekanslarda sinyal olusturur ve 100Hz'de ornekler.
        Gercekci timestamp jitter ekler.
        
        Args:
            true_frequency: Gercek rezonans frekansi (Hz)
            lra_frequency: LRA motor frekansi (Hz)
            imu_rate: IMU ornekleme hizi (Hz)
            duration: Sure (saniye)
            noise_level: Gurultu seviyesi (0-1)
            harmonics: Ek harmonikler [(freq, amp), ...]
            
        Returns:
            imu_data: Orneklenmis IMU verisi (N,)
            timestamps_ns: Nanosaniye zaman damgalari (N,)
            true_signal: Gercek yuksek cozunurluklu sinyal (referans)
        """
        target_rate = 1600.0
        n_target = int(target_rate * duration)
        t_target = np.arange(n_target) / target_rate
        
        # Gercek sinyal: hafif sonumlu sinuzoidal (karpuz vurus yaniti)
        # Sonum yavastir ki OMP yeterli veri gorebilsin
        decay_rate = 0.5  # Yavas sonum
        envelope = np.exp(-decay_rate * t_target)
        true_signal = 0.8 * np.sin(2 * np.pi * true_frequency * t_target) * envelope
        
        if harmonics:
            for h_freq, h_amp in harmonics:
                true_signal += h_amp * np.sin(2 * np.pi * h_freq * t_target) * envelope
        
        # 100Hz'de ornekle
        n_imu = int(imu_rate * duration)
        t_imu = np.arange(n_imu) / imu_rate
        
        # Buyuk jitter (+/- 3ms) -> non-uniform ornekleme
        # Vi-Liquid: gercek cihazlarda jitter ~2-4ms olabilir
        jitter_ns = np.random.randint(-3000000, 3000000, n_imu)
        timestamps_ns = (t_imu * 1e9).astype(np.int64) + jitter_ns
        timestamps_ns = np.sort(timestamps_ns)  # Siralama garanti
        # Negatif zaman damgasi onleme
        timestamps_ns = np.maximum(timestamps_ns, 0)
        t_actual = timestamps_ns / 1e9
        
        # IMU ornekleme (gercek zamanlarla)
        env_imu = np.exp(-decay_rate * t_actual)
        imu_data = 0.8 * np.sin(2 * np.pi * true_frequency * t_actual) * env_imu
        if harmonics:
            for h_freq, h_amp in harmonics:
                imu_data += h_amp * np.sin(2 * np.pi * h_freq * t_actual) * env_imu
        imu_data += noise_level * np.random.randn(n_imu)
        
        return imu_data, timestamps_ns, true_signal
    
    @staticmethod
    def evaluate_reconstruction(
        reconstructed: np.ndarray,
        true_signal: np.ndarray,
        true_frequency: float,
        sample_rate: float = 1600.0
    ) -> Dict:
        """SRR rekonstruksiyon kalitesini degerlendirir."""
        srr = SRRReconstructor()
        f2, f2_db, _ = srr.extract_dominant_f2(reconstructed, sample_rate)
        
        freq_error = abs(f2 - true_frequency)
        freq_error_pct = (freq_error / true_frequency) * 100 if true_frequency > 0 else 0
        
        min_len = min(len(reconstructed), len(true_signal))
        r = reconstructed[:min_len]
        t = true_signal[:min_len]
        
        if np.std(r) > 0 and np.std(t) > 0:
            correlation = float(np.corrcoef(r, t)[0, 1])
        else:
            correlation = 0.0
        
        mse = float(np.mean((r - t) ** 2))
        snr = 10 * np.log10(np.var(t) / (mse + 1e-10))
        
        return {
            "detected_f2": f2,
            "true_f2": true_frequency,
            "freq_error_hz": freq_error,
            "freq_error_pct": freq_error_pct,
            "f2_magnitude_db": f2_db,
            "correlation": correlation,
            "mse": mse,
            "snr_db": snr
        }
