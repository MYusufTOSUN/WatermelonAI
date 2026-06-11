"""
MODULE_A: Goruntu Tabanli Hacim ve Kutle Tahmini
Ali Bulent Koc (2007) - Disk Metodu & Elipsoid Yaklasimi

Tek bir yuksek cozunurluklu goruntuden eksen-simetrik meyvenin
hacmini tahmin eder. Hesaplanan kutle (m), Elasticity Index
formulunde kullanilir: EI = f2^2 * m^(2/3)

=================================================================
ALGORITMA ADIMLARI:
=================================================================
1. IMAGE_PREPROCESSING:
   - RGB -> Grayscale donusumu
   - Adaptive thresholding (ikili maske)
   - Morfolojik islemler (opening/closing) ile gurultu temizleme

2. BOUNDARY_EXTRACTION:
   - Canny kenar algilama
   - findContours ile sinir cikarimi
   - En buyuk kontur secimi (karpuz)

3. AXIS_DETERMINATION:
   - fitEllipse ile ana/yan eksen bulma
   - Minimum bounding rectangle
   - Major axis (uzunluk) ve Minor axis (cap) hesabi

4. DISK_METHOD_INTEGRATION:
   - Ana eksen boyunca dilimler (genislik h)
   - Her dilimin yaricapi r_i = sinir mesafesi
   - V = SUM(pi * r_i^2 * h)  (silindirik diskler toplami)

5. ELLIPSOID_APPROXIMATION (Dogrulama):
   - V_ellipsoid = (4/3) * pi * a * b * c
   - Disk metodu sonucu ile karsilastirma

6. MASS_ESTIMATION:
   - M = V * rho  (rho = 0.98 g/cm^3 karpuz yogunlugu)
   - cm^3 -> Litre donusumu

7. SCALE_CALIBRATION:
   - Referans nesne (bozuk para, cetvel) veya piksel/cm orani
   - ARCore/ARKit uzaklik verisi ile kalibrasyon

Referanslar:
  - Koc, A.B. (2007). Determination of watermelon volume using
    ellipsoid approximation and image processing.
  - Hedef: Ortalama mutlak bagil hata (MARE) < %8
=================================================================
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import WATERMELON_DENSITY, PI


# =================================================================
# ANA SINIF: ImageVolumeEstimator
# =================================================================

class ImageVolumeEstimator:
    """
    Goruntu tabanli hacim ve kutle tahmin motoru.
    
    Tek bir goruntuden Disk Metodu ve Elipsoid Yaklasimi ile
    karpuzun hacmini ve kutlesini hesaplar.
    
    Kullanim:
        estimator = ImageVolumeEstimator(pixels_per_cm=10.0)
        result = estimator.estimate_from_image(image)
        mass_kg = result['mass_kg']
    """

    def __init__(
        self,
        density: float = WATERMELON_DENSITY,
        pixels_per_cm: float = None,
        reference_length_cm: float = None,
        reference_length_px: float = None,
        disk_slice_width: int = 2,
        morph_kernel_size: int = 15,
        adaptive_block_size: int = 51,
        adaptive_c: int = 5,
        canny_low: int = 30,
        canny_high: int = 100,
        min_contour_area_ratio: float = 0.01,
        depth_ratio: float = 0.85
    ):
        """
        Args:
            density: Karpuz yogunlugu (g/cm^3), varsayilan 0.98
            pixels_per_cm: Piksel/cm olcek orani (bilinen ise)
            reference_length_cm: Referans nesne gercek uzunlugu (cm)
            reference_length_px: Referans nesne piksel uzunlugu
            disk_slice_width: Disk metodu dilim genisligi (piksel)
            morph_kernel_size: Morfolojik cekirdek boyutu
            adaptive_block_size: Adaptive threshold blok boyutu (tek sayi)
            adaptive_c: Adaptive threshold sabiti
            canny_low: Canny alt esik
            canny_high: Canny ust esik
            min_contour_area_ratio: Minimum kontur alan orani
            depth_ratio: 2D'den derinlik tahmini icin oran (depth/width)
        """
        self.density = density
        self.disk_slice_width = disk_slice_width
        self.morph_kernel_size = morph_kernel_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area_ratio = min_contour_area_ratio
        self.depth_ratio = depth_ratio
        
        # Olcek kalibrasyonu
        if pixels_per_cm is not None:
            self.pixels_per_cm = pixels_per_cm
        elif reference_length_cm and reference_length_px:
            self.pixels_per_cm = reference_length_px / reference_length_cm
        else:
            self.pixels_per_cm = None  # Kalibrasyon gerekmeden calisir

    # =============================================================
    # ADIM 1: GORUNTU ON-ISLEME
    # =============================================================

    def preprocess_image(
        self,
        image: np.ndarray,
        use_hsv: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RGB goruntuyu ikili maskeye donusturur.
        
        Adimlar:
          1. RGB -> Grayscale (veya HSV bazli segmentasyon)
          2. Gaussian bulaniklastirma (gurultu azaltma)
          3. Adaptive thresholding (Otsu alternatifi)
          4. Morfolojik acma/kapama (gurultu temizleme)
        
        Args:
            image: BGR/RGB goruntu (H, W, 3)
            use_hsv: HSV renk uzayinda segmentasyon kullan
            
        Returns:
            gray: Gri tonlamali goruntu
            binary_mask: Temiz ikili maske
            preprocessed: On-islenmis goruntu (debug icin)
        """
        if image is None or image.size == 0:
            raise ValueError("Gecersiz goruntu!")
        
        # BGR kontrolu ve donusum
        if len(image.shape) == 2:
            gray = image.copy()
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian bulaniklastirma (gurultu azaltma)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        if use_hsv and len(image.shape) == 3 and image.shape[2] >= 3:
            # HSV bazli segmentasyon (karpuz yesil/koyu renk)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Genis renk araligi (karpuz yesil + koyu yesilden sariya)
            lower_green = np.array([20, 30, 30])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Koyu bolgeler (karpuz koyu seritler)
            lower_dark = np.array([0, 0, 30])
            upper_dark = np.array([180, 255, 120])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Birlestir
            binary_mask = cv2.bitwise_or(mask_green, mask_dark)
        else:
            # Adaptive thresholding (genel amacli)
            block_size = self.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
            
            binary_mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.adaptive_c
            )
        
        # Otsu thresholding (ek dogrulama)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Maskeleri birlestir (OR)
        binary_mask = cv2.bitwise_or(binary_mask, otsu_mask)
        
        # Morfolojik islemler
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Acma (opening): kucuk gurultu noktalarini sil
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Kapama (closing): ic bosluklari doldur
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Bosluk doldurma (flood fill)
        binary_mask = self._fill_holes(binary_mask)
        
        return gray, binary_mask, blurred

    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Ikili maskede ic bosluklari doldurur (flood fill)."""
        h, w = binary_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        filled = binary_mask.copy()
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        
        filled_inv = cv2.bitwise_not(filled)
        result = cv2.bitwise_or(binary_mask, filled_inv)
        return result

    # =============================================================
    # ADIM 2: SINIR CIKARIMI
    # =============================================================

    def extract_boundary(
        self,
        binary_mask: np.ndarray,
        gray: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Karpuz sinirrini cikarir.
        
        Adimlar:
          1. findContours ile tum konturlari bul
          2. En buyuk konturu sec (karpuz)
          3. convexHull ile dis siniri duzelt
          4. Canny kenar haritasi (opsiyonel dogrulama)
        
        Args:
            binary_mask: Ikili maske
            gray: Gri tonlamali goruntu (Canny icin)
            
        Returns:
            contour: Karpuz konturu (N, 1, 2)
            hull: Convex hull
            boundary_info: Sinir bilgileri
        """
        # Tum konturlari bul
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            raise ValueError("Goruntude kontur bulunamadi! Segmentasyon basarisiz.")
        
        # Minimum alan filtresi
        img_area = binary_mask.shape[0] * binary_mask.shape[1]
        min_area = img_area * self.min_contour_area_ratio
        
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if not valid_contours:
            raise ValueError(f"Yeterli buyuklukte kontur bulunamadi (min: {min_area:.0f} px^2)")
        
        # En buyuk konturu sec
        contour = max(valid_contours, key=cv2.contourArea)
        
        # Convex hull (dis sinir duzeltmesi)
        hull = cv2.convexHull(contour)
        
        # Sinir bilgileri
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * PI * area) / (perimeter ** 2) if perimeter > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Canny kenar haritasi (opsiyonel)
        edge_map = None
        if gray is not None:
            edge_map = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        boundary_info = {
            "contour_area_px": float(area),
            "hull_area_px": float(hull_area),
            "perimeter_px": float(perimeter),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "n_contour_points": len(contour),
            "edge_map": edge_map
        }
        
        return contour, hull, boundary_info

    # =============================================================
    # ADIM 3: EKSEN BELIRLEME
    # =============================================================

    def determine_axes(
        self,
        contour: np.ndarray,
        hull: np.ndarray
    ) -> Dict:
        """
        Ana ve yan eksenleri belirler.
        
        Yontemler:
          1. fitEllipse: En uygun elips (direkt eksen verir)
          2. minAreaRect: Minimum cevreleyen dikdortgen
          3. PCA: Temel bilesen analizi (eksen yonu)
        
        Args:
            contour: Karpuz konturu
            hull: Convex hull
            
        Returns:
            Eksen bilgileri (piksel cinsinden)
        """
        if len(contour) < 5:
            raise ValueError("fitEllipse icin en az 5 nokta gerekli!")
        
        # 1) fitEllipse
        (cx, cy), (minor_d, major_d), angle = cv2.fitEllipse(contour)
        
        # Buyuk eksen her zaman major olmali
        if minor_d > major_d:
            minor_d, major_d = major_d, minor_d
            angle = (angle + 90) % 180
        
        major_axis_px = major_d  # Buyuk eksen (uzunluk)
        minor_axis_px = minor_d  # Kucuk eksen (cap)
        
        # 2) Minimum cevreleyen dikdortgen
        rect = cv2.minAreaRect(contour)
        rect_center, (rect_w, rect_h), rect_angle = rect
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
            rect_angle = (rect_angle + 90) % 180
        
        # 3) PCA ile eksen yonu (kontur noktalarindan)
        pts = contour.reshape(-1, 2).astype(np.float64)
        mean = np.mean(pts, axis=0)
        pts_centered = pts - mean
        
        cov_matrix = np.cov(pts_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # En buyuk eigenvalue -> ana eksen yonu
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        pca_major_direction = eigenvectors[:, 0]  # Ana eksen yonu
        pca_angle = np.degrees(np.arctan2(pca_major_direction[1], pca_major_direction[0]))
        
        # Eliptisite (ne kadar yuvarlak)
        eccentricity = np.sqrt(1 - (minor_axis_px / major_axis_px) ** 2) if major_axis_px > 0 else 0
        aspect_ratio = minor_axis_px / major_axis_px if major_axis_px > 0 else 1
        
        axes_info = {
            # Elips fit sonuclari
            "center_px": (float(cx), float(cy)),
            "major_axis_px": float(major_axis_px),
            "minor_axis_px": float(minor_axis_px),
            "major_radius_px": float(major_axis_px / 2),
            "minor_radius_px": float(minor_axis_px / 2),
            "ellipse_angle_deg": float(angle),
            
            # Dikdortgen fit
            "rect_width_px": float(rect_w),
            "rect_height_px": float(rect_h),
            "rect_angle_deg": float(rect_angle),
            
            # PCA
            "pca_angle_deg": float(pca_angle),
            "pca_eigenvalues": eigenvalues.tolist(),
            
            # Sekil metrikleri
            "eccentricity": float(eccentricity),
            "aspect_ratio": float(aspect_ratio),
            
            # CM cinsinden (kalibrasyon varsa)
            "major_axis_cm": float(major_axis_px / self.pixels_per_cm) if self.pixels_per_cm else None,
            "minor_axis_cm": float(minor_axis_px / self.pixels_per_cm) if self.pixels_per_cm else None,
        }
        
        return axes_info

    # =============================================================
    # ADIM 4: DISK METODU ENTEGRASYONU
    # =============================================================

    def disk_method_volume(
        self,
        contour: np.ndarray,
        axes_info: Dict,
        slice_width_px: int = None,
        use_smoothing: bool = True,
        smooth_window: int = 7
    ) -> Dict:
        """
        Disk Metodu ile hacim hesabi.
        
        Koc (2007) yaklasimi:
          1. Konturu ana eksen boyunca h genisliginde dilimlere bol
          2. Her dilimde, ana eksene olan mesafe = yaricap r_i
          3. Yaricap profilini yumustama (kontur gurultusu azaltma)
          4. Eliptik disk duzeltmesi (2D -> 3D)
          5. V = SUM(pi * r_visible * r_depth * h)
        
        Eksen-simetri varsayimi:
          - Karpuz ana eksen etrafinda donme simetrisine sahiptir
          - Her dilim bir eliptik disk olusturur
          - Gorunur yaricap: konturun ana eksene mesafesi
          - Derinlik yaricap: gorunur_yaricap * depth_ratio
        
        Args:
            contour: Karpuz konturu
            axes_info: Eksen bilgileri
            slice_width_px: Dilim genisligi (piksel)
            use_smoothing: Yaricap profili yumusatma
            smooth_window: Yumusatma pencere boyutu
            
        Returns:
            Disk metodu sonuclari (piksel^3 ve cm^3)
        """
        if slice_width_px is None:
            slice_width_px = self.disk_slice_width
        
        # Ana eksen bilgileri
        cx, cy = axes_info["center_px"]
        angle_deg = axes_info["ellipse_angle_deg"]
        major_r = axes_info["major_radius_px"]
        minor_r = axes_info["minor_radius_px"]
        
        # Ana eksen yonu (birim vektor)
        angle_rad = np.radians(angle_deg)
        axis_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        axis_perp = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
        
        center = np.array([cx, cy])
        
        # Kontur noktalarini ana eksen koordinat sistemine donustur
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts_relative = pts - center
        
        # Her noktanin ana eksen projeksiyonu ve dik mesafesi
        proj_along = pts_relative @ axis_dir
        proj_perp = np.abs(pts_relative @ axis_perp)
        
        # Ana eksen araligi - fit edilen elipsten al (daha stabil)
        proj_min = -major_r
        proj_max = major_r
        total_length_px = proj_max - proj_min
        
        # Dilimlere bol (1 piksel hassasiyet)
        n_slices = max(1, int(total_length_px / slice_width_px))
        slice_edges = np.linspace(proj_min, proj_max, n_slices + 1)
        h = total_length_px / n_slices
        
        # Her dilim icin yaricap hesapla
        radii_raw = []
        
        for i in range(n_slices):
            s_start = slice_edges[i]
            s_end = slice_edges[i + 1]
            
            mask = (proj_along >= s_start) & (proj_along < s_end)
            
            if np.sum(mask) > 0:
                # Median yaricap (outlier direncli)
                perp_vals = proj_perp[mask]
                # Ust %10 hariç ortanca (disarida kalan noktalar icin)
                upper_trim = np.percentile(perp_vals, 90)
                trimmed = perp_vals[perp_vals <= upper_trim]
                if len(trimmed) > 0:
                    r_i = np.max(trimmed)
                else:
                    r_i = np.median(perp_vals)
            else:
                # Bos dilim: fit edilmis elipsten interpole et
                # Elips denklemi: r(x) = b * sqrt(1 - x^2/a^2)
                x_center = (s_start + s_end) / 2
                x_norm = x_center / major_r if major_r > 0 else 0
                if abs(x_norm) < 1.0:
                    r_i = minor_r * np.sqrt(1 - x_norm ** 2)
                else:
                    r_i = 0.0
            
            radii_raw.append(r_i)
        
        radii_arr = np.array(radii_raw)
        
        # Yaricap profili yumusatma (gurultu azaltma)
        if use_smoothing and len(radii_arr) > smooth_window:
            # Hareketli ortalama
            kernel_size = min(smooth_window, len(radii_arr))
            if kernel_size % 2 == 0:
                kernel_size += 1
            pad = kernel_size // 2
            padded = np.pad(radii_arr, pad, mode='edge')
            smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')
            
            # Fit edilmis elips profili ile ust sinir kontrolu
            # Elips r(x) = b*sqrt(1 - x^2/a^2)
            slice_centers = (slice_edges[:-1] + slice_edges[1:]) / 2
            x_norm = slice_centers / major_r if major_r > 0 else np.zeros_like(slice_centers)
            x_norm = np.clip(x_norm, -0.999, 0.999)
            ellipse_profile = minor_r * np.sqrt(1 - x_norm ** 2)
            
            # Yumusatilmis yaricap, elips profilinin %105'ini asamaz
            max_allowed = ellipse_profile * 1.05
            radii_arr = np.minimum(smoothed[:n_slices], max_allowed)
        
        # Disk hacimleri hesapla
        slice_volumes_px3 = PI * radii_arr ** 2 * h
        
        # Toplam hacim (px^3) - dairesel kesit
        volume_px3_circular = float(np.sum(slice_volumes_px3))
        
        # 2D'den 3D eliptik disk duzeltmesi:
        # Gorunur yaricap r_visible, derinlik r_depth = r_visible * depth_ratio
        # Eliptik disk: V_i = pi * r_visible * r_depth * h
        # V_corrected = V_circular * depth_ratio
        volume_px3 = volume_px3_circular * self.depth_ratio
        
        # CM cinsinden
        volume_cm3 = None
        h_cm = None
        radii_cm = None
        
        if self.pixels_per_cm:
            scale = self.pixels_per_cm
            h_cm = h / scale
            radii_cm = radii_arr / scale
            volume_cm3 = float(self.depth_ratio * np.sum(PI * radii_cm ** 2 * h_cm))
        
        result = {
            "volume_px3": float(volume_px3),
            "volume_px3_circular": float(volume_px3_circular),
            "volume_cm3": volume_cm3,
            "volume_liters": volume_cm3 / 1000.0 if volume_cm3 else None,
            "n_slices": n_slices,
            "slice_width_px": float(h),
            "slice_width_cm": h_cm,
            "radii_px": radii_arr,
            "radii_cm": radii_cm,
            "total_length_px": float(total_length_px),
            "max_radius_px": float(np.max(radii_arr)) if len(radii_arr) > 0 else 0,
            "slice_volumes_px3": slice_volumes_px3,
            "depth_ratio_correction": self.depth_ratio,
            "smoothing_applied": use_smoothing,
        }
        
        return result

    # =============================================================
    # ADIM 5: ELIPSOID YAKLASIMI (Dogrulama)
    # =============================================================

    def ellipsoid_volume(
        self,
        axes_info: Dict
    ) -> Dict:
        """
        Elipsoid yaklasimi ile hacim hesabi.
        
        V = (4/3) * pi * a * b * c
        
        Burada:
          a = major_radius (buyuk yari-eksen)
          b = minor_radius (kucuk yari-eksen)
          c = depth_radius (derinlik yari-ekseni, depth_ratio ile tahmin)
        
        Args:
            axes_info: Eksen bilgileri
            
        Returns:
            Elipsoid hacim sonuclari
        """
        major_r_px = axes_info["major_radius_px"]
        minor_r_px = axes_info["minor_radius_px"]
        depth_r_px = minor_r_px * self.depth_ratio  # 2D'den derinlik tahmini
        
        # Piksel cinsinden hacim
        volume_px3 = (4.0 / 3.0) * PI * major_r_px * minor_r_px * depth_r_px
        
        # CM cinsinden
        volume_cm3 = None
        if self.pixels_per_cm:
            scale = self.pixels_per_cm
            a_cm = major_r_px / scale
            b_cm = minor_r_px / scale
            c_cm = depth_r_px / scale
            volume_cm3 = (4.0 / 3.0) * PI * a_cm * b_cm * c_cm
        
        result = {
            "volume_px3": float(volume_px3),
            "volume_cm3": volume_cm3,
            "volume_liters": volume_cm3 / 1000.0 if volume_cm3 else None,
            "semi_axes_px": {
                "a": float(major_r_px),
                "b": float(minor_r_px),
                "c": float(depth_r_px)
            },
            "depth_ratio": self.depth_ratio,
        }
        
        return result

    # =============================================================
    # ADIM 6: KUTLE TAHMINI
    # =============================================================

    def estimate_mass(
        self,
        volume_cm3: float
    ) -> Dict:
        """
        Hacimden kutle tahmini.
        
        M = V * rho
        rho = 0.98 g/cm^3 (karpuz yogunlugu ~ su)
        
        Args:
            volume_cm3: Hacim (cm^3)
            
        Returns:
            Kutle tahmini
        """
        mass_g = volume_cm3 * self.density
        mass_kg = mass_g / 1000.0
        
        return {
            "mass_g": float(mass_g),
            "mass_kg": float(mass_kg),
            "density_g_per_cm3": self.density,
            "volume_cm3": volume_cm3,
        }

    # =============================================================
    # ADIM 7: OLCEK KALIBRASYONU
    # =============================================================

    def calibrate_scale(
        self,
        reference_px: float,
        reference_cm: float
    ) -> float:
        """
        Referans nesne ile piksel/cm kalibrasyonu.
        
        Args:
            reference_px: Referans nesne piksel uzunlugu
            reference_cm: Referans nesne gercek uzunlugu (cm)
            
        Returns:
            pixels_per_cm orani
        """
        self.pixels_per_cm = reference_px / reference_cm
        return self.pixels_per_cm

    def calibrate_from_coin(
        self,
        image: np.ndarray,
        coin_diameter_cm: float = 2.63
    ) -> Optional[float]:
        """
        Goruntudeki dairesel referans nesneden (bozuk para) kalibrasyon.
        
        Args:
            image: Goruntu
            coin_diameter_cm: Bozuk para capi (cm) - TR 1TL = 2.63cm
            
        Returns:
            pixels_per_cm veya None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # HoughCircles ile daire bulma
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=50,
            minRadius=10,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # En kucuk daireyi sec (muhtemelen bozuk para)
            smallest = min(circles, key=lambda c: c[2])
            coin_radius_px = smallest[2]
            coin_diameter_px = coin_radius_px * 2
            
            self.pixels_per_cm = coin_diameter_px / coin_diameter_cm
            return self.pixels_per_cm
        
        return None

    # =============================================================
    # TAM PIPELINE: estimate_from_image()
    # =============================================================

    def estimate_from_image(
        self,
        image: np.ndarray,
        return_debug: bool = False
    ) -> Dict:
        """
        Tek bir goruntuden hacim ve kutle tahmini (tam pipeline).
        
        Args:
            image: BGR goruntu (H, W, 3)
            return_debug: Debug gorsellerini dondur
            
        Returns:
            Hacim, kutle ve tum ara sonuclar
        """
        print("\n" + "=" * 60)
        print("  GORUNTU TABANLI HACIM/KUTLE TAHMINI")
        print(f"  Koc (2007) Disk Metodu + Elipsoid Yaklasimi")
        print("=" * 60)
        
        # 1) On-isleme
        print("\n[1/6] Goruntu on-isleme...")
        gray, binary_mask, preprocessed = self.preprocess_image(image)
        h, w = gray.shape
        print(f"  Goruntu: {w}x{h} piksel")
        print(f"  Maske: {np.sum(binary_mask > 0)} beyaz piksel")
        
        # 2) Sinir cikarimi
        print("\n[2/6] Sinir cikarimi...")
        contour, hull, boundary_info = self.extract_boundary(binary_mask, gray)
        print(f"  Kontur: {boundary_info['n_contour_points']} nokta")
        print(f"  Dairesellik: {boundary_info['circularity']:.3f}")
        print(f"  Katiglik: {boundary_info['solidity']:.3f}")
        
        # 3) Eksen belirleme
        print("\n[3/6] Eksen belirleme...")
        axes_info = self.determine_axes(contour, hull)
        print(f"  Buyuk eksen: {axes_info['major_axis_px']:.1f} px")
        print(f"  Kucuk eksen: {axes_info['minor_axis_px']:.1f} px")
        print(f"  Eliptisite: {axes_info['eccentricity']:.3f}")
        print(f"  En-boy orani: {axes_info['aspect_ratio']:.3f}")
        
        if self.pixels_per_cm:
            print(f"  Buyuk eksen: {axes_info['major_axis_cm']:.1f} cm")
            print(f"  Kucuk eksen: {axes_info['minor_axis_cm']:.1f} cm")
            print(f"  Olcek: {self.pixels_per_cm:.1f} px/cm")
        
        # 4) Disk metodu
        print("\n[4/6] Disk metodu hacim hesabi...")
        disk_result = self.disk_method_volume(contour, axes_info)
        print(f"  Dilim sayisi: {disk_result['n_slices']}")
        print(f"  Dilim genisligi: {disk_result['slice_width_px']:.1f} px")
        print(f"  Hacim (piksel^3): {disk_result['volume_px3']:.0f}")
        
        if disk_result['volume_cm3']:
            print(f"  Hacim (cm^3): {disk_result['volume_cm3']:.0f}")
            print(f"  Hacim (litre): {disk_result['volume_liters']:.2f}")
        
        # 5) Elipsoid yaklasimi
        print("\n[5/6] Elipsoid yaklasimi (dogrulama)...")
        ellipsoid_result = self.ellipsoid_volume(axes_info)
        print(f"  Hacim (piksel^3): {ellipsoid_result['volume_px3']:.0f}")
        
        if ellipsoid_result['volume_cm3']:
            print(f"  Hacim (cm^3): {ellipsoid_result['volume_cm3']:.0f}")
            print(f"  Hacim (litre): {ellipsoid_result['volume_liters']:.2f}")
        
        # Disk vs Elipsoid karsilastirma
        if disk_result['volume_px3'] > 0 and ellipsoid_result['volume_px3'] > 0:
            ratio = disk_result['volume_px3'] / ellipsoid_result['volume_px3']
            diff_pct = abs(1 - ratio) * 100
            print(f"  Disk/Elipsoid orani: {ratio:.3f} (fark: %{diff_pct:.1f})")
        
        # En iyi hacim tahmini: Agirlikli ortalama
        # Elipsoid, 2D goruntude analitik olarak daha stabil -> agirlik 0.6
        # Disk metodu, kontur seklini yakalama avantaji -> agirlik 0.4
        w_disk = 0.4
        w_ellipsoid = 0.6
        
        if disk_result['volume_cm3'] and ellipsoid_result['volume_cm3']:
            best_volume_cm3 = (
                w_disk * disk_result['volume_cm3'] +
                w_ellipsoid * ellipsoid_result['volume_cm3']
            )
            print(f"  Agirlikli ortalama: %{w_disk*100:.0f} Disk + %{w_ellipsoid*100:.0f} Elipsoid")
        elif disk_result['volume_cm3']:
            best_volume_cm3 = disk_result['volume_cm3']
        elif ellipsoid_result['volume_cm3']:
            best_volume_cm3 = ellipsoid_result['volume_cm3']
        else:
            # Piksel bazli hacim - kalibrasyon yok uyarisi
            best_volume_cm3 = None
        
        # 6) Kutle tahmini
        print("\n[6/6] Kutle tahmini...")
        mass_result = None
        if best_volume_cm3:
            mass_result = self.estimate_mass(best_volume_cm3)
            print(f"  Ortalama hacim: {best_volume_cm3:.0f} cm^3")
            print(f"  Kutle: {mass_result['mass_g']:.0f} g ({mass_result['mass_kg']:.2f} kg)")
            print(f"  Yogunluk: {self.density} g/cm^3")
        else:
            print("  [!] Kalibrasyon yok - kutle hesaplanamadi!")
            print("  calibrate_scale() veya pixels_per_cm parametresi gerekli")
        
        # Sonuc
        result = {
            "boundary": boundary_info,
            "axes": axes_info,
            "disk_method": disk_result,
            "ellipsoid": ellipsoid_result,
            "mass": mass_result,
            "best_volume_cm3": best_volume_cm3,
            "best_volume_liters": best_volume_cm3 / 1000.0 if best_volume_cm3 else None,
            "mass_kg": mass_result['mass_kg'] if mass_result else None,
            "mass_g": mass_result['mass_g'] if mass_result else None,
            "pixels_per_cm": self.pixels_per_cm,
            "calibrated": self.pixels_per_cm is not None,
        }
        
        if return_debug:
            result["debug"] = {
                "gray": gray,
                "binary_mask": binary_mask,
                "contour": contour,
                "hull": hull,
                "preprocessed": preprocessed,
            }
        
        print(f"\n{'='*60}")
        if mass_result:
            print(f"  SONUC: V={best_volume_cm3:.0f} cm^3, M={mass_result['mass_kg']:.2f} kg")
        else:
            print(f"  SONUC: V={disk_result['volume_px3']:.0f} px^3 (kalibrasyon gerekli)")
        print(f"{'='*60}")
        
        return result

    # =============================================================
    # GORSEL CIKTI (Debug & Raporlama)
    # =============================================================

    def draw_analysis(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        axes_info: Dict,
        disk_result: Dict
    ) -> np.ndarray:
        """
        Analiz sonuclarini goruntu uzerine cizer.
        
        Args:
            image: Orijinal goruntu
            contour: Karpuz konturu
            axes_info: Eksen bilgileri
            disk_result: Disk metodu sonuclari
            
        Returns:
            Uzerine cizimlenmis goruntu
        """
        output = image.copy()
        
        # Kontur
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        
        # Elips fit
        cx, cy = axes_info["center_px"]
        major = axes_info["major_axis_px"]
        minor = axes_info["minor_axis_px"]
        angle = axes_info["ellipse_angle_deg"]
        
        cv2.ellipse(
            output,
            (int(cx), int(cy)),
            (int(major / 2), int(minor / 2)),
            angle, 0, 360,
            (255, 0, 0), 2
        )
        
        # Ana eksen cizgisi
        angle_rad = np.radians(angle)
        dx = np.cos(angle_rad) * major / 2
        dy = np.sin(angle_rad) * major / 2
        pt1 = (int(cx - dx), int(cy - dy))
        pt2 = (int(cx + dx), int(cy + dy))
        cv2.line(output, pt1, pt2, (0, 0, 255), 2)
        
        # Yan eksen cizgisi
        dx2 = np.cos(angle_rad + PI / 2) * minor / 2
        dy2 = np.sin(angle_rad + PI / 2) * minor / 2
        pt3 = (int(cx - dx2), int(cy - dy2))
        pt4 = (int(cx + dx2), int(cy + dy2))
        cv2.line(output, pt3, pt4, (255, 255, 0), 2)
        
        # Merkez
        cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # Disk yaricaplari (her 10. dilim)
        radii = disk_result.get("radii_px", np.array([]))
        n_slices = disk_result.get("n_slices", 0)
        if len(radii) > 0 and n_slices > 10:
            step = max(1, n_slices // 10)
            for i in range(0, n_slices, step):
                frac = (i / n_slices - 0.5) * 2  # -1 to 1
                sx = int(cx + frac * major / 2 * np.cos(angle_rad))
                sy = int(cy + frac * major / 2 * np.sin(angle_rad))
                r = int(radii[i])
                
                cv2.line(
                    output,
                    (int(sx - r * np.sin(angle_rad)), int(sy + r * np.cos(angle_rad))),
                    (int(sx + r * np.sin(angle_rad)), int(sy - r * np.cos(angle_rad))),
                    (0, 200, 200), 1
                )
        
        # Bilgi yazisi
        text_y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        if axes_info.get("major_axis_cm"):
            cv2.putText(output, f"L={axes_info['major_axis_cm']:.1f}cm D={axes_info['minor_axis_cm']:.1f}cm",
                        (10, text_y), font, font_scale, (255, 255, 255), 2)
        
        if disk_result.get("volume_cm3"):
            text_y += 25
            cv2.putText(output, f"V_disk={disk_result['volume_cm3']:.0f}cm3",
                        (10, text_y), font, font_scale, (255, 255, 255), 2)
        
        return output


# =================================================================
# SENTETIK GORUNTU URETICISI (Test & Demo)
# =================================================================

class SyntheticWatermelonGenerator:
    """
    Test icin sentetik karpuz goruntusu uretir.
    Bilinen boyutlarla hacimlerin dogrulanmasini saglar.
    """
    
    @staticmethod
    def generate(
        major_axis_px: int = 400,
        minor_axis_px: int = 300,
        image_size: Tuple[int, int] = (640, 480),
        angle: float = 0.0,
        noise_level: float = 10.0,
        add_stripes: bool = True,
        add_stem: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Sentetik karpuz goruntusu uretir.
        
        Args:
            major_axis_px: Buyuk eksen uzunlugu (piksel)
            minor_axis_px: Kucuk eksen uzunlugu (piksel)
            image_size: Goruntu boyutu (W, H)
            angle: Donme acisi (derece)
            noise_level: Gurultu seviyesi
            add_stripes: Karpuz seritleri ekle
            add_stem: Sap bolgesi ekle
            
        Returns:
            image: Sentetik goruntu (BGR)
            ground_truth: Bilinen gercek degerler
        """
        W, H = image_size
        image = np.full((H, W, 3), (200, 180, 160), dtype=np.uint8)  # Acik arka plan
        
        cx, cy = W // 2, H // 2
        
        # Karpuz govdesi (koyu yesil elips)
        cv2.ellipse(
            image,
            (cx, cy),
            (major_axis_px // 2, minor_axis_px // 2),
            angle,
            0, 360,
            (30, 100, 30),  # Koyu yesil
            -1  # Dolu
        )
        
        # Seritler (acik yesil)
        if add_stripes:
            n_stripes = 7
            for i in range(n_stripes):
                stripe_angle = angle + (i * 180.0 / n_stripes) - 90
                a_rad = np.radians(stripe_angle)
                
                # Serit pozisyonu (merkeze gore)
                offset = (i - n_stripes // 2) * (minor_axis_px // (n_stripes + 1))
                
                sx = int(cx + offset * np.cos(np.radians(angle + 90)))
                sy = int(cy + offset * np.sin(np.radians(angle + 90)))
                
                # Ince elips seklinde serit
                stripe_major = int(major_axis_px * 0.48)
                stripe_minor = max(3, int(minor_axis_px * 0.02))
                
                cv2.ellipse(
                    image,
                    (sx, sy),
                    (stripe_major, stripe_minor),
                    angle,
                    0, 360,
                    (50, 140, 50),  # Acik yesil
                    -1
                )
        
        # Sap bolgesi
        if add_stem:
            stem_end_x = int(cx - (major_axis_px // 2 + 15) * np.cos(np.radians(angle)))
            stem_end_y = int(cy - (major_axis_px // 2 + 15) * np.sin(np.radians(angle)))
            stem_start_x = int(cx - (major_axis_px // 2 - 5) * np.cos(np.radians(angle)))
            stem_start_y = int(cy - (major_axis_px // 2 - 5) * np.sin(np.radians(angle)))
            cv2.line(image, (stem_start_x, stem_start_y), (stem_end_x, stem_end_y), (20, 80, 20), 4)
        
        # Gurultu ekle
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Ground truth
        a = major_axis_px / 2.0
        b = minor_axis_px / 2.0
        
        # Elips alani = pi * a * b
        area_px2 = PI * a * b
        
        ground_truth = {
            "center_px": (cx, cy),
            "major_axis_px": major_axis_px,
            "minor_axis_px": minor_axis_px,
            "major_radius_px": a,
            "minor_radius_px": b,
            "angle_deg": angle,
            "area_px2": area_px2,
        }
        
        return image, ground_truth
    
    @staticmethod
    def generate_with_known_volume(
        length_cm: float = 35.0,
        diameter_cm: float = 28.0,
        depth_cm: float = 24.0,
        pixels_per_cm: float = 12.0,
        image_size: Tuple[int, int] = (800, 600),
        density: float = WATERMELON_DENSITY
    ) -> Tuple[np.ndarray, Dict, float]:
        """
        Bilinen hacimle sentetik goruntu uretir (dogrulama icin).
        
        Args:
            length_cm: Uzunluk (cm) - buyuk eksen
            diameter_cm: Cap (cm) - kucuk eksen
            depth_cm: Derinlik (cm) - 3. eksen (gorunmez)
            pixels_per_cm: Olcek
            density: Yogunluk
            
        Returns:
            image: Sentetik goruntu
            ground_truth: Gercek degerler
            true_volume_cm3: Hesaplanmis gercek hacim
        """
        major_px = int(length_cm * pixels_per_cm)
        minor_px = int(diameter_cm * pixels_per_cm)
        
        image, gt = SyntheticWatermelonGenerator.generate(
            major_axis_px=major_px,
            minor_axis_px=minor_px,
            image_size=image_size,
            angle=0.0,
            noise_level=8.0
        )
        
        # Gercek elipsoid hacmi
        a = length_cm / 2
        b = diameter_cm / 2
        c = depth_cm / 2
        true_volume_cm3 = (4.0 / 3.0) * PI * a * b * c
        true_mass_g = true_volume_cm3 * density
        
        gt.update({
            "length_cm": length_cm,
            "diameter_cm": diameter_cm,
            "depth_cm": depth_cm,
            "pixels_per_cm": pixels_per_cm,
            "true_volume_cm3": true_volume_cm3,
            "true_volume_liters": true_volume_cm3 / 1000.0,
            "true_mass_g": true_mass_g,
            "true_mass_kg": true_mass_g / 1000.0,
        })
        
        return image, gt, true_volume_cm3


# =================================================================
# DOGRULAMA & BENCHMARK
# =================================================================

def validate_estimation_accuracy(
    estimated_volume: float,
    true_volume: float,
    estimated_mass: float = None,
    true_mass: float = None
) -> Dict:
    """
    Tahmin dogrulugunu degerlendirir.
    
    MARE (Mean Absolute Relative Error) hesaplaması:
        MARE = |V_estimated - V_true| / V_true * 100
    
    Hedef: MARE < %8 (Koc, 2007)
    """
    volume_error = abs(estimated_volume - true_volume)
    mare_volume = (volume_error / true_volume) * 100 if true_volume > 0 else float('inf')
    
    result = {
        "estimated_volume_cm3": estimated_volume,
        "true_volume_cm3": true_volume,
        "volume_error_cm3": volume_error,
        "mare_volume_pct": mare_volume,
        "mare_target_pct": 8.0,
        "mare_passed": mare_volume < 8.0,
    }
    
    if estimated_mass is not None and true_mass is not None:
        mass_error = abs(estimated_mass - true_mass)
        mare_mass = (mass_error / true_mass) * 100 if true_mass > 0 else float('inf')
        result.update({
            "estimated_mass_g": estimated_mass,
            "true_mass_g": true_mass,
            "mass_error_g": mass_error,
            "mare_mass_pct": mare_mass,
        })
    
    return result

