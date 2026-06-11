"""
Karpuz Olgunluk Tespit Sistemi - Ana Giris Noktasi

Non-Destructive Multi-Modal Fruit Ripeness Assessment System

Bu script tum modulleri bir araya getirerek uctan uca bir
demo pipeline calistirir.

Kullanim:
    python main.py                  # Demo modunda calistir
    python main.py --train          # Model egitimi
    python main.py --convert        # TFLite donusturme
    python main.py --evaluate       # Model degerlendirme
    python main.py --mrd-detect     # MRD-YOLO gorsel tespit
    python main.py --mrd-validate   # MRD-YOLO model validasyonu
    python main.py --volume img.jpg # Goruntuden hacim tahmini
    python main.py --pressure-demo  # Basinc analizi / temas denetimi demosu
"""

import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from backend.config import *
from backend.module_a.ar_volume import ARVolumeEstimator
from backend.module_a.volume_estimator import (
    ImageVolumeEstimator, SyntheticWatermelonGenerator,
    validate_estimation_accuracy
)
from backend.module_a.visual_analyzer import VisualAnalyzer
from backend.module_b.haptic_controller import HapticSweepDesigner, HapticSimulator
from backend.module_b.pressure_analyzer import PressureAnalyzer
from backend.module_c.srr_reconstruction import (
    SRRReconstructor, SRRSimulator,
    extract_fft_amplitude, baseline_interference_cancellation
)
from backend.module_d.feature_extractor import AcousticFeatureExtractor, AccelerometerFeatureExtractor
from backend.module_e.elasticity_index import ElasticityIndexCalculator
from backend.module_e.late_fusion import LateFusionEngine
from backend.module_e.hollow_heart_detector import HollowHeartDetector


def run_demo():
    """
    Sentetik veri ile tam pipeline demostrayonu.
    
    Gerçek karpuz verisi olmadan tüm modüllerin birlikte
    çalıştığını gösterir.
    """
    print("=" * 70)
    print("   KARPUZ OLGUNLUK TESPIT SISTEMI - DEMO")
    print("   Non-Destructive Multi-Modal Ripeness Assessment")
    print("=" * 70)

    # -------------------------------------------------------
    # MODULE_A-1: AR Hacim ve Kutle Tahmini (Klasik)
    # -------------------------------------------------------
    print("\n>> MODULE_A-1: AR Hacim & Kutle Tahmini (Klasik Elipsoid)")
    print("-" * 50)

    ar_estimator = ARVolumeEstimator()
    dimensions = ar_estimator.estimate_from_2d(width_cm=28.0, length_cm=35.0)

    print(f"  Boyutlar: {dimensions['width_cm']}cm x {dimensions['length_cm']}cm")
    print(f"  Tahmini derinlik: {dimensions['depth_cm_estimated']:.1f}cm")
    print(f"  Hacim: {dimensions['volume_cm3']:.0f} cm3")
    print(f"  Kutle: {dimensions['mass_kg']:.2f} kg")

    # -------------------------------------------------------
    # MODULE_A-2: Goruntu Tabanli Hacim Tahmini - Koc (2007)
    #             Disk Metodu + Elipsoid Yaklasimi
    # -------------------------------------------------------
    print("\n>> MODULE_A-2: Goruntu Tabanli Hacim Tahmini [Koc (2007)]")
    print("-" * 50)
    print("  Disk Metodu + Elipsoid Yaklasimi")

    # Bilinen boyutlarla sentetik karpuz goruntusu uret
    true_length_cm = 35.0
    true_diameter_cm = 28.0
    true_depth_cm = 24.0
    ppc = 12.0  # piksel/cm

    syn_image, ground_truth, true_volume = SyntheticWatermelonGenerator.generate_with_known_volume(
        length_cm=true_length_cm,
        diameter_cm=true_diameter_cm,
        depth_cm=true_depth_cm,
        pixels_per_cm=ppc,
        image_size=(800, 600)
    )

    print(f"\n  [Sentetik Test Goruntusu]")
    print(f"  Gercek boyut: {true_length_cm}x{true_diameter_cm}x{true_depth_cm} cm")
    print(f"  Gercek hacim: {true_volume:.0f} cm3 ({true_volume/1000:.2f} L)")
    print(f"  Gercek kutle: {ground_truth['true_mass_kg']:.2f} kg")
    print(f"  Olcek: {ppc} px/cm")

    # ImageVolumeEstimator ile hacim tahmin et
    vol_estimator = ImageVolumeEstimator(
        pixels_per_cm=ppc,
        disk_slice_width=DISK_SLICE_WIDTH_PX,
        depth_ratio=DISK_DEPTH_RATIO
    )

    vol_result = vol_estimator.estimate_from_image(syn_image)

    # Dogrulama
    if vol_result['best_volume_cm3']:
        accuracy = validate_estimation_accuracy(
            estimated_volume=vol_result['best_volume_cm3'],
            true_volume=true_volume,
            estimated_mass=vol_result['mass_g'],
            true_mass=ground_truth['true_mass_g']
        )
        
        print(f"\n  --- Dogrulama (Koc 2007 MARE < %{VOLUME_MARE_TARGET}) ---")
        print(f"  Tahmini hacim: {accuracy['estimated_volume_cm3']:.0f} cm3")
        print(f"  Gercek hacim:  {accuracy['true_volume_cm3']:.0f} cm3")
        print(f"  Hacim hatasi:  {accuracy['volume_error_cm3']:.0f} cm3")
        print(f"  MARE (hacim):  %{accuracy['mare_volume_pct']:.2f}")
        status = "[+] BASARILI" if accuracy['mare_passed'] else "[-] HEDEF ASILDI"
        print(f"  Durum: {status}")
        
        if accuracy.get('mare_mass_pct') is not None:
            print(f"  MARE (kutle):  %{accuracy['mare_mass_pct']:.2f}")

    # EI hesabi icin kutle degeri (goruntu tabanli veya AR tabanli)
    if vol_result['mass_kg']:
        mass_kg = vol_result['mass_kg']
        print(f"\n  -> EI icin goruntu tabanli kutle: {mass_kg:.2f} kg")
    else:
        mass_kg = dimensions['mass_kg']
        print(f"\n  -> EI icin AR tabanli kutle: {mass_kg:.2f} kg")

    # -------------------------------------------------------
    # MODULE_A-3: MRD-YOLO Gorsel Olgunluk Tespiti
    # -------------------------------------------------------
    print("\n>> MODULE_A-3: MRD-YOLO Gorsel Olgunluk Tespiti")
    print("-" * 50)
    print("  Kaynak: github.com/XuebinJing/Melon-Ripeness-Detection")
    print("  Backbone: MobileNetV3 | Attention: CoordAtt | Neck: VoVGSCSP+GSConv")

    try:
        analyzer = VisualAnalyzer(auto_load=True)
        
        if analyzer.model_loaded:
            # Test goruntusu uzerinde tespit
            test_img_dir = str(MRD_TEST_IMAGES)
            if os.path.isdir(test_img_dir):
                import glob
                test_imgs = glob.glob(os.path.join(test_img_dir, "*.jpg")) + \
                            glob.glob(os.path.join(test_img_dir, "*.JPG"))
                
                if test_imgs:
                    # Ilk 3 goruntude test
                    sample_imgs = test_imgs[:3]
                    print(f"\n  Test seti: {len(test_imgs)} goruntu bulundu")
                    print(f"  Ornek tespit ({len(sample_imgs)} goruntu):")
                    
                    import cv2
                    for img_path in sample_imgs:
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        result = analyzer.predict_single(image)
                        img_name = os.path.basename(img_path)
                        
                        print(f"    {img_name}: {result['class_name']} (guven: {result['confidence']:.2f})")
                else:
                    print(f"  [!] Test goruntusu bulunamadi: {test_img_dir}")
            else:
                print(f"  [!] Test dizini bulunamadi: {test_img_dir}")
                print(f"  Sentetik goruntu ile test ediliyor...")
                
                # Sentetik goruntu ile heristik analiz
                visual_result = analyzer.get_visual_ripeness_score(syn_image)
                print(f"    Heristik skor: {visual_result['visual_score']:.3f}")
                print(f"    Sinif: {visual_result['visual_class_name']}")
            
            # Model bilgisi
            info = analyzer.get_model_info()
            print(f"\n  Model: {info['model_name']}")
            print(f"  Siniflar: {info['class_names']}")
            print(f"  Backbone: {info['architecture']['backbone']}")
            print(f"  Attention: {info['architecture']['attention']}")
        else:
            print("  [!] MRD-YOLO modeli yuklenemedi")
            print("  mrd_yolo_repo/weights/MRD.pt dosyasini kontrol edin")
            
            # Heristik analiz ile devam et
            print("\n  Heristik gorsel analiz ile devam ediliyor...")
            analyzer_fallback = VisualAnalyzer(auto_load=False)
            visual_result = analyzer_fallback.get_visual_ripeness_score(syn_image)
            print(f"    Heristik skor: {visual_result['visual_score']:.3f}")
            print(f"    Sinif: {visual_result['visual_class_name']}")
    except Exception as e:
        print(f"  [!] MRD-YOLO hatasi: {e}")
        print("  ultralytics paketi veya mrd_yolo_repo gerekli olabilir")

    # -------------------------------------------------------
    # MODULE_B: Haptik Frekans Taramasi
    # -------------------------------------------------------
    print("\n>> MODULE_B: Haptik Frekans Taramasi")
    print("-" * 50)

    sweep_designer = HapticSweepDesigner()
    t_chirp, chirp_signal = sweep_designer.generate_chirp_signal(sample_rate=44100)
    steps = sweep_designer.generate_stepped_frequencies()

    print(f"  Frekans aralığı: {HAPTIC_FREQ_START}-{HAPTIC_FREQ_END} Hz")
    print(f"  Tarama süresi: {HAPTIC_SWEEP_DURATION} saniye")
    print(f"  Toplam adım: {len(steps)}")
    print(f"  Chirp sinyal uzunluğu: {len(chirp_signal)} örnek")

    # Simüle edilmiş karpuz yanıtı
    simulator = HapticSimulator()
    response_ripe = simulator.simulate_watermelon_response(
        chirp_signal, sample_rate=44100, ripeness="ripe"
    )
    response_immature = simulator.simulate_watermelon_response(
        chirp_signal, sample_rate=44100, ripeness="immature"
    )

    print(f"  Simulasyon: Olgun yanit RMS = {np.sqrt(np.mean(response_ripe**2)):.4f}")
    print(f"  Simulasyon: Olgunlasmamis yanit RMS = {np.sqrt(np.mean(response_immature**2)):.4f}")

    # -------------------------------------------------------
    # MODULE_B-2: Basinc Analizi / Temas Denetimi
    # -------------------------------------------------------
    print("\n>> MODULE_B-2: Basinc Analizi / Temas Denetimi")
    print("-" * 50)

    pressure_analyzer = PressureAnalyzer()

    # Kalibrasyon
    idle_z = np.random.normal(9.81, 0.02, 200)
    cal = pressure_analyzer.calibrate(idle_z, sample_rate=100.0)
    print(f"  Kalibrasyon: g_ref = {cal['baseline_gravity']:.4f} m/s^2")

    # Ideal temas simule et
    ax, ay, az = PressureAnalyzer.simulate_contact_data(
        duration_s=3.0, sample_rate=100.0, contact_type="good"
    )
    pressure_result = pressure_analyzer.analyze(ax, ay, az, sample_rate=100.0)
    print(f"  Temas kalitesi: {pressure_result['contact_quality']}")
    print(f"  Basinc (norm): {pressure_result['contact_pressure']:.2f}")
    print(f"  Mesaj: {pressure_result['message']}")

    # Haptik ozellik vektoru
    haptic_features = pressure_analyzer.extract_pressure_features(ax, ay, az, 100.0)
    print(f"  Haptik ozellik vektoru: {len(haptic_features)} boyut")
    print(f"  [contact_pressure={haptic_features[0]:.2f}, stability={haptic_features[5]:.2f}]")

    # -------------------------------------------------------
    # MODULE_C: SRR Sinyal Rekonstruksiyonu (Vi-Liquid)
    # -------------------------------------------------------
    print("\n>> MODULE_C: SRR Sinyal Rekonstruksiyonu (Vi-Liquid Metodolojisi)")
    print("-" * 50)

    # --- 1. Sentetik test verisi uret ---
    true_f2 = 120.0  # Bilinen gercek frekans (olgun karpuz)
    test_duration = 5.0  # 5 saniye (daha fazla veri = daha iyi geri kazanim)
    print(f"\n  [Test Verisi] Gercek f2={true_f2}Hz, sure={test_duration}s")
    print(f"  NOT: 100Hz IMU Nyquist=50Hz, NUFFT-OMP non-uniform")
    print(f"       ornekleme ile super-Nyquist geri kazanim dener")

    imu_data, timestamps, true_signal = SRRSimulator.generate_test_data(
        true_frequency=true_f2,
        lra_frequency=SRR_LRA_FREQUENCY,
        imu_rate=IMU_NATIVE_RATE,
        duration=test_duration,
        noise_level=0.02,
        harmonics=[(240.0, 0.2)]  # 2. harmonik
    )
    print(f"  IMU ornekleri: {len(imu_data)} @ {IMU_NATIVE_RATE}Hz")

    # --- 2. Baseline (bosta profil) uret ---
    baseline_data = 0.02 * np.random.randn(len(imu_data))
    print(f"  Baseline: {len(baseline_data)} ornek (ortam gurultusu)")

    # --- 3. Chirp Sweep + SRR: f2 tespiti (Ana Yontem) ---
    srr = SRRReconstructor(lra_frequency=SRR_LRA_FREQUENCY)

    print(f"\n  === Chirp Sweep + SRR (Vi-Liquid Ana Yontemi) ===")
    print(f"  LRA'yi 80-400Hz arasi surarak transfer fonksiyonu olusturur")
    print(f"  H(f)'nin tepe noktasi = dogal frekans f2")

    # Karpuz yaniti simule et (true_f2=120Hz'de rezonans)
    sweep_data = srr.simulate_chirp_sweep(
        true_f2=true_f2,
        freq_start=80.0,
        freq_end=400.0,
        n_steps=32,
        step_duration=0.3,
        damping=0.08,
        noise_level=0.02
    )

    # Chirp sweep + SRR ile f2 bul
    detected_f2, f2_h_db, sweep_result = srr.chirp_sweep_f2_detection(
        sweep_data,
        freq_range=(80.0, 300.0)
    )

    freq_error = abs(detected_f2 - true_f2)
    freq_error_pct = (freq_error / true_f2) * 100

    print(f"\n  --- Chirp Sweep Sonucu ---")
    print(f"  Algilanan f2: {detected_f2:.1f} Hz (Gercek: {true_f2:.1f} Hz)")
    print(f"  Frekans hatasi: {freq_error:.1f} Hz ({freq_error_pct:.1f}%)")
    print(f"  Transfer fonk. tepesi: {f2_h_db:.1f} dB")
    print(f"  Tarama: {sweep_result['n_steps']} frekans adimi")

    # --- 4. NUFFT-OMP ile rekonstruksiyon (Ek Yontem) ---
    print(f"\n  === NUFFT-OMP Rekonstruksiyon (Ek Dogrulama) ===")
    try:
        time_recon, signal_recon, srr_details = srr.reconstruct(
            imu_data=imu_data,
            timestamps_ns=timestamps,
            baseline_signal=baseline_data,
            n_omp_atoms=SRR_OMP_N_ATOMS,
            omp_freq_range=SRR_OMP_FREQ_RANGE
        )
        print(f"  Giris: {IMU_NATIVE_RATE}Hz, {len(imu_data)} ornek")
        print(f"  Cikis: {SRR_TARGET_RATE}Hz, {len(signal_recon)} sanal ornek")
        print(f"  Yukseltme: {SRR_UPSAMPLE_FACTOR}x")
    except Exception as e:
        print(f"  NUFFT-OMP hata: {e}")
        signal_recon = imu_data

    # --- 5. SPI Girisim iptali ---
    clean_signal = baseline_interference_cancellation(imu_data, baseline_data)
    freqs, amps, dom_freq, dom_db = extract_fft_amplitude(
        clean_signal, IMU_NATIVE_RATE, freq_range=(50, 500)
    )
    print(f"\n  [SPI+FFT] Ham sinyal dominant: {dom_freq:.1f} Hz ({dom_db:.1f} dB)")

    # -------------------------------------------------------
    # MODULE_D: Özellik Mühendisliği
    # -------------------------------------------------------
    print("\n>> MODULE_D: Ozellik Muhendisligi")
    print("-" * 50)

    # Sentetik ses sinyali (vuruş sesi simülasyonu)
    duration = 0.5  # 0.5 saniye
    sr = AUDIO_SAMPLE_RATE
    t_audio = np.linspace(0, duration, int(sr * duration))
    # Sönümlenmiş sinüsoidal (karpuz vuruş sesi benzeri)
    audio_signal = (
        0.8 * np.exp(-10 * t_audio) * np.sin(2 * np.pi * 120 * t_audio) +
        0.3 * np.exp(-15 * t_audio) * np.sin(2 * np.pi * 240 * t_audio) +
        0.05 * np.random.randn(len(t_audio))
    )

    acoustic_extractor = AcousticFeatureExtractor()
    features = acoustic_extractor.extract_all_features(audio_signal, sr)

    print(f"  f2 (dominant frekans): {features['f2']:.1f} Hz")
    print(f"  f2 genligi: {features['f2_magnitude_db']:.1f} dB")
    print(f"  ZCR (Sifir Gecis Orani): {features['zcr_mean']:.4f}")
    print(f"  Spektral Centroid: {features['spectral_centroid_mean']:.1f} Hz")
    print(f"  Spektral Entropi: {features['spectral_entropy']:.4f}")
    print(f"  RMS Enerji: {features['rms_mean']:.4f}")
    print(f"  Peak Genlik: {features['peak_amplitude']:.4f}")
    print(f"  Crest Factor: {features['crest_factor']:.2f}")
    print(f"  Decay Rate: {features['decay_rate']:.2f}")

    feature_vector = acoustic_extractor.extract_feature_vector(audio_signal, sr)
    print(f"\n  Koc & Akbalik (2025) Ozellik Vektoru:")
    print(f"  Toplam boyut: {len(feature_vector)} ozellik")
    print(f"    [0:52]    MFCC istatistikleri (mean/std/min/max x 13)")
    print(f"    [52:78]   Delta MFCC (mean/std x 13)")
    print(f"    [78:80]   ZCR (mean, std)")
    print(f"    [80:95]   Spektral (centroid/bw/rolloff/flatness/contrast)")
    print(f"    [95:99]   Enerji (RMS + Log Energy)")
    print(f"    [99:111]  Chroma (12 pitch sinifi)")
    print(f"    [111:114] Frekans & rezonans (f2, f2_db, entropy)")
    print(f"    [114:120] Zaman alani (peak/crest/centroid/attack/decay/duration)")

    # Ivmeolcer ozellikleri
    accel_extractor = AccelerometerFeatureExtractor(sample_rate=1600)
    accel_features = accel_extractor.extract_features(clean_signal)
    print(f"  Ivmeolcer f2: {accel_features['accel_f2']:.1f} Hz")

    # -------------------------------------------------------
    # MODULE_E: Çıkarım Motoru
    # -------------------------------------------------------
    print("\n>> MODULE_E: Cikarim Motoru & Late Fusion")
    print("-" * 50)

    # Elasticity Index
    ei_calc = ElasticityIndexCalculator()
    f2 = features['f2']
    ei = ei_calc.calculate_ei(f2, mass_kg)

    print(f"  Elasticity Index (EI): {ei:.2f}")
    print(f"  Formul: EI = f2^2 x m^(2/3) = {f2:.1f}^2 x {mass_kg:.2f}^(2/3)")

    # Fiziksel dogrulama
    validation = ei_calc.validate_physical_constraints(f2, features['f2_magnitude_db'])
    print(f"  f2 < {F2_RIPE_THRESHOLD}Hz -> Olgun: {'[+] EVET' if validation['f2_indicates_ripe'] else '[-] HAYIR'}")
    print(f"  Genlik > {MAGNITUDE_RIPE_THRESHOLD}dB -> Yeterli: {'[+] EVET' if validation['magnitude_sufficient'] else '[-] HAYIR'}")

    # EI tabanli siniflandirma
    class_id, class_label, confidence = ei_calc.classify_by_ei(
        ei, f2, features['zcr_mean']
    )
    print(f"  EI Siniflandirma: {class_label} (Guven: {confidence:.2f})")

    # Late Fusion
    fusion = LateFusionEngine()

    # Gorsel skor (simule)
    visual_score = 0.6  # Orta-yuksek gorsel olgunluk
    # Akustik sertlik
    stiffness = ei_calc.compute_stiffness_index(f2, features['f2_magnitude_db'], mass_kg)

    fusion_result = fusion.fuse_scores(
        visual_score=visual_score,
        acoustic_stiffness=stiffness['stiffness_score']
    )

    print(f"\n  --- Late Fusion Sonucu ---")
    print(f"  Gorsel skor: {visual_score:.2f} (agirlik: {fusion.visual_weight:.2f})")
    print(f"  Akustik sertlik: {stiffness['stiffness_score']:.2f} (agirlik: {fusion.acoustic_weight:.2f})")
    print(f"  Birlesik skor: {fusion_result['fused_score']:.2f}")
    print(f"  +======================================+")
    print(f"  | NIHAI SONUC: {fusion_result['class_label']:>22} |")
    print(f"  | Guven Skoru: {fusion_result['confidence']:>22.1%} |")
    print(f"  +======================================+")

    print("\n" + "=" * 70)
    print("   Demo tamamlandi! Tum moduller basariyla calisti.")
    print("=" * 70)


def run_volume_estimation(image_path: str, scale: float = None):
    """
    Verilen goruntuden Disk Metodu ile hacim/kutle tahmin eder.
    
    Kullanim:
        python main.py --volume karpuz.jpg --scale 12.0
        python main.py --volume karpuz.jpg   (kalibrasyon olmadan)
    """
    import cv2
    
    print("=" * 70)
    print("   GORUNTU TABANLI HACIM/KUTLE TAHMINI")
    print("   Ali Bulent Koc (2007) - Disk Metodu + Elipsoid")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"\n  [HATA] Goruntu bulunamadi: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"\n  [HATA] Goruntu okunamadi: {image_path}")
        return
    
    print(f"\n  Goruntu: {image_path}")
    print(f"  Boyut: {image.shape[1]}x{image.shape[0]} piksel")
    
    estimator = ImageVolumeEstimator(
        pixels_per_cm=scale,
        disk_slice_width=DISK_SLICE_WIDTH_PX,
        depth_ratio=DISK_DEPTH_RATIO
    )
    
    result = estimator.estimate_from_image(image, return_debug=True)
    
    # Sonuc ozeti
    print("\n" + "=" * 70)
    print("   HACIM/KUTLE TAHMIN SONUCU")
    print("=" * 70)
    
    print(f"\n  Disk Metodu Hacmi:")
    if result['disk_method']['volume_cm3']:
        print(f"    {result['disk_method']['volume_cm3']:.0f} cm3 ({result['disk_method']['volume_liters']:.2f} L)")
    else:
        print(f"    {result['disk_method']['volume_px3']:.0f} px3 (kalibrasyon gerekli)")
    
    print(f"\n  Elipsoid Yaklasimi:")
    if result['ellipsoid']['volume_cm3']:
        print(f"    {result['ellipsoid']['volume_cm3']:.0f} cm3 ({result['ellipsoid']['volume_liters']:.2f} L)")
    else:
        print(f"    {result['ellipsoid']['volume_px3']:.0f} px3 (kalibrasyon gerekli)")
    
    if result['mass_kg']:
        print(f"\n  Tahmini Kutle: {result['mass_kg']:.2f} kg ({result['mass_g']:.0f} g)")
        print(f"  Yogunluk: {WATERMELON_DENSITY} g/cm3")
        
        # EI hesabi icin kutle degerini goster
        from backend.module_e.elasticity_index import ElasticityIndexCalculator
        ei_calc = ElasticityIndexCalculator()
        # Ornek f2 = 120 Hz (tipik olgun karpuz)
        sample_f2 = 120.0
        ei = ei_calc.calculate_ei(sample_f2, result['mass_kg'])
        print(f"\n  -> Elasticity Index (f2={sample_f2}Hz): EI = {ei:.2f}")
        print(f"     Formul: EI = f2^2 * m^(2/3) = {sample_f2}^2 * {result['mass_kg']:.2f}^(2/3)")
    else:
        print(f"\n  [!] Kutle hesabi icin --scale parametresi gerekli")
        print(f"  Ornek: python main.py --volume {image_path} --scale 12.0")
    
    # Analiz goruntusunu kaydet
    if result.get('debug'):
        contour = result['debug']['contour']
        analysis_img = estimator.draw_analysis(
            image, contour, result['axes'], result['disk_method']
        )
        output_path = image_path.rsplit('.', 1)[0] + "_analysis.jpg"
        cv2.imwrite(output_path, analysis_img)
        print(f"\n  Analiz goruntusu kaydedildi: {output_path}")
    
    print("=" * 70)


def check_dataset():
    """
    Qilin dataset'inin durumunu kontrol eder.
    dataset/datasets/ dizinini tarar ve rapor verir.
    """
    from backend.pipeline.data_loader import QilinDatasetLoader
    
    print("=" * 70)
    print("   QILIN DATASET DURUM KONTROLU")
    print("=" * 70)
    
    loader = QilinDatasetLoader()
    info = loader.get_dataset_info()
    
    print(f"\n  Yapi tipi: {info['structure_type']}")
    print(f"  Qilin datasets dizini: {info['qilin_datasets_dir']}")
    print(f"  Data dizini: {info['data_dir']}")
    print(f"  WAV sayisi: {info['wav_count']}")
    print(f"  Goruntu sayisi: {info.get('image_count', 0)}")
    
    if info.get("sample_count"):
        print(f"\n  Toplam ornek: {info['sample_count']}")
        if info.get("brix_range"):
            print(f"  Brix araligi: {info['brix_range'][0]:.1f} - {info['brix_range'][1]:.1f}")
        if info.get("class_distribution"):
            print(f"  Sinif dagilimi:")
            for k, v in info["class_distribution"].items():
                print(f"    {k}: {v}")
        if info.get("sample_example"):
            ex = info["sample_example"]
            print(f"\n  Ornek kayit:")
            print(f"    ID: {ex.get('data_id')}")
            print(f"    Brix: {ex.get('brix')}")
            print(f"    WAV: {ex.get('wav_path')}")
            print(f"    JPG: {ex.get('jpg_path')}")
    else:
        print(f"\n  [!] Dataset bulunamadi!")
        print(f"  Beklenen yapi:")
        print(f"    dataset/datasets/{{id}}_{{brix}}/chu/{{folder}}/{{*.wav, *.jpg}}")
        print(f"\n  Ornek:")
        print(f"    dataset/datasets/001_8.5/chu/1/knock.wav")
        print(f"    dataset/datasets/001_8.5/chu/1/photo.jpg")
    
    print("\n  Kullanilabilir komutlar:")
    print("    python main.py --preprocess     # Veriyi isle")
    print("    python main.py --train-real      # Gercek veriyle egit")
    print("    python main.py --train           # Sentetik veriyle egit")
    print("=" * 70)


def run_mrd_detect(image_path: str = None, max_images: int = 10, save_output: bool = True):
    """
    MRD-YOLO ile karpuz olgunluk tespiti.
    
    Kullanim:
        python main.py --mrd-detect                    # Test seti
        python main.py --mrd-detect --image karpuz.jpg # Tek goruntu
    """
    import cv2
    
    print("=" * 70)
    print("   MRD-YOLO KARPUZ OLGUNLUK TESPITI")
    print("   github.com/XuebinJing/Melon-Ripeness-Detection")
    print("   Mimari: MobileNetV3 + CoordAtt + VoVGSCSP + GSConv")
    print("=" * 70)
    
    analyzer = VisualAnalyzer(auto_load=True)
    
    if not analyzer.model_loaded:
        print("\n[HATA] MRD-YOLO model yuklenemedi!")
        print("  MRD.pt dosyasinin mevcut oldugunu kontrol edin:")
        print(f"    1. {MRD_YOLO_MODEL_PATH}")
        print(f"    2. {MRD_YOLO_FALLBACK_PATH}")
        return
    
    # Mimari ozeti goster
    print(analyzer.get_architecture_summary())
    
    if image_path and os.path.isfile(image_path):
        # Tek goruntu
        print(f"\n  Goruntu: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"  [HATA] Goruntu okunamadi!")
            return
        
        result = analyzer.predict_single(image)
        
        print(f"\n  Kaynak: {result['source']}")
        print(f"  Sinif: {result['class_name']}")
        print(f"  Guven: {result['confidence']:.3f}")
        print(f"  Olgun: {'Evet' if result['is_ripe'] else 'Hayir'}")
        
        if result.get('all_detections'):
            print(f"\n  Toplam tespit: {result['n_detections']}")
            for i, d in enumerate(result['all_detections']):
                print(f"  [{i+1}] {d['class_name']} - Guven: {d['confidence']:.3f}")
                print(f"      BBox: [{d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f}]")
        
        if save_output and result.get('all_detections'):
            output_img = analyzer.draw_detections(image, result['all_detections'])
            out_path = image_path.rsplit('.', 1)[0] + "_mrd_detect.jpg"
            cv2.imwrite(out_path, output_img)
            print(f"\n  Sonuc goruntusu: {out_path}")
    else:
        # Toplu tespit (test seti)
        import glob as glob_mod
        
        test_dir = str(MRD_TEST_IMAGES)
        if not os.path.exists(test_dir):
            print(f"\n[HATA] Test dizini bulunamadi: {test_dir}")
            return
        
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
            image_files.extend(glob_mod.glob(os.path.join(test_dir, ext)))
        
        image_files = image_files[:max_images]
        
        print(f"\n  Test dizini: {test_dir}")
        print(f"  Goruntu sayisi: {len(image_files)}")
        
        ripe_count = 0
        unripe_count = 0
        total_conf = 0.0
        
        for i, img_path in enumerate(image_files):
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            dets = analyzer.detect_and_classify(image)
            for d in dets:
                if d['is_ripe']:
                    ripe_count += 1
                else:
                    unripe_count += 1
                total_conf += d['confidence']
            
            if (i + 1) % 5 == 0 or i == 0:
                status = dets[0]['class_name'] if dets else "Tespit yok"
                conf = f"{dets[0]['confidence']:.3f}" if dets else "-"
                print(f"  [{i+1}/{len(image_files)}] {os.path.basename(img_path)}: {status} ({conf})")
        
        total = ripe_count + unripe_count
        if total > 0:
            print(f"\n  Sonuclar:")
            print(f"    Olgun (Ripe):       {ripe_count} ({ripe_count/total*100:.1f}%)")
            print(f"    Olgunlasmamis:      {unripe_count} ({unripe_count/total*100:.1f}%)")
            print(f"    Ort. Guven:         {total_conf/total:.3f}")
        
        if save_output and image_files:
            os.makedirs("data/mrd_results", exist_ok=True)
            for img_path in image_files[:5]:
                img = cv2.imread(img_path)
                if img is not None:
                    dets = analyzer.detect_and_classify(img)
                    if dets:
                        out_img = analyzer.draw_detections(img, dets)
                        out_path = os.path.join("data", "mrd_results", f"detect_{os.path.basename(img_path)}")
                        cv2.imwrite(out_path, out_img)
            print(f"\n  Sonuc goruntuleri: data/mrd_results/")
    
    print("=" * 70)


def run_mrd_validate():
    """
    MRD-YOLO model validasyonu (mAP hesaplama).
    
    Kullanim:
        python main.py --mrd-validate
    """
    print("=" * 70)
    print("   MRD-YOLO MODEL VALIDASYONU")
    print("   Mimari: MobileNetV3 + CoordAtt + VoVGSCSP + GSConv")
    print("=" * 70)
    
    analyzer = VisualAnalyzer(auto_load=True)
    
    if not analyzer.model_loaded:
        print("\n[HATA] Model yuklenemedi!")
        return
    
    # Model bilgilerini goster
    info = analyzer.get_model_info()
    print(f"\n  Model: {info['model_name']}")
    print(f"  Backbone: {info['architecture']['backbone']}")
    print(f"  Neck: {info['architecture']['neck']}")
    print(f"  Head: {info['architecture']['head']}")
    print(f"  Attention: {info['architecture']['attention']}")
    print(f"  Siniflar: {info['class_names']}")
    
    result = analyzer.validate_model()
    
    if result:
        print(f"\n  Genel Performans:")
        print(f"    mAP@0.5:      {result['map50']*100:.2f}%")
        print(f"    mAP@0.75:     {result['map75']*100:.2f}%")
        print(f"    mAP@0.5:0.95: {result['map50_95']*100:.2f}%")
        
        if result.get('maps_per_class'):
            print(f"\n  Sinif Bazli mAP@0.5:0.95:")
            for i, m in enumerate(result['maps_per_class']):
                name = MRD_CLASS_NAMES.get(i, f"class_{i}")
                print(f"    {name}: {m*100:.2f}%")
    
    print("=" * 70)


def run_tflite_benchmark():
    """TFLite modellerini benchmark eder."""
    from backend.pipeline.tflite_converter import benchmark_all_models
    benchmark_all_models()


def run_crowdsourcing_demo():
    """Crowdsourcing demo calistirir ve rapor gosterir."""
    from backend.pipeline.crowdsourcing import FeedbackLogger, generate_demo_feedback

    print("=" * 70)
    print("   CROWDSOURCING GERI BILDIRIM SISTEMI")
    print("=" * 70)

    logger = FeedbackLogger()

    if logger.total_entries == 0:
        print("\n  Henuz geri bildirim yok, demo verisi uretiliyor...")
        logger = generate_demo_feedback(n_samples=30)

    # Rapor
    logger.print_summary()

    # CSV export
    logger.export_to_csv()

    # Performans trendi
    trend = logger.get_model_performance_trend(window=10)
    if trend:
        print(f"\n  Model Performans Trendi (son 5 pencere):")
        for t in trend[-5:]:
            print(f"    {t['window_start'][:19]}: dogruluk={t['accuracy']:.1%}, "
                  f"guven={t['avg_confidence']:.2f}")

    # Egitim verisi kontrol
    X, X_hh, y = logger.prepare_training_data()
    if X is not None:
        print(f"\n  Egitim icin hazir veri: {len(y)} ornek")
        print(f"  Komut: python main.py --retrain-with-feedback")

    print("=" * 70)


def run_pressure_demo():
    """
    Basinc analizi ve temas denetimi demosu.
    
    Simule ivmeolcer verileri ile PressureAnalyzer modulunu test eder.
    4 farkli temas senaryosunu calistirir.
    
    Kullanim:
        python main.py --pressure-demo
    """
    print("=" * 70)
    print("   BASINC ANALIZI / TEMAS DENETIMI DEMOSU")
    print("   MODULE_B: Ivmeolcer Tabanli Basinc Gostergesi")
    print("=" * 70)

    analyzer = PressureAnalyzer()

    # Kalibrasyon (bosta yercekimi referansi)
    print("\n>> 1. KALIBRASYON")
    print("-" * 50)
    idle_z = np.random.normal(9.81, 0.02, 200)  # Bosta 2 saniye @ 100Hz
    cal_result = analyzer.calibrate(idle_z, sample_rate=100.0)
    print(f"  Baseline yercekimi: {cal_result['baseline_gravity']:.4f} m/s^2")
    print(f"  Standart sapma: {cal_result['gravity_std']:.4f}")
    print(f"  Sure: {cal_result['duration_s']:.1f} saniye")

    # 4 farkli temas senaryosu
    scenarios = [
        ("Temas Yok", "none"),
        ("Hafif Temas", "light"),
        ("Ideal Temas", "good"),
        ("Fazla Basinc", "strong"),
    ]

    print("\n>> 2. TEMAS SENARYOLARI")
    print("-" * 50)

    for label, contact_type in scenarios:
        print(f"\n  --- {label.upper()} ---")
        ax, ay, az = PressureAnalyzer.simulate_contact_data(
            duration_s=3.0,
            sample_rate=100.0,
            contact_type=contact_type,
            noise_level=0.05,
        )
        result = analyzer.analyze(ax, ay, az, sample_rate=100.0, verbose=True)

    # Degisken basinc profili
    print(f"\n  --- DEGISKEN BASINC PROFILI ---")
    ax, ay, az = PressureAnalyzer.simulate_contact_data(
        duration_s=5.0,
        sample_rate=100.0,
        contact_type="variable",
        noise_level=0.03,
    )
    result = analyzer.analyze(ax, ay, az, sample_rate=100.0, verbose=True)
    
    profile = result["pressure_profile"]
    print(f"\n  Basinc Profili Ozeti:")
    print(f"    Pencere sayisi: {profile['n_windows']}")
    print(f"    Ortalama basinc: {profile['mean_pressure']:.2f}")
    print(f"    Max basinc: {profile['max_pressure']:.2f}")
    print(f"    Min basinc: {profile['min_pressure']:.2f}")
    
    events = result["contact_events"]
    print(f"\n  Temas Olaylari: {len(events)} adet")
    for i, e in enumerate(events[:5]):
        print(f"    [{i+1}] {e['start_time_s']:.2f}s - {e['end_time_s']:.2f}s "
              f"(sure: {e['duration_s']:.2f}s, tepe: {e['peak_deviation']:.3f} m/s^2)")

    # Late Fusion icin ozellik vektoru
    print(f"\n>> 3. HAPTIK OZELLIK VEKTORU (Late Fusion Girdisi)")
    print("-" * 50)
    features = analyzer.extract_pressure_features(ax, ay, az, sample_rate=100.0)
    feature_names = [
        "contact_pressure",
        "gravity_deviation",
        "vibration_rms",
        "tilt_angle_norm",
        "magnitude_mean",
        "stability_score",
        "contact_quality_score",
    ]
    print(f"  Boyut: {len(features)} ozellik")
    for name, val in zip(feature_names, features):
        print(f"    {name:25s}: {val:.4f}")

    print(f"\n  Bu vektor Late Fusion modeline haptic_features olarak verilir.")
    print(f"  TFLite girdi boyutu: TFLITE_HAPTIC_INPUT_DIM = {len(features)}")

    print("\n" + "=" * 70)
    print("   Basinc analizi demosu tamamlandi!")
    print("=" * 70)


def run_feedback_report():
    """Mevcut geri bildirim raporunu gosterir."""
    from backend.pipeline.crowdsourcing import FeedbackLogger

    logger = FeedbackLogger()

    if logger.total_entries == 0:
        print("\n  [!] Henuz geri bildirim verisi toplanmadi.")
        print("  Demo icin: python main.py --crowdsourcing-demo")
        return

    logger.print_summary()
    logger.export_to_csv()


def main():
    parser = argparse.ArgumentParser(
        description="Karpuz Olgunluk Tespit Sistemi"
    )
    parser.add_argument("--train", action="store_true",
                        help="Model egitimi baslat")
    parser.add_argument("--convert", action="store_true",
                        help="TFLite donusturme")
    parser.add_argument("--evaluate", action="store_true",
                        help="Model degerlendirme")
    parser.add_argument("--model", type=str, default="both",
                        choices=["knn", "rfc", "both"],
                        help="Model tipi")
    parser.add_argument("--check-dataset", action="store_true",
                        help="Dataset durumunu kontrol et")
    parser.add_argument("--preprocess", action="store_true",
                        help="Qilin dataset on isleme")
    parser.add_argument("--train-real", action="store_true",
                        help="Gercek Qilin verisiyle egitim")
    parser.add_argument("--volume", type=str, default=None,
                        help="Goruntu dosyasindan hacim/kutle tahmini (goruntu yolu)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Piksel/cm olcek orani (--volume ile)")
    # MRD-YOLO komutlari
    parser.add_argument("--mrd-detect", action="store_true",
                        help="MRD-YOLO ile karpuz olgunluk tespiti")
    parser.add_argument("--mrd-validate", action="store_true",
                        help="MRD-YOLO model validasyonu (mAP)")
    parser.add_argument("--image", type=str, default=None,
                        help="Goruntu dosya yolu (--mrd-detect ile)")
    parser.add_argument("--max-images", type=int, default=10,
                        help="Toplu tespit icin maksimum goruntu sayisi")
    # TFLite komutlari
    parser.add_argument("--quantize", type=str, default="all",
                        choices=["none", "float16", "int8", "dynamic", "all"],
                        help="TFLite quantization turu (--convert ile)")
    parser.add_argument("--tflite-benchmark", action="store_true",
                        help="TFLite model benchmark karsilastirmasi")
    # Crowdsourcing komutlari
    parser.add_argument("--crowdsourcing-demo", action="store_true",
                        help="Crowdsourcing demo calistir")
    parser.add_argument("--feedback-report", action="store_true",
                        help="Geri bildirim raporunu goster")
    parser.add_argument("--retrain-with-feedback", action="store_true",
                        help="Geri bildirim verisiyle yeniden egit")
    # Basinc / Temas Denetimi
    parser.add_argument("--pressure-demo", action="store_true",
                        help="Basinc analizi ve temas denetimi demosu")

    args = parser.parse_args()

    if args.mrd_detect:
        run_mrd_detect(args.image, args.max_images)
    elif args.mrd_validate:
        run_mrd_validate()
    elif args.volume:
        run_volume_estimation(args.volume, args.scale)
    elif args.check_dataset:
        check_dataset()
    elif args.preprocess:
        from scripts.preprocess_qilin import preprocess_dataset
        preprocess_dataset()
    elif args.train_real:
        from backend.pipeline.train import train_pipeline
        train_pipeline(model_type=args.model, use_synthetic=False)
    elif args.retrain_with_feedback:
        from backend.pipeline.train import train_pipeline
        train_pipeline(model_type=args.model, use_synthetic=True, use_feedback=True)
    elif args.train:
        from backend.pipeline.train import train_pipeline
        train_pipeline(model_type=args.model, use_synthetic=True)
    elif args.convert:
        from backend.pipeline.tflite_converter import main as convert_main
        convert_main(quantize=args.quantize, benchmark=args.tflite_benchmark)
    elif args.tflite_benchmark:
        run_tflite_benchmark()
    elif args.crowdsourcing_demo:
        run_crowdsourcing_demo()
    elif args.feedback_report:
        run_feedback_report()
    elif args.pressure_demo:
        run_pressure_demo()
    elif args.evaluate:
        from scripts.evaluate import evaluate_model
        evaluate_model(args.model if args.model != "both" else "knn")
    else:
        run_demo()


if __name__ == "__main__":
    main()

