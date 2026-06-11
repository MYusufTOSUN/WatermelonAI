"""
Proje genelinde kullanılan konfigürasyon sabitleri ve ayarlar.
"""

import os
from pathlib import Path

# ============================================
# Dizin Yolları
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Qilin Dataset Yollari
# Yapi: dataset/datasets/{id}_{brix}/chu/{folder}/{*.wav, *.jpg}
QILIN_DATASET_DIR = PROJECT_ROOT / "dataset"
QILIN_DATASETS_DIR = QILIN_DATASET_DIR / "datasets"

# ============================================
# MODULE_B: Aktif Algılama Sabitleri
# ============================================
HAPTIC_FREQ_START = 100        # Hz - Frekans taraması başlangıcı
HAPTIC_FREQ_END = 400          # Hz - Frekans taraması bitişi
HAPTIC_SWEEP_DURATION = 2.0    # saniye - Toplam tarama süresi
HAPTIC_STEP_MS = 10            # ms - Her titreşim periyodu

# ============================================
# MODULE_C: SRR Sabitleri (Vi-Liquid Metodolojisi)
# ============================================
IMU_NATIVE_RATE = 100          # Hz - Android IMU'nun dogal ornekleme hizi
SRR_TARGET_RATE = 1600         # Hz - SRR ile hedeflenen sanal ornekleme hizi
SRR_UPSAMPLE_FACTOR = SRR_TARGET_RATE // IMU_NATIVE_RATE  # 16x
SRR_LRA_FREQUENCY = 167.0     # Hz - LRA motor nominal frekansi
SRR_LRA_PERIOD_NS = int(1e9 / SRR_LRA_FREQUENCY)  # ~5988024 ns
SRR_OMP_N_ATOMS = 8           # OMP atom sayisi (seyrek geri kazanim)
SRR_OMP_FREQ_RANGE = (50.0, 800.0)  # OMP frekans arama araligi (Hz)
SRR_SPI_ALPHA = 1.0           # SPI cikartma katsayisi
SRR_SPI_FLOOR_DB = -60.0      # SPI minimum spektral zemin (dB)

# ============================================
# MODULE_D: Özellik Mühendisliği Sabitleri
# Koç & Akbalık (2025): 120 boyutlu akustik özellik vektörü
# ============================================
AUDIO_SAMPLE_RATE = 44100      # Hz - Ses örnekleme hızı
N_MFCC = 13                    # MFCC katsayı sayısı
N_FFT = 2048                   # FFT pencere boyutu
HOP_LENGTH = 512               # FFT hop uzunluğu
FREQ_BAND_LOW = 50             # Hz - İlgi frekans bandı alt sınırı
FREQ_BAND_HIGH = 500           # Hz - İlgi frekans bandı üst sınırı

# 120 Özellik Vektörü Boyut Dağılımı:
#   Grup 1: MFCC İstatistikleri (mean/std/min/max × 13)  = 52
#   Grup 2: Delta MFCC İstatistikleri (mean/std × 13)    = 26
#   Grup 3: ZCR (mean, std)                               =  2
#   Grup 4: Spektral Özellikler                            = 15
#   Grup 5: Enerji Özellikleri (RMS + Log Energy)         =  4
#   Grup 6: Chroma (12 pitch sınıfı)                      = 12
#   Grup 7: Frekans & Rezonans (f2, f2_db, entropy)      =  3
#   Grup 8: Zaman Alanı (peak/crest/centroid/attack/decay) =  6
#   TOPLAM                                                 = 120
N_FEATURES = 120               # Toplam akustik özellik sayısı
N_FEATURES_V1 = 37             # Eski versiyon (geriye uyumluluk)
N_CHROMA = 12                  # Chroma pitch sınıfı sayısı
N_SPECTRAL_CONTRAST_BANDS = 7  # Spektral kontrast bant sayısı

# ============================================
# MODULE_E: Çıkarım Motoru Sabitleri
# ============================================
# Fiziksel doğrulama eşikleri
F2_RIPE_THRESHOLD = 134        # Hz - f2 < 134Hz → olgun karpuz (Yamamoto et al.)
MAGNITUDE_RIPE_THRESHOLD = 25  # dB - magnitude > 25dB → olgun karpuz

# Brix eşikleri (Qilin 19-datasets aralığına göre kalibre)
BRIX_IMMATURE_THRESHOLD = 10.0   # Brix < 10.0 → Olgunlaşmamış
BRIX_OVERRIPE_THRESHOLD = 11.5   # Brix > 11.5 → İçi Geçmiş

# Sınıflandırma etiketleri
CLASS_LABELS = {
    0: "Olgunlaşmamış (Immature)",
    1: "Olgun (Mature/Ripe)",
    2: "İçi Geçmiş (Over-mature/Hollow Heart)"
}

# KNN model parametreleri
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_METRIC = "minkowski"

# Random Forest parametreleri
# Az veride (19 karpuz) sig agac + min leaf overfitting'i sinirlar
RFC_N_ESTIMATORS = 300
RFC_MAX_DEPTH = 8
RFC_MIN_SAMPLES_LEAF = 3
RFC_MIN_SAMPLES_SPLIT = 5
RFC_RANDOM_STATE = 42

# ============================================
# MODULE_A: Gorsel Analiz Sabitleri
# ============================================
YOLO_MODEL_PATH = str(MODELS_DIR / "yolov11n.pt")

# MRD-YOLO: XuebinJing/Melon-Ripeness-Detection
# Hafif MobileNetV3 backbone + CoordAtt + VoVGSCSP + GSConv
MRD_YOLO_MODEL_PATH = str(PROJECT_ROOT / "model" / "MRD.pt")
MRD_YOLO_FALLBACK_PATH = str(PROJECT_ROOT / "mrd_yolo_repo" / "weights" / "MRD.pt")

# Modifiye ultralytics paketi (custom modüller için)
MRD_ULTRALYTICS_PATH = str(PROJECT_ROOT / "mrd_yolo_repo")

YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_NMS_IOU_THRESHOLD = 0.5
IMAGE_INPUT_SIZE = (640, 640)

# MRD-YOLO Sinif Haritalama (melon.yaml: nc=2)
MRD_CLASS_NAMES = {
    0: "Olgunlasmamis (Unripe)",
    1: "Olgun (Ripe)"
}
MRD_NUM_CLASSES = 2

# MRD-YOLO Dataset Yollari
MRD_REPO_DIR = PROJECT_ROOT / "mrd_yolo_repo"
MRD_DATASET_DIR = MRD_REPO_DIR / "dataset"
MRD_DATASET_YAML = MRD_DATASET_DIR / "melon.yaml"
MRD_DATASET_IMAGES = MRD_DATASET_DIR / "images"
MRD_DATASET_LABELS = MRD_DATASET_DIR / "labels"
MRD_TEST_IMAGES = MRD_DATASET_IMAGES / "test"

# ============================================
# Koc & Akbalik (2025) - Matematiksel Parametreler
# %96.04 mAP@0.5 Basari Oraninin Altyapisi
# ============================================

# --- Kayip Fonksiyonu Agirliklari ---
# L_total = lambda_loc * L_CIoU + lambda_cls * L_focal + lambda_conf * L_BCE
LOSS_LAMBDA_LOCALIZATION = 7.5      # CIoU kayip agirligi
LOSS_LAMBDA_CLASSIFICATION = 0.5    # Focal Loss agirligi
LOSS_LAMBDA_CONFIDENCE = 1.5        # BCE kayip agirligi

# --- Focal Loss Parametreleri ---
# L_focal = -alpha_t * (1 - p_t)^gamma * log(p_t)
FOCAL_LOSS_ALPHA = 0.25             # Azinlik sinif agirligi
FOCAL_LOSS_GAMMA = 2.0              # Odaklanma parametresi

# Sinif bazli alpha agirliklari
CLASS_ALPHA_WEIGHTS = {
    0: 0.35,    # Olgunlasmamis (~%30 veri)
    1: 0.25,    # Olgun (~%45 veri)
    2: 0.40     # Ici gecmis (~%25 veri)
}

# --- Optimizasyon Parametreleri (AdamW) ---
# theta_t = theta_{t-1} - lr * (m_hat/(sqrt(v_hat)+eps) + lambda*theta_{t-1})
OPTIMIZER_LR_INITIAL = 0.01         # Baslangic ogrenme orani
OPTIMIZER_LR_MIN = 0.0001           # Minimum ogrenme orani (Cosine Annealing sonu)
OPTIMIZER_BETA1 = 0.937             # AdamW birinci moment katsayisi
OPTIMIZER_BETA2 = 0.999             # AdamW ikinci moment katsayisi
OPTIMIZER_EPSILON = 1e-8            # Numerik kararlilik
OPTIMIZER_WEIGHT_DECAY = 0.0005     # L2 regularizasyon (overfitting onleme)

# --- Ogrenme Orani Cizelgesi (Cosine Annealing) ---
# lr_t = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(pi*t/T))
SCHEDULER_WARMUP_EPOCHS = 3         # Lineer warmup epoch sayisi
SCHEDULER_TOTAL_EPOCHS = 300        # Toplam egitim epoch sayisi
SCHEDULER_WARMUP_BIAS_LR = 0.1     # Warmup bias ogrenme orani

# --- Regularizasyon ---
DROPOUT_RATE = 0.3                  # Head katmaninda dropout
MIXUP_ALPHA = 0.1                   # MixUp augmentation alpha
EARLY_STOPPING_PATIENCE = 50        # Early stopping sabir (epoch)

# --- YOLOv11 Mimari Parametreleri ---
# Backbone: C2f (Cross Stage Partial with 2 convolutions - fused)
# Neck: PAN-FPN (Path Aggregation Network + Feature Pyramid Network)
# Head: Decoupled Head (Ayristirilmis siniflandirma + regresyon)
# Attention: CBAM (Channel + Spatial Attention)
YOLO_BACKBONE = "C2f"
YOLO_NECK = "PAN-FPN"
YOLO_HEAD = "Decoupled"
YOLO_ATTENTION = "CBAM"

# --- Performans Metrikleri ---
# %96.04 = mAP@0.5 (Mean Average Precision, IoU esik=0.5)
TARGET_MAP_50 = 0.9604              # Hedef mAP@0.5
TARGET_MAP_75 = 0.83                # Tahmini mAP@0.75
TARGET_MAP_50_95 = 0.70             # Tahmini mAP@0.5:0.95
OPTIMAL_CONFIDENCE_THRESHOLD = 0.45 # F1-Score optimum esigi

# ============================================
# Hacim & Kutle Tahmini - Koc (2007) Parametreleri
# ============================================
WATERMELON_DENSITY = 0.98      # g/cm^3 - Karpuz yogunlugu (yaklasik su)
WATERMELON_DENSITY_MAX = 1.0   # g/cm^3 - Ust sinir
PI = 3.141592653589793

# Disk Metodu Parametreleri
DISK_SLICE_WIDTH_PX = 2        # Piksel - dilim genisligi
DISK_DEPTH_RATIO = 0.85        # 2D'den derinlik tahmini orani (depth/width)

# Goruntu On-Isleme Parametreleri
MORPH_KERNEL_SIZE = 15         # Morfolojik cekirdek boyutu
ADAPTIVE_BLOCK_SIZE = 51       # Adaptive threshold blok boyutu
ADAPTIVE_C = 5                 # Adaptive threshold sabiti
CANNY_LOW = 30                 # Canny alt esik
CANNY_HIGH = 100               # Canny ust esik
MIN_CONTOUR_AREA_RATIO = 0.01  # Minimum kontur alan orani

# MARE Hedefi (Koc, 2007)
VOLUME_MARE_TARGET = 8.0       # % - Ortalama Mutlak Bagil Hata hedefi

# ============================================
# Hollow Heart (İçi Geçmiş) Tespit Parametreleri
# ============================================
# Akustik imza tabanlı boşluk tespiti
# Referans: Hollow heart karpuzlarda rezonans bölünmesi,
# hızlı sönüm ve spektral yayılma gözlemlenir.

# Dual-peak (çift tepe) rezonans analizi
HH_DUAL_PEAK_MIN_DISTANCE_HZ = 15.0     # İki tepe arasındaki minimum frekans farkı (Hz)
HH_DUAL_PEAK_MAX_DISTANCE_HZ = 80.0     # İki tepe arasındaki maksimum frekans farkı (Hz)
HH_DUAL_PEAK_PROMINENCE_DB = 3.0        # Tepe belirginliği eşiği (dB)
HH_DUAL_PEAK_VALLEY_DEPTH_DB = 2.0      # İki tepe arasındaki vadi derinliği (dB)

# Sönüm (damping) analizi
HH_DAMPING_RATIO_THRESHOLD = 0.12       # Hollow heart sönüm oranı eşiği (zeta)
HH_DECAY_RATE_FAST_THRESHOLD = 25.0     # Hızlı sönüm eşiği (1/s)
HH_DECAY_IRREGULARITY_THRESHOLD = 0.3   # Sönüm düzensizliği eşiği (normalize)

# Spektral yayılma (spread) analizi
HH_SPECTRAL_SPREAD_THRESHOLD = 0.65     # Spektral yayılma indeksi eşiği (0-1)
HH_SPECTRAL_ENTROPY_HIGH = 0.75         # Yüksek entropi eşiği (hollow → yüksek entropi)
HH_SPECTRAL_FLATNESS_HIGH = 0.4         # Yüksek düzlük eşiği (hollow → düz spektrum)

# Cepstral analiz
HH_CEPSTRAL_PEAK_PROMINENCE = 0.15      # Cepstral tepe belirginliği eşiği
HH_QUEFRENCY_RANGE = (0.002, 0.020)     # İlgi quefrency aralığı (saniye)

# Harmonik-gürültü oranı (HNR)
HH_HNR_LOW_THRESHOLD = 5.0              # Düşük HNR eşiği (dB) - hollow → düşük HNR

# Hollow Heart tespit ağırlıkları (çok-göstergeli karar)
HH_WEIGHT_DUAL_PEAK = 0.30              # Çift tepe rezonans ağırlığı
HH_WEIGHT_DAMPING = 0.20                # Sönüm analizi ağırlığı
HH_WEIGHT_SPECTRAL = 0.20               # Spektral yayılma ağırlığı
HH_WEIGHT_CEPSTRAL = 0.15               # Cepstral analiz ağırlığı
HH_WEIGHT_HNR = 0.15                    # HNR ağırlığı

# Nihai hollow heart karar eşiği
HH_DETECTION_THRESHOLD = 0.45           # Toplam HH skoru > bu değer → hollow heart
HH_HIGH_CONFIDENCE_THRESHOLD = 0.65     # Yüksek güvenli HH tespiti

# ============================================
# Late Fusion Ağırlıkları
# ============================================
FUSION_VISUAL_WEIGHT = 0.35
FUSION_ACOUSTIC_WEIGHT = 0.45
FUSION_HAPTIC_WEIGHT = 0.20

# ============================================
# TFLite Dönüştürme
# ============================================
TFLITE_OUTPUT_PATH = str(MODELS_DIR / "fusion_model.tflite")
TFLITE_INT8_OUTPUT_PATH = str(MODELS_DIR / "fusion_model_int8.tflite")
TFLITE_FLOAT16_OUTPUT_PATH = str(MODELS_DIR / "fusion_model_fp16.tflite")

# Fusion model girdi boyutlari
TFLITE_ACOUSTIC_INPUT_DIM = 120        # Koc & Akbalik (2025) akustik ozellik
TFLITE_VISUAL_INPUT_DIM = 11           # Gorsel ozellik boyutu
TFLITE_HAPTIC_INPUT_DIM = 7            # Haptik ozellik boyutu
TFLITE_HH_INPUT_DIM = 8               # Hollow Heart ozellik boyutu
TFLITE_N_CLASSES = 3                   # Sinif sayisi

# INT8 Quantization parametreleri
TFLITE_INT8_REPRESENTATIVE_SAMPLES = 200  # Kalibrasyon icin ornek sayisi
TFLITE_INT8_CALIBRATION_SEED = 42

# Model metadata
TFLITE_MODEL_NAME = "WatermelonRipeness_FusionModel"
TFLITE_MODEL_VERSION = "2.0.0"
TFLITE_MODEL_AUTHOR = "Koc & Akbalik (2025)"
TFLITE_MODEL_DESCRIPTION = (
    "Multi-modal late fusion model for watermelon ripeness classification. "
    "Inputs: acoustic(120), visual(11), haptic(7), hollow_heart(8). "
    "Outputs: 3-class probability (immature, ripe, hollow_heart)."
)

# ============================================
# Crowdsourcing / Kullanici Geri Bildirim
# ============================================
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DB_PATH = str(FEEDBACK_DIR / "feedback_log.json")
FEEDBACK_EXPORT_PATH = str(FEEDBACK_DIR / "feedback_export.csv")

# Geri bildirim sinif isimleri (Flutter UI icin)
FEEDBACK_CLASS_OPTIONS = {
    0: {"tr": "Olgunlasmamis", "en": "Immature/Unripe"},
    1: {"tr": "Olgun", "en": "Mature/Ripe"},
    2: {"tr": "Ici Gecmis / Bosluklu", "en": "Over-mature/Hollow Heart"},
}

# Geri bildirim toplama politikasi
FEEDBACK_MIN_CONFIDENCE_FOR_PROMPT = 0.7   # Bu guvenden dusukse kullaniciya sor
FEEDBACK_MAX_SAMPLES_PER_SESSION = 50      # Oturum basina max geri bildirim
FEEDBACK_RETRAIN_THRESHOLD = 100           # Bu kadar feedback birikince yeniden egit

# ============================================
# MODULE_B: Basinc Analizi / Temas Denetimi
# ============================================
# Ivmeolcer tabanli temas kalitesi degerlendirmesi
# Flutter ContactChecker ile senkron esik degerleri

# Yercekimi sapmasi esikleri (m/s^2)
PRESSURE_LIGHT_THRESHOLD = 0.3     # Hafif temas baslangici
PRESSURE_GOOD_THRESHOLD = 0.8      # Iyi (ideal) temas alt siniri
PRESSURE_STRONG_THRESHOLD = 2.5    # Fazla basinc uyari esigi

# Kararlilik degerlendirmesi
PRESSURE_STABILITY_THRESHOLD = 0.15  # Magnitude standart sapma esigi
PRESSURE_WINDOW_SIZE = 50           # Analiz penceresi (ornek sayisi, ~0.5s @ 100Hz)

# Yercekimi referansi
PRESSURE_GRAVITY_DEFAULT = 9.81    # Varsayilan yercekimi (m/s^2)

# Dusey geciren filtre kesim frekansi (yercekimi ayristirma)
PRESSURE_LP_CUTOFF = 0.5           # Hz - Yercekimi bant genisligi

# Temas kalitesi puanlama
PRESSURE_QUALITY_SCORES = {
    "no_contact": 0.0,
    "light_contact": 0.3,
    "good_contact": 0.8,
    "too_strong": 0.5,
}

# Haptik ozellik vektoru boyutu (Late Fusion girdi)
HAPTIC_FEATURE_DIM = 7             # PressureAnalyzer.extract_pressure_features() ciktisi

# ============================================
# Xigua Full Fusion Model (Audio + Image)
# ============================================
XIGUA_PTH_PATH = str(QILIN_DATASETS_DIR / "watermelon_dataset" / "xigua_pytorch.pth")
XIGUA_FUSION_ONNX_PATH = str(MODELS_DIR / "xigua_fusion.onnx")
XIGUA_FUSION_TFLITE_FP16_PATH = str(MODELS_DIR / "xigua_fusion_fp16.tflite")
XIGUA_FUSION_TFLITE_INT8_PATH = str(MODELS_DIR / "xigua_fusion_int8.tflite")
XIGUA_AUDIO_LENGTH = 16000           # 16k sample
XIGUA_AUDIO_SEQ_LEN = 160            # LSTM zaman adımı
XIGUA_AUDIO_FEAT_DIM = 100           # LSTM girdi boyutu
XIGUA_IMAGE_SIZE = 224               # ResNet50 girdi boyutu
XIGUA_LSTM_HIDDEN = 128
XIGUA_RESNET_FEAT = 2048
XIGUA_FC1_OUT = 64

# ============================================
# Sinyal Kalitesi & Dinamik Füzyon
# ============================================
# Sinyal kalitesi eşikleri
SQ_BRIGHTNESS_IDEAL_MIN = 0.3
SQ_BRIGHTNESS_IDEAL_MAX = 0.7
SQ_SNR_GOOD_THRESHOLD = 20.0        # dB
SQ_CLIPPING_MAX_RATIO = 0.01        # %1
SQ_IMU_RMS_MIN = 0.05               # m/s^2

# Dinamik ağırlık sabitleri
DYNAMIC_FUSION_MIN_WEIGHT = 0.05     # Hiçbir modalite tamamen göz ardı edilmez
DYNAMIC_FUSION_XIGUA_BASE_WEIGHT = 0.20  # Xigua erken füzyon baz ağırlığı

