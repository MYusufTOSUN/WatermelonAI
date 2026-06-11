"""
TFLite Donusturme ve Quantization Pipeline

Egitilmis fusion modelini TensorFlow Lite formatina donusturur.
Mobil cihazda (Flutter) calistirilmak uzere optimize eder.

Desteklenen quantization turleri:
  - Float32 (tam hassasiyet, buyuk boyut)
  - Float16 (yarim hassasiyet, ~%50 boyut azalma)
  - INT8 (tam quantization, ~%75 boyut azalma, en hizli inference)

Hollow Heart entegrasyonu:
  - Fusion modeline HH feature (8 boyut) eklendi
  - Toplam girdi: acoustic(120) + visual(11) + haptic(7) + hh(8) = 146

Flutter Uyumluluk:
  - TFLite metadata eklenir (model bilgisi, girdi/cikti tanimlari)
  - tflite_flutter paketi ile direkt kullanilabilir

Kullanim:
    python -m backend.pipeline.tflite_converter
    python -m backend.pipeline.tflite_converter --quantize int8
    python -m backend.pipeline.tflite_converter --quantize float16
    python -m backend.pipeline.tflite_converter --quantize all
    python -m backend.pipeline.tflite_converter --benchmark
"""

import numpy as np
import os
import json
import time
from typing import Dict, Optional, Tuple, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    MODELS_DIR,
    TFLITE_OUTPUT_PATH,
    TFLITE_INT8_OUTPUT_PATH,
    TFLITE_FLOAT16_OUTPUT_PATH,
    TFLITE_ACOUSTIC_INPUT_DIM,
    TFLITE_VISUAL_INPUT_DIM,
    TFLITE_HAPTIC_INPUT_DIM,
    TFLITE_HH_INPUT_DIM,
    TFLITE_N_CLASSES,
    TFLITE_INT8_REPRESENTATIVE_SAMPLES,
    TFLITE_INT8_CALIBRATION_SEED,
    TFLITE_MODEL_NAME,
    TFLITE_MODEL_VERSION,
    TFLITE_MODEL_AUTHOR,
    TFLITE_MODEL_DESCRIPTION,
    CLASS_LABELS,
)


# ============================================
# MODEL OLUSTURMA
# ============================================

def create_fusion_keras_model(
    acoustic_input_dim: int = TFLITE_ACOUSTIC_INPUT_DIM,
    visual_input_dim: int = TFLITE_VISUAL_INPUT_DIM,
    haptic_input_dim: int = TFLITE_HAPTIC_INPUT_DIM,
    hh_input_dim: int = TFLITE_HH_INPUT_DIM,
    n_classes: int = TFLITE_N_CLASSES,
    include_hh_branch: bool = True
):
    """
    Late Fusion Keras modeli olusturur.

    Dort modaliteden (akustik, gorsel, haptik, hollow heart)
    gelen ozellikleri birlestirip siniflandiran bir neural network.

    Koc & Akbalik (2025): 120 boyutlu akustik ozellik vektoru
    Hollow Heart: 8 boyutlu HH gosterge vektoru

    Args:
        acoustic_input_dim: Akustik ozellik boyutu (120)
        visual_input_dim: Gorsel ozellik boyutu (11)
        haptic_input_dim: Haptik ozellik boyutu (7)
        hh_input_dim: Hollow Heart ozellik boyutu (8)
        n_classes: Sinif sayisi (3)
        include_hh_branch: Hollow Heart dalini dahil et

    Returns:
        Keras model
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # --- Akustik dal (120 giris -> derin ag) ---
    acoustic_input = keras.Input(shape=(acoustic_input_dim,), name='acoustic_input')
    acoustic_x = layers.Dense(128, activation='relu', name='acoustic_dense1')(acoustic_input)
    acoustic_x = layers.BatchNormalization(name='acoustic_bn1')(acoustic_x)
    acoustic_x = layers.Dropout(0.3, name='acoustic_drop1')(acoustic_x)
    acoustic_x = layers.Dense(64, activation='relu', name='acoustic_dense2')(acoustic_x)
    acoustic_x = layers.BatchNormalization(name='acoustic_bn2')(acoustic_x)
    acoustic_x = layers.Dropout(0.2, name='acoustic_drop2')(acoustic_x)
    acoustic_x = layers.Dense(32, activation='relu', name='acoustic_dense3')(acoustic_x)

    # --- Gorsel dal (11 giris) ---
    visual_input = keras.Input(shape=(visual_input_dim,), name='visual_input')
    visual_x = layers.Dense(32, activation='relu', name='visual_dense1')(visual_input)
    visual_x = layers.BatchNormalization(name='visual_bn1')(visual_x)
    visual_x = layers.Dropout(0.3, name='visual_drop1')(visual_x)
    visual_x = layers.Dense(16, activation='relu', name='visual_dense2')(visual_x)

    # --- Haptik dal (7 giris) ---
    haptic_input = keras.Input(shape=(haptic_input_dim,), name='haptic_input')
    haptic_x = layers.Dense(32, activation='relu', name='haptic_dense1')(haptic_input)
    haptic_x = layers.BatchNormalization(name='haptic_bn1')(haptic_x)
    haptic_x = layers.Dropout(0.3, name='haptic_drop1')(haptic_x)
    haptic_x = layers.Dense(16, activation='relu', name='haptic_dense2')(haptic_x)

    inputs = [acoustic_input, visual_input, haptic_input]
    branches = [acoustic_x, visual_x, haptic_x]

    # --- Hollow Heart dal (8 giris, opsiyonel) ---
    if include_hh_branch:
        hh_input = keras.Input(shape=(hh_input_dim,), name='hh_input')
        hh_x = layers.Dense(16, activation='relu', name='hh_dense1')(hh_input)
        hh_x = layers.BatchNormalization(name='hh_bn1')(hh_x)
        hh_x = layers.Dense(8, activation='relu', name='hh_dense2')(hh_x)
        inputs.append(hh_input)
        branches.append(hh_x)

    # --- Late Fusion - Birlestirme ---
    merged = layers.Concatenate(name='fusion_concat')(branches)
    merged = layers.Dense(64, activation='relu', name='fusion_dense1')(merged)
    merged = layers.BatchNormalization(name='fusion_bn1')(merged)
    merged = layers.Dropout(0.4, name='fusion_drop1')(merged)
    merged = layers.Dense(32, activation='relu', name='fusion_dense2')(merged)
    merged = layers.Dropout(0.2, name='fusion_drop2')(merged)

    # --- Cikis katmani ---
    output = layers.Dense(n_classes, activation='softmax', name='output')(merged)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name='watermelon_fusion_model'
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================
# EGITIM
# ============================================

def train_fusion_model(
    model,
    X_acoustic: np.ndarray,
    X_visual: np.ndarray,
    X_haptic: np.ndarray,
    y: np.ndarray,
    X_hh: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2
):
    """
    Fusion modelini egitir.

    Args:
        model: Keras model
        X_acoustic, X_visual, X_haptic: Modalite ozellikleri
        y: Etiketler
        X_hh: Hollow Heart ozellikleri (opsiyonel)
        epochs: Epoch sayisi
        batch_size: Batch boyutu
        validation_split: Dogrulama seti orani

    Returns:
        Egitim gecmisi
    """
    import tensorflow as tf

    # Early stopping + LR reducer
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5
        )
    ]

    # Girdi listesi
    inputs = [X_acoustic, X_visual, X_haptic]
    if X_hh is not None:
        inputs.append(X_hh)

    history = model.fit(
        inputs, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ============================================
# SENTETIK VERI URETIMI
# ============================================

def generate_synthetic_fusion_data(
    n_samples: int = 500,
    include_hh: bool = True
) -> tuple:
    """
    Koc & Akbalik (2025) uyumlu sentetik fusion verisi olusturur.

    120 boyutlu akustik + 8 boyutlu HH ozellik vektoru ile calisir.

    Args:
        n_samples: Toplam ornek sayisi
        include_hh: Hollow Heart ozelliklerini dahil et

    Returns:
        (X_acoustic, X_visual, X_haptic, X_hh, y) veya
        (X_acoustic, X_visual, X_haptic, y) (include_hh=False)
    """
    np.random.seed(42)

    n_per_class = n_samples // 3

    X_acoustic_list = []
    X_visual_list = []
    X_haptic_list = []
    X_hh_list = []
    y_list = []

    for class_id in range(3):
        for _ in range(n_per_class):
            # Akustik ozellikler (120 boyut - Koc & Akbalik 2025)
            acoustic = np.random.randn(TFLITE_ACOUSTIC_INPUT_DIM) * 0.5

            # Sinif bazli fiziksel parametreler
            if class_id == 0:  # Olgunlasmamis
                acoustic[111] = np.random.uniform(160, 250)   # f2
                acoustic[112] = np.random.uniform(20, 30)     # f2_db
                acoustic[113] = np.random.uniform(0.6, 0.8)   # entropy
                acoustic[78] = np.random.uniform(0.08, 0.15)  # zcr_mean
                acoustic[118] = np.random.uniform(15, 30)     # decay_rate
                # HH gostergeleri: dusuk (normal)
                hh = np.array([
                    np.random.uniform(0.0, 0.2),   # hh_score
                    np.random.uniform(0.0, 0.2),   # dual_peak
                    np.random.uniform(0.0, 0.2),   # damping
                    np.random.uniform(0.0, 0.2),   # spectral
                    np.random.uniform(0.0, 0.3),   # cepstral
                    np.random.uniform(0.0, 0.2),   # hnr_score
                    np.random.uniform(8, 15),       # hnr_db
                    np.random.uniform(0.02, 0.06),  # damping_ratio
                ])
            elif class_id == 1:  # Olgun
                acoustic[111] = np.random.uniform(80, 145)
                acoustic[112] = np.random.uniform(26, 35)
                acoustic[113] = np.random.uniform(0.3, 0.6)
                acoustic[78] = np.random.uniform(0.04, 0.09)
                acoustic[118] = np.random.uniform(5, 15)
                # HH gostergeleri: dusuk (normal)
                hh = np.array([
                    np.random.uniform(0.0, 0.25),
                    np.random.uniform(0.0, 0.15),
                    np.random.uniform(0.0, 0.2),
                    np.random.uniform(0.0, 0.15),
                    np.random.uniform(0.0, 0.25),
                    np.random.uniform(0.0, 0.15),
                    np.random.uniform(10, 20),
                    np.random.uniform(0.03, 0.08),
                ])
            else:  # Ici gecmis / Hollow Heart
                acoustic[111] = np.random.uniform(50, 90)
                acoustic[112] = np.random.uniform(15, 25)
                acoustic[113] = np.random.uniform(0.7, 0.95)
                acoustic[78] = np.random.uniform(0.01, 0.05)
                acoustic[118] = np.random.uniform(20, 50)
                # HH gostergeleri: yuksek (hollow heart)
                hh = np.array([
                    np.random.uniform(0.5, 0.95),  # hh_score
                    np.random.uniform(0.4, 0.9),   # dual_peak
                    np.random.uniform(0.5, 0.9),   # damping
                    np.random.uniform(0.4, 0.85),  # spectral
                    np.random.uniform(0.3, 0.8),   # cepstral
                    np.random.uniform(0.5, 0.9),   # hnr_score
                    np.random.uniform(-5, 5),       # hnr_db (dusuk)
                    np.random.uniform(0.12, 0.3),  # damping_ratio (yuksek)
                ])

            # Gorsel ozellikler (11 boyut)
            visual = np.random.rand(TFLITE_VISUAL_INPUT_DIM) * 0.5
            visual[-1] = class_id / 2.0 + np.random.randn() * 0.1

            # Haptik ozellikler (7 boyut)
            haptic = np.random.randn(TFLITE_HAPTIC_INPUT_DIM) * 0.3
            haptic[0] = acoustic[111] + np.random.randn() * 10

            X_acoustic_list.append(acoustic)
            X_visual_list.append(visual)
            X_haptic_list.append(haptic)
            X_hh_list.append(hh)
            y_list.append(class_id)

    X_acoustic = np.array(X_acoustic_list, dtype=np.float32)
    X_visual = np.array(X_visual_list, dtype=np.float32)
    X_haptic = np.array(X_haptic_list, dtype=np.float32)
    X_hh = np.array(X_hh_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Karistir
    idx = np.random.permutation(len(y))
    X_acoustic = X_acoustic[idx]
    X_visual = X_visual[idx]
    X_haptic = X_haptic[idx]
    X_hh = X_hh[idx]
    y = y[idx]

    if include_hh:
        return X_acoustic, X_visual, X_haptic, X_hh, y
    else:
        return X_acoustic, X_visual, X_haptic, y


# ============================================
# INT8 REPRESENTATIVE DATASET GENERATOR
# ============================================

def create_representative_dataset(
    X_acoustic: np.ndarray,
    X_visual: np.ndarray,
    X_haptic: np.ndarray,
    X_hh: Optional[np.ndarray] = None,
    n_samples: int = TFLITE_INT8_REPRESENTATIVE_SAMPLES
):
    """
    INT8 quantization icin representative dataset uretir.

    TFLite INT8 donusumu, modelin tamsayi agirliklarini kalibre
    etmek icin gercek veri dagilimini temsil eden orneklere ihtiyac duyar.

    Bu fonksiyon, egitim verisinden rastgele ornekler secerek
    bir Python generator dondurur.

    Args:
        X_acoustic, X_visual, X_haptic: Egitim verileri
        X_hh: Hollow Heart ozellikleri (opsiyonel)
        n_samples: Kalibrasyon ornegi sayisi

    Returns:
        Generator fonksiyonu (TFLite converter icin)
    """
    np.random.seed(TFLITE_INT8_CALIBRATION_SEED)
    n_available = len(X_acoustic)
    n_samples = min(n_samples, n_available)
    indices = np.random.choice(n_available, size=n_samples, replace=False)

    def representative_data_gen():
        for i in indices:
            sample = [
                X_acoustic[i:i+1].astype(np.float32),
                X_visual[i:i+1].astype(np.float32),
                X_haptic[i:i+1].astype(np.float32),
            ]
            if X_hh is not None:
                sample.append(X_hh[i:i+1].astype(np.float32))
            yield sample

    return representative_data_gen


# ============================================
# TFLITE DONUSTURME
# ============================================

def convert_to_tflite(
    model,
    output_path: str = None,
    quantize: str = "float16",
    representative_dataset=None
) -> Dict[str, object]:
    """
    Keras modelini TFLite formatina donusturur.

    Desteklenen quantization turleri:
      - "none"    : Float32 (tam hassasiyet)
      - "float16" : Float16 (yarim hassasiyet, ~%50 kucultme)
      - "int8"    : INT8 (tam quantization, ~%75 kucultme, en hizli)

    Args:
        model: Egitilmis Keras model
        output_path: Cikis dosya yolu
        quantize: Quantization turu ("none", "float16", "int8")
        representative_dataset: INT8 icin kalibrasyon verisi generator

    Returns:
        Donusturme sonuclari (boyut, yol, tur)
    """
    import tensorflow as tf

    if output_path is None:
        if quantize == "int8":
            output_path = TFLITE_INT8_OUTPUT_PATH
        elif quantize == "float16":
            output_path = TFLITE_FLOAT16_OUTPUT_PATH
        else:
            output_path = TFLITE_OUTPUT_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # SavedModel olarak kaydet (gecici)
    saved_model_dir = os.path.join(os.path.dirname(output_path), "saved_model_temp")
    try:
        model.export(saved_model_dir)
    except AttributeError:
        model.save(saved_model_dir)

    # TFLite'a donustur
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize == "int8":
        # Hybrid INT8: Dense katmanlar INT8, CONCATENATION float'ta calisir
        # (Pure INT8 CONCATENATION'da boyut uyumsuzlugu veriyor)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        # Girdi/cikti float kalsin (CONCATENATION uyumlulugu)
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
        else:
            print("  [!] INT8 icin representative dataset yok, fallback kullaniliyor")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantize == "float16":
        # Float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantize == "dynamic":
        # Dynamic range quantization (INT8 agirliklari, Float girdi/cikti)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # else: "none" — Float32 (varsayilan)

    # Donusturme
    start_time = time.time()
    tflite_model = converter.convert()
    convert_time = time.time() - start_time

    # Dosyaya yaz
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Gecici dosyalari temizle
    import shutil
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    size_bytes = os.path.getsize(output_path)
    size_mb = size_bytes / (1024 * 1024)

    result = {
        "output_path": output_path,
        "quantization": quantize,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 4),
        "convert_time_s": round(convert_time, 2),
    }

    print(f"  [TFLite] Donusturuldu: {output_path}")
    print(f"  [TFLite] Quantization: {quantize}")
    print(f"  [TFLite] Boyut: {size_mb:.4f} MB ({size_bytes:,} bytes)")
    print(f"  [TFLite] Donusturme suresi: {convert_time:.2f}s")

    return result


# ============================================
# TUM QUANTIZATION TURLERINI URET
# ============================================

def convert_all_quantizations(
    model,
    representative_dataset=None
) -> Dict[str, Dict]:
    """
    Modeli tum quantization turleriyle donusturur.

    Float32, Float16, Dynamic ve INT8 versiyonlarini uretir.

    Returns:
        Her quantization turu icin sonuc dict'i
    """
    results = {}

    print("\n" + "=" * 60)
    print("  TUM QUANTIZATION TURLERI URETILIYOR")
    print("=" * 60)

    # 1) Float32 (referans)
    print("\n  --- [1/4] Float32 (Tam Hassasiyet) ---")
    results["float32"] = convert_to_tflite(
        model,
        output_path=TFLITE_OUTPUT_PATH.replace(".tflite", "_fp32.tflite"),
        quantize="none"
    )

    # 2) Float16
    print("\n  --- [2/4] Float16 (Yarim Hassasiyet) ---")
    results["float16"] = convert_to_tflite(
        model,
        output_path=TFLITE_FLOAT16_OUTPUT_PATH,
        quantize="float16"
    )

    # 3) Dynamic Range
    print("\n  --- [3/4] Dynamic Range (INT8 agirliklari) ---")
    results["dynamic"] = convert_to_tflite(
        model,
        output_path=TFLITE_OUTPUT_PATH.replace(".tflite", "_dynamic.tflite"),
        quantize="dynamic"
    )

    # 4) INT8 Full Quantization
    print("\n  --- [4/4] INT8 Full Quantization ---")
    try:
        results["int8"] = convert_to_tflite(
            model,
            output_path=TFLITE_INT8_OUTPUT_PATH,
            quantize="int8",
            representative_dataset=representative_dataset
        )
    except Exception as e:
        print(f"  [!] INT8 donusturme hatasi: {e}")
        print("  [!] Dynamic range ile fallback yapiliyor...")
        results["int8"] = convert_to_tflite(
            model,
            output_path=TFLITE_INT8_OUTPUT_PATH,
            quantize="dynamic"
        )
        results["int8"]["note"] = f"INT8 fallback to dynamic: {str(e)}"

    return results


# ============================================
# MODEL BENCHMARK
# ============================================

def benchmark_tflite_model(
    tflite_path: str,
    n_iterations: int = 100,
    include_hh: bool = True
) -> Dict[str, object]:
    """
    TFLite modelinin inference performansini olcer.

    Args:
        tflite_path: TFLite model yolu
        n_iterations: Olcum tekrar sayisi
        include_hh: HH girdisi dahil mi

    Returns:
        Benchmark sonuclari (latency, throughput)
    """
    import tensorflow as tf

    if not os.path.exists(tflite_path):
        return {"error": f"Model bulunamadi: {tflite_path}"}

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Rastgele girdi uret
    test_inputs = {}
    for inp in input_details:
        shape = inp['shape']
        dtype = inp['dtype']
        if dtype == np.int8:
            data = np.random.randint(-128, 127, size=shape).astype(np.int8)
        elif dtype == np.uint8:
            data = np.random.randint(0, 255, size=shape).astype(np.uint8)
        else:
            data = np.random.randn(*shape).astype(np.float32)
        test_inputs[inp['index']] = data

    # Isinma (warmup)
    for _ in range(10):
        for idx, data in test_inputs.items():
            interpreter.set_tensor(idx, data)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        for idx, data in test_inputs.items():
            interpreter.set_tensor(idx, data)

        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    # Cikti bilgisi
    output_data = interpreter.get_tensor(output_details[0]['index'])

    result = {
        "model_path": tflite_path,
        "model_size_mb": os.path.getsize(tflite_path) / (1024 * 1024),
        "n_inputs": len(input_details),
        "n_outputs": len(output_details),
        "input_shapes": [inp['shape'].tolist() for inp in input_details],
        "input_dtypes": [str(inp['dtype']) for inp in input_details],
        "output_shape": output_details[0]['shape'].tolist(),
        "output_dtype": str(output_details[0]['dtype']),
        "n_iterations": n_iterations,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "throughput_fps": float(1000.0 / np.mean(latencies)),
    }

    return result


def benchmark_all_models(model_dir: str = None) -> Dict[str, Dict]:
    """
    Tum TFLite modellerini benchmark eder ve karsilastirir.

    Returns:
        Her model icin benchmark sonuclari
    """
    import glob

    if model_dir is None:
        model_dir = str(MODELS_DIR)

    tflite_files = glob.glob(os.path.join(model_dir, "*.tflite"))

    if not tflite_files:
        print(f"  [!] {model_dir} dizininde TFLite modeli bulunamadi")
        return {}

    results = {}
    print("\n" + "=" * 60)
    print("  TFLITE MODEL BENCHMARK KARSILASTIRMASI")
    print("=" * 60)

    for tflite_path in sorted(tflite_files):
        model_name = os.path.basename(tflite_path)
        print(f"\n  --- {model_name} ---")

        bench = benchmark_tflite_model(tflite_path)
        results[model_name] = bench

        if "error" not in bench:
            print(f"    Boyut: {bench['model_size_mb']:.4f} MB")
            print(f"    Latency: {bench['latency_ms']['mean']:.2f} +/- {bench['latency_ms']['std']:.2f} ms")
            print(f"    P95: {bench['latency_ms']['p95']:.2f} ms")
            print(f"    Throughput: {bench['throughput_fps']:.1f} FPS")
            print(f"    Girdiler: {bench['input_dtypes']}")
        else:
            print(f"    HATA: {bench['error']}")

    # Karsilastirma tablosu
    if len(results) > 1:
        print("\n  " + "-" * 60)
        print(f"  {'Model':<35} {'Boyut(MB)':>10} {'Latency(ms)':>12} {'FPS':>8}")
        print("  " + "-" * 60)

        # Referans boyut (en buyuk model)
        ref_size = max(r.get('model_size_mb', 0) for r in results.values() if "error" not in r)

        for name, bench in sorted(results.items()):
            if "error" in bench:
                continue
            size = bench['model_size_mb']
            lat = bench['latency_ms']['mean']
            fps = bench['throughput_fps']
            compression = (1.0 - size / ref_size) * 100 if ref_size > 0 else 0
            print(f"  {name:<35} {size:>9.4f} {lat:>11.2f} {fps:>7.1f}  ({compression:+.0f}%)")

        print("  " + "-" * 60)

    return results


# ============================================
# METADATA EKLEME (FLUTTER UYUMLULUK)
# ============================================

def embed_metadata(tflite_path: str, metadata: Dict = None) -> str:
    """
    TFLite modeline metadata ekler.

    Flutter tflite_flutter paketi ile girdi/cikti tanimlarini
    otomatik okuyabilmek icin metadata embedder kullanir.

    Metadata JSON olarak modelin yanina kaydedilir.

    Args:
        tflite_path: TFLite model dosya yolu
        metadata: Ek metadata (opsiyonel)

    Returns:
        Metadata JSON dosya yolu
    """
    if metadata is None:
        metadata = {}

    # Model bilgisi
    model_metadata = {
        "model_name": TFLITE_MODEL_NAME,
        "model_version": TFLITE_MODEL_VERSION,
        "model_author": TFLITE_MODEL_AUTHOR,
        "model_description": TFLITE_MODEL_DESCRIPTION,
        "model_file": os.path.basename(tflite_path),
        "model_size_bytes": os.path.getsize(tflite_path) if os.path.exists(tflite_path) else 0,
        "inputs": {
            "acoustic_input": {
                "index": 0,
                "shape": [1, TFLITE_ACOUSTIC_INPUT_DIM],
                "dtype": "float32",
                "description": "120-dim acoustic feature vector (Koc & Akbalik 2025)",
                "features": [
                    "MFCC stats (52): mean/std/min/max x 13",
                    "Delta MFCC stats (26): mean/std x 13",
                    "ZCR (2): mean, std",
                    "Spectral (15): centroid/bw/rolloff/flatness/contrast",
                    "Energy (4): RMS + Log Energy",
                    "Chroma (12): 12 pitch classes",
                    "Frequency (3): f2, f2_db, entropy",
                    "Time-domain (6): peak/crest/centroid/attack/decay/duration",
                ]
            },
            "visual_input": {
                "index": 1,
                "shape": [1, TFLITE_VISUAL_INPUT_DIM],
                "dtype": "float32",
                "description": "Visual features from MRD-YOLO / heuristic analysis"
            },
            "haptic_input": {
                "index": 2,
                "shape": [1, TFLITE_HAPTIC_INPUT_DIM],
                "dtype": "float32",
                "description": "Haptic response features (LRA vibration analysis)"
            },
            "hh_input": {
                "index": 3,
                "shape": [1, TFLITE_HH_INPUT_DIM],
                "dtype": "float32",
                "description": "8-dim Hollow Heart indicator vector",
                "features": [
                    "hh_score (0-1): Overall hollow heart score",
                    "dual_peak_score (0-1): Split resonance indicator",
                    "damping_score (0-1): Fast/irregular damping indicator",
                    "spectral_score (0-1): Spectral spread indicator",
                    "cepstral_score (0-1): Cepstral periodicity loss",
                    "hnr_score (0-1): Harmonic-to-noise ratio indicator",
                    "hnr_db: Combined HNR value in dB",
                    "damping_ratio: Estimated damping ratio (zeta)",
                ]
            }
        },
        "outputs": {
            "output": {
                "index": 0,
                "shape": [1, TFLITE_N_CLASSES],
                "dtype": "float32",
                "description": "3-class probability vector",
                "classes": {
                    str(k): v for k, v in CLASS_LABELS.items()
                }
            }
        },
        "flutter_integration": {
            "package": "tflite_flutter: ^0.10.4",
            "usage": (
                "final interpreter = await Interpreter.fromAsset('fusion_model.tflite');\n"
                "var input = [acousticFeatures, visualFeatures, hapticFeatures, hhFeatures];\n"
                "var output = List.filled(1, List.filled(3, 0.0));\n"
                "interpreter.runForMultipleInputs(input, {0: output});"
            ),
        },
    }

    model_metadata.update(metadata)

    # JSON olarak kaydet
    json_path = tflite_path.replace(".tflite", "_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)

    print(f"  [Metadata] Kaydedildi: {json_path}")
    return json_path


# ============================================
# TFLITE MODEL DOGRULAMA
# ============================================

def validate_tflite_model(
    tflite_path: str,
    X_acoustic: np.ndarray,
    X_visual: np.ndarray,
    X_haptic: np.ndarray,
    y: np.ndarray,
    X_hh: Optional[np.ndarray] = None,
    n_test: int = 50
) -> Dict[str, object]:
    """
    Donusturulmus TFLite modelini dogrular.

    Keras model ile TFLite tahminlerini karsilastirir.

    Args:
        tflite_path: TFLite model yolu
        X_acoustic, X_visual, X_haptic, y: Test verisi
        X_hh: Hollow Heart ozellikleri (opsiyonel)
        n_test: Test ornegi sayisi

    Returns:
        Dogrulama sonuclari
    """
    import tensorflow as tf

    if not os.path.exists(tflite_path):
        return {"error": f"Model bulunamadi: {tflite_path}"}

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    n_test = min(n_test, len(y))
    correct = 0
    predictions = []

    for i in range(n_test):
        # Girdi hazirla
        inputs_data = [
            X_acoustic[i:i+1].astype(np.float32),
            X_visual[i:i+1].astype(np.float32),
            X_haptic[i:i+1].astype(np.float32),
        ]
        if X_hh is not None and len(input_details) > 3:
            inputs_data.append(X_hh[i:i+1].astype(np.float32))

        # INT8 modeller icin quantize et
        for inp in input_details:
            shape = inp['shape']
            if len(shape) > 1 and shape[1] == 120:
                data = inputs_data[0]
            elif len(shape) > 1 and shape[1] == 11:
                data = inputs_data[1]
            elif len(shape) > 1 and shape[1] == 7:
                data = inputs_data[2]
            elif len(shape) > 1 and shape[1] == 8 and len(inputs_data) > 3:
                data = inputs_data[3]
            else:
                data = np.zeros(inp['shape'], dtype=np.float32)

            if inp['dtype'] == np.int8:
                scale = inp['quantization_parameters']['scales']
                zero_point = inp['quantization_parameters']['zero_points']
                if len(scale) > 0:
                    data = (data / scale[0] + zero_point[0]).astype(np.int8)
                else:
                    data = data.astype(np.int8)
            elif inp['dtype'] == np.uint8:
                data = np.clip(data * 255, 0, 255).astype(np.uint8)
            else:
                data = data.astype(inp['dtype'])

            interpreter.set_tensor(inp['index'], data)

        interpreter.invoke()

        # Cikti al
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # INT8 cikis icin dequantize et
        if output_details[0]['dtype'] in (np.int8, np.uint8):
            scale = output_details[0]['quantization_parameters']['scales']
            zero_point = output_details[0]['quantization_parameters']['zero_points']
            if len(scale) > 0:
                output_data = (output_data.astype(np.float32) - zero_point[0]) * scale[0]

        pred_class = int(np.argmax(output_data[0]))
        predictions.append(pred_class)

        if pred_class == y[i]:
            correct += 1

    accuracy = correct / n_test if n_test > 0 else 0.0

    result = {
        "tflite_path": tflite_path,
        "n_test": n_test,
        "accuracy": float(accuracy),
        "correct": correct,
        "predictions": predictions[:20],
        "ground_truth": y[:20].tolist(),
        "input_details": [
            {"name": d['name'], "shape": d['shape'].tolist(), "dtype": str(d['dtype'])}
            for d in input_details
        ],
        "output_details": [
            {"name": d['name'], "shape": d['shape'].tolist(), "dtype": str(d['dtype'])}
            for d in output_details
        ],
    }

    print(f"  [Dogrulama] {tflite_path}")
    print(f"  [Dogrulama] Dogruluk: {accuracy:.2%} ({correct}/{n_test})")

    return result


# ============================================
# ANA PIPELINE
# ============================================

def main(quantize: str = "all", benchmark: bool = False):
    """
    Ana donusturme pipeline'i.

    Args:
        quantize: "float16", "int8", "all" veya "none"
        benchmark: Benchmark calistir
    """
    print("=" * 60)
    print("  TFLite Fusion Model Donusturme Pipeline")
    print("  Koc & Akbalik (2025) + Hollow Heart Entegrasyonu")
    print("=" * 60)

    # 1) Sentetik veri olustur
    print("\n[1/6] Veri hazirlaniyor...")
    X_acoustic, X_visual, X_haptic, X_hh, y = generate_synthetic_fusion_data(
        n_samples=500, include_hh=True
    )
    print(f"  Veri boyutu: {len(y)} ornek")
    print(f"  Akustik: {X_acoustic.shape}, Gorsel: {X_visual.shape}")
    print(f"  Haptik: {X_haptic.shape}, HH: {X_hh.shape}")

    # 2) Model olustur (HH dali dahil)
    print("\n[2/6] Fusion modeli olusturuluyor (HH dali dahil)...")
    model = create_fusion_keras_model(include_hh_branch=True)
    model.summary()

    # 3) Egit
    print("\n[3/6] Model egitiliyor...")
    train_fusion_model(
        model, X_acoustic, X_visual, X_haptic, y,
        X_hh=X_hh, epochs=30
    )

    # 4) Representative dataset olustur (INT8 icin)
    print("\n[4/6] Representative dataset hazirlaniyor (INT8 kalibrasyon)...")
    rep_dataset = create_representative_dataset(
        X_acoustic, X_visual, X_haptic, X_hh
    )
    print(f"  Kalibrasyon ornegi: {TFLITE_INT8_REPRESENTATIVE_SAMPLES}")

    # 5) TFLite donusturme
    print("\n[5/6] TFLite donusturme basladi...")

    if quantize == "all":
        results = convert_all_quantizations(model, representative_dataset=rep_dataset)
    else:
        results = {
            quantize: convert_to_tflite(
                model, quantize=quantize,
                representative_dataset=rep_dataset if quantize == "int8" else None
            )
        }

    # 6) Dogrulama + Metadata
    print("\n[6/6] Dogrulama ve metadata...")

    for q_type, q_result in results.items():
        if "error" in q_result:
            continue

        path = q_result.get("output_path", "")
        if os.path.exists(path):
            # Dogrulama
            val = validate_tflite_model(
                path, X_acoustic, X_visual, X_haptic, y,
                X_hh=X_hh, n_test=50
            )
            q_result["validation_accuracy"] = val.get("accuracy", 0.0)

            # Metadata
            embed_metadata(path)

    # Benchmark (opsiyonel)
    if benchmark:
        bench_results = benchmark_all_models()
        results["benchmark"] = bench_results

    # Sonuc ozeti
    print("\n" + "=" * 60)
    print("  DONUSTURME TAMAMLANDI!")
    print("=" * 60)

    for q_type, q_result in results.items():
        if q_type == "benchmark":
            continue
        if isinstance(q_result, dict) and "output_path" in q_result:
            acc_str = ""
            if "validation_accuracy" in q_result:
                acc_str = f", Dogruluk: {q_result['validation_accuracy']:.2%}"
            print(f"  [{q_type:>8}] {q_result['size_mb']:.4f} MB{acc_str}")
            print(f"            {q_result['output_path']}")

    print("=" * 60)

    # Sonuclari JSON olarak kaydet
    summary_path = os.path.join(str(MODELS_DIR), "tflite_conversion_summary.json")
    os.makedirs(str(MODELS_DIR), exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                for kk, vv in v.items()
            }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Ozet: {summary_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TFLite Donusturme Pipeline")
    parser.add_argument("--quantize", type=str, default="all",
                        choices=["none", "float16", "int8", "dynamic", "all"],
                        help="Quantization turu")
    parser.add_argument("--benchmark", action="store_true",
                        help="Model benchmark calistir")

    args = parser.parse_args()
    main(quantize=args.quantize, benchmark=args.benchmark)
