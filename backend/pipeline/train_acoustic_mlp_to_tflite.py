"""
120-boyutlu akustik/haptik feature'lar için MLP eğitip TFLite'a çeviren script.

ROLE: Senior Mobile AI Deployment Engineer

Kaynak:
  - Qilin Watermelon Dataset üzerinden çıkarılmış 120-dim feature'lar
  - Koç & Akbalık (2025) akustik fingerprint: MFCC, ZCR, Spectral Flatness vb.

Girdi:
  data/processed/X_features.npy  -> shape: (N, 120)
  data/processed/y_labels.npy    -> shape: (N,)  (0=Immature, 1=Ripe, 2=Overripe)

Çıktı:
  data/models/acoustic_mlp_fp16.tflite  (1D vektör girişi için optimize edilmiş)
"""

import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# TFLite metadata_writer (Audio / Generic Classifier entegrasyonu için)
try:
    from tflite_support.metadata_writers import audio_classifier
    from tflite_support.metadata_writers import writer_utils as ac_writer_utils
    _HAS_TFLITE_SUPPORT = True
except Exception:
    _HAS_TFLITE_SUPPORT = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = PROCESSED_DIR / "X_features.npy"  # 120-dim acoustic/haptic feature
Y_PATH = PROCESSED_DIR / "y_labels.npy"    # integer labels 0/1/2


def load_qilin_features():
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(
            f"Feature dosyaları bulunamadı:\n  {X_PATH}\n  {Y_PATH}\n"
            "Önce backend.pipeline.train ile feature çıkarımını tamamladığından emin ol."
        )

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"[MLP] X shape: {X.shape}, y shape: {y.shape}")
    return X.astype("float32"), y.astype("int32")


def build_mlp(input_dim: int = 120, n_classes: int = 3) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="acoustic_input")

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="acoustic_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_and_export():
    X, y = load_qilin_features()
    n_classes = len(np.unique(y))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights (sınıf dengesizliği için)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[MLP] Class weights: {class_weight_dict}")

    model = build_mlp(input_dim=X.shape[1], n_classes=n_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = history.history.get("val_accuracy", [0.0])[-1]
    print(f"[MLP] Son val_accuracy: {val_acc:.4f}")

    # Keras SavedModel olarak kaydet (geçici)
    saved_model_dir = MODELS_DIR / "acoustic_mlp_saved"
    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)
    model.save(saved_model_dir)

    # TFLite FP16 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path = MODELS_DIR / "acoustic_mlp_fp16.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"[MLP] TFLite FP16 model kaydedildi: {tflite_path} ({size_mb:.4f} MB)")

    # AudioClassifier / Task Library için metadata gömme (opsiyonel ama önerilir)
    if _HAS_TFLITE_SUPPORT:
        try:
            _embed_audio_classifier_metadata(tflite_path, n_classes)
        except Exception as e:
            print(f"[MLP][Uyarı] Audio metadata gömülürken hata: {e}")

    # Temizlik
    import shutil
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    return tflite_path


def _embed_audio_classifier_metadata(tflite_path: Path, n_classes: int) -> None:
    """
    AudioClassifier.createFromFile ile kullanılabilecek minimal metadata'yı
    model dosyasının içine gömer.

    Not:
      - Bu MLP modelinin girdisi 120-boyutlu feature vektörü olduğu için
        tam anlamıyla ham WAV girişli bir AudioClassifier değildir.
      - Yine de Task Library'nin Generic / Audio API'leri ile entegrasyonu
        kolaylaştırmak için sınıf isimlerini ve temel audio parametrelerini
        metadata'ya ekliyoruz.
    """
    if not _HAS_TFLITE_SUPPORT:
        return

    labels = ["Immature", "Ripe", "Overripe"][:n_classes]
    model_buf = tflite_path.read_bytes()

    # Örnek parametreler (gerekirse dataset'e göre düzenlenebilir)
    sample_rate = 44100
    channels = 1

    writer = audio_classifier.MetadataWriter.create_for_inference(
        model_content=model_buf,
        sample_rate=sample_rate,
        channels=channels,
        labels=labels,
    )
    metadata_buf = writer.populate()
    ac_writer_utils.save_file(metadata_buf, str(tflite_path))

    # Ayrıca label dosyası da kaydet
    labels_path = tflite_path.with_name("acoustic_mlp_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    print(f"[MLP] AudioClassifier metadata gömüldü: {tflite_path}")
    print(f"[MLP] Label dosyası: {labels_path}")


def main():
    print("=" * 72)
    print("  120-dim Acoustic/Haptic MLP -> TFLite (FP16) Pipeline")
    print("=" * 72)

    tflite_path = train_and_export()
    print(f"[OK] Akustik MLP TFLite hazır: {tflite_path}")


if __name__ == "__main__":
    main()


