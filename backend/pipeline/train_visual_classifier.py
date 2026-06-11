"""
Gorsel Siniflandirici Egitimi

Roboflow + MRD-YOLO + Qilin gorsellerini birlestirerek
MobileNetV3-Small tabanli 3-sinif karpuz olgunluk siniflandirici egitir.

TF/Keras ile egitim yapilir (TFLite donusumu icin en guvenilir yol).

Kullanim:
    python -m backend.pipeline.train_visual_classifier
    python -m backend.pipeline.train_visual_classifier --epochs 30 --quick
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import MODELS_DIR, CLASS_LABELS

PROJECT_ROOT = Path(__file__).parent.parent.parent
VISUAL_MODEL_DIR = PROJECT_ROOT / "data" / "models"


def train_visual_classifier(
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    image_size: int = 224,
    augment_roboflow: int = 8,
    quick: bool = False,
):
    """
    MobileNetV3-Small fine-tune ederek 3-sinif gorsel siniflandirici egitir.

    Args:
        epochs: Egitim epoch sayisi
        batch_size: Batch boyutu
        learning_rate: Baslangic ogrenme orani
        image_size: Gorsel girdi boyutu (224x224)
        augment_roboflow: Roboflow gorselleri icin augment katsayisi
        quick: Hizli mod (daha az epoch)
    """
    import tensorflow as tf
    from tensorflow import keras

    if quick:
        epochs = min(epochs, 15)

    print("=" * 60)
    print("Gorsel Siniflandirici Egitimi (MobileNetV3-Small)")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------------
    # 1) Veri yukleme
    # -----------------------------------------------------------
    print("\n[1/5] Birlesik gorsel dataset yukleniyor...")
    from backend.pipeline.visual_dataset_loader import UnifiedVisualDatasetLoader
    loader = UnifiedVisualDatasetLoader(image_size=image_size)
    X_all, y_all, groups_all = loader.get_unified_dataset(augment_roboflow=augment_roboflow)

    # -----------------------------------------------------------
    # 2) Train / Val / Test bolmesi
    # -----------------------------------------------------------
    print("\n[2/5] Veri bolunuyor (stratified, Qilin ID-aware)...")
    from sklearn.model_selection import train_test_split

    # Qilin gorsellerini watermelon ID bazli bol
    qilin_mask = groups_all >= 0
    external_mask = groups_all < 0

    X_qilin, y_qilin, g_qilin = X_all[qilin_mask], y_all[qilin_mask], groups_all[qilin_mask]
    X_ext, y_ext = X_all[external_mask], y_all[external_mask]

    # Qilin: sinif-stratified watermelon ID bazli group split
    # Her siniftan en az 1 karpuz test ve val'de olmali
    np.random.seed(42)

    # Karpuz ID → sinif haritalamasi (cogunluk sinifi)
    id_to_cat = {}
    for gid in np.unique(g_qilin):
        mask = g_qilin == gid
        cats, cnts = np.unique(y_qilin[mask], return_counts=True)
        id_to_cat[gid] = cats[np.argmax(cnts)]

    # Her siniftan karpuzlari ayir
    cat_ids = {0: [], 1: [], 2: []}
    for gid, cat in id_to_cat.items():
        cat_ids[cat].append(gid)
    for cat in cat_ids:
        np.random.shuffle(cat_ids[cat])

    # Her siniftan 1 karpuz test, 1 val, geri kalan train
    test_ids = set()
    val_ids = set()
    for cat, ids in cat_ids.items():
        if len(ids) >= 3:
            test_ids.add(ids[0])
            val_ids.add(ids[1])
        elif len(ids) >= 2:
            test_ids.add(ids[0])
            val_ids.add(ids[1])
        elif len(ids) >= 1:
            test_ids.add(ids[0])

    q_train_mask = np.array([g not in test_ids and g not in val_ids for g in g_qilin])
    q_val_mask = np.array([g in val_ids for g in g_qilin])
    q_test_mask = np.array([g in test_ids for g in g_qilin])

    print(f"  Qilin split: test_ids={test_ids}, val_ids={val_ids}")

    # Dis veriyi (Roboflow + MRD) train/val olarak bol
    if len(X_ext) > 5:
        X_ext_train, X_ext_val, y_ext_train, y_ext_val = train_test_split(
            X_ext, y_ext, test_size=0.15, random_state=42, stratify=y_ext
        )
    else:
        X_ext_train, y_ext_train = X_ext, y_ext
        X_ext_val, y_ext_val = np.array([]).reshape(0, *X_ext.shape[1:]), np.array([])

    # Birlestir
    X_train = np.concatenate([X_qilin[q_train_mask], X_ext_train]) if len(X_ext_train) > 0 else X_qilin[q_train_mask]
    y_train = np.concatenate([y_qilin[q_train_mask], y_ext_train]) if len(y_ext_train) > 0 else y_qilin[q_train_mask]

    X_val = np.concatenate([X_qilin[q_val_mask], X_ext_val]) if len(X_ext_val) > 0 else X_qilin[q_val_mask]
    y_val = np.concatenate([y_qilin[q_val_mask], y_ext_val]) if len(y_ext_val) > 0 else y_qilin[q_val_mask]

    X_test = X_qilin[q_test_mask]
    y_test = y_qilin[q_test_mask]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        u, c = np.unique(ys, return_counts=True)
        print(f"    {name} dagilimi: {dict(zip(u.tolist(), c.tolist()))}")

    # -----------------------------------------------------------
    # 3) Preprocessing & Dataset
    # -----------------------------------------------------------
    print("\n[3/5] Dataset hazirlaniyor...")

    # MobileNetV3 preprocessing: [-1, 1] normalizasyon
    X_train_f = keras.applications.mobilenet_v3.preprocess_input(X_train.astype(np.float32))
    X_val_f = keras.applications.mobilenet_v3.preprocess_input(X_val.astype(np.float32))
    X_test_f = keras.applications.mobilenet_v3.preprocess_input(X_test.astype(np.float32))

    # Bellek tasarrufu: uint8 versiyonlari sil
    del X_train, X_val, X_test, X_all

    n_classes = 3

    # Keras augmentation katmani (train icin)
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
    ])

    # -----------------------------------------------------------
    # 4) Model olusturma
    # -----------------------------------------------------------
    print("\n[4/5] MobileNetV3-Small olusturuluyor...")

    # Backbone: ImageNet pretrained
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )

    # Backbone'un %40'ini dondur (eski %80 cok donmustu, sinif prior'a sikiSti)
    n_layers = len(base_model.layers)
    freeze_until = int(n_layers * 0.4)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Backbone: {n_layers} katman, {trainable_count} egitilecek")

    # Full model
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # --- Class weights ---
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, cw)}
    print(f"  Class weights: {class_weight}")

    # --- Focal Loss ---
    def focal_loss(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), n_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, 2.0) * 0.25
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    # Daha dusuk LR kullan cunku backbone unfreezed - fine-tune modu
    fine_tune_lr = learning_rate * 0.1
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=fine_tune_lr,
            weight_decay=1e-4,
        ),
        loss=focal_loss,
        metrics=["accuracy"],
    )
    print(f"  Fine-tune LR: {fine_tune_lr}")

    model.summary(print_fn=lambda x: None)  # Sessiz

    # --- Callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # -----------------------------------------------------------
    # 5) Egitim
    # -----------------------------------------------------------
    print(f"\n[5/5] Egitim basliyor ({epochs} epoch)...")
    history = model.fit(
        X_train_f, y_train,
        validation_data=(X_val_f, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # -----------------------------------------------------------
    # Degerlendirme
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEGERLENDIRME")
    print("=" * 60)

    # Val sonucu
    val_loss, val_acc = model.evaluate(X_val_f, y_val, verbose=0)
    print(f"  Val Accuracy: {val_acc:.4f}")

    # Test sonucu
    test_loss, test_acc = model.evaluate(X_test_f, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test_f, verbose=0), axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    all_labels = [0, 1, 2]
    label_names = ["Olgunlasmamis", "Olgun", "Ici Gecmis"]
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    report = classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"  Confusion Matrix:")
    for row in cm.tolist():
        print(f"    {row}")
    print(classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=label_names, zero_division=0,
    ))

    # -----------------------------------------------------------
    # TFLite donusumu
    # -----------------------------------------------------------
    print("\nTFLite donusumu...")
    import shutil

    # Keras model → TFLite (SavedModel veya dogrudan Keras)
    saved_model_dir = str(VISUAL_MODEL_DIR / "_temp_visual_saved_model")
    try:
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    except Exception as e:
        print(f"  SavedModel export basarisiz ({e}), Keras'tan donusturuluyor...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # FP16 TFLite
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = str(VISUAL_MODEL_DIR / "visual_classifier_fp16.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  TFLite kaydedildi: {tflite_path} ({size_kb:.1f} KB)")

    # TFLite dogrulama (Turkce path uyumlulugu icin binary okuma)
    try:
        with open(tflite_path, 'rb') as f:
            tflite_content = f.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_content)
        interpreter.allocate_tensors()
        inp_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        print(f"  TFLite Input:  {inp_details[0]['shape']} {inp_details[0]['dtype']}")
        print(f"  TFLite Output: {out_details[0]['shape']} {out_details[0]['dtype']}")
    except Exception as e:
        print(f"  TFLite dogrulama atlandi (model uretildi): {e}")

    # Sonuclari kaydet
    results = {
        "model": "MobileNetV3-Small",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "epochs_trained": len(history.history["loss"]),
        "confusion_matrix": cm.tolist(),
        "tflite_path": tflite_path,
        "tflite_size_kb": round(size_kb, 1),
    }

    results_path = str(VISUAL_MODEL_DIR / "visual_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Sonuclar: {results_path}")

    print("\n" + "=" * 60)
    print(f"GORSEL SINIFLANDIRICI EGITIMI TAMAMLANDI")
    print(f"  Val: {val_acc:.2%} | Test: {test_acc:.2%} | TFLite: {size_kb:.1f} KB")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gorsel siniflandirici egitimi")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    train_visual_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        quick=args.quick,
    )
