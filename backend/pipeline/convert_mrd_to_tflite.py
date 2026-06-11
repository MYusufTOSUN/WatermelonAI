"""
MRD-YOLO (PyTorch) -> TFLite (INT8) dönüştürme scripti.

ROLE: Senior Mobile AI Deployment Engineer

Bu script:
- XuebinJing/Melon-Ripeness-Detection tabanlı MRD.pt modelini Ultralytics ile yükler
- INT8 quantized TFLite modele export eder (mobil CPU için düşük latency)
- TensorFlow Lite Task Library Vision API ile uyumlu basit metadata + label dosyası üretir
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np  # Ultralytics bazı path'lerde numpy bekleyebilir

# TFLite metadata_writer (Task Library entegrasyonu için)
try:
    from tflite_support.metadata_writers import object_detector
    from tflite_support.metadata_writers import writer_utils
    _HAS_TFLITE_SUPPORT = True
except Exception:
    _HAS_TFLITE_SUPPORT = False


# Proje kökü: C:\Users\tosun\Desktop\watermelon
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MRD_REPO_PATH = PROJECT_ROOT / "mrd_yolo_repo"
MRD_DEFAULT_WEIGHTS = PROJECT_ROOT / "model" / "MRD.pt"
MRD_FALLBACK_WEIGHTS = MRD_REPO_PATH / "weights" / "MRD.pt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_mrd_weights() -> Optional[Path]:
    """MRD.pt ağırlık dosyasını bul."""
    candidates = [MRD_DEFAULT_WEIGHTS, MRD_FALLBACK_WEIGHTS]
    for p in candidates:
        if p.exists():
            return p
    return None


def export_mrd_to_tflite_int8() -> Optional[Path]:
    """
    MRD-YOLO (PyTorch) -> TFLite (INT8) export.

    Not:
      - Kullandığın Ultralytics sürümüne göre 'export' API'sı küçük değişiklik gösterebilir.
      - Gerekirse:  python -m ultralytics.export --help  ile kontrol et.
    """
    # Modifiye ultralytics repo'yu path'e ekle (MRD özel layer'ları için)
    sys.path.insert(0, str(MRD_REPO_PATH))
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover - sadece runtime ortamı için
        print(f"[HATA] ultralytics import edilemedi: {e}")
        return None

    weights_path = _resolve_mrd_weights()
    if weights_path is None:
        print("[HATA] MRD.pt bulunamadı, beklenen yerler:")
        print(f"  - {MRD_DEFAULT_WEIGHTS}")
        print(f"  - {MRD_FALLBACK_WEIGHTS}")
        return None

    print(f"[MRD] Ağırlıklar yükleniyor: {weights_path}")
    model = YOLO(str(weights_path))

    # Ultralytics export: TFLite INT8
    print("[MRD] TFLite INT8 export başlıyor...")
    tflite_target = OUTPUT_DIR / "mrd_yolo_int8.tflite"

    try:
        # Çoğu ultralytics sürümünde aşağıdaki gibi:
        #   model.export(format="tflite", int8=True, imgsz=640)
        model.export(
            format="tflite",
            imgsz=640,
            int8=True,   # FULL INT8 quantization (mobil CPU için)
        )
    except Exception as e:
        print(f"[HATA] TFLite export sırasında hata: {e}")
        return None

    # Ultralytics genelde aynı klasöre 'MRD.tflite' oluşturur
    auto_tflite = weights_path.with_suffix(".tflite")
    if auto_tflite.exists():
        auto_tflite.replace(tflite_target)
        print(f"[MRD] INT8 TFLite kaydedildi: {tflite_target}")
        return tflite_target

    print("[Uyarı] Otomatik .tflite dosyası bulunamadı, path’i elle kontrol etmen gerekebilir.")
    return None


def write_mrd_tflite_metadata(tflite_path: Path) -> None:
    """
    TensorFlow Lite Task Library – Vision API için metadata_writer ile
    model dosyasının İÇİNE metadata gömer.

    Label map: Immature, Ripe, Overripe
    """
    if not _HAS_TFLITE_SUPPORT:
        print("[MRD][Uyarı] tflite-support kurulu değil, metadata_writer kullanılamıyor.")
        print("           Yine de INT8 TFLite modeli hazır, ancak Task Library için")
        print("           metadata eklemek istersen: pip install tflite-support")
        return

    labels = ["Immature", "Ripe", "Overripe"]

    # Label dosyası (ayrı da tutuluyor)
    labels_path = tflite_path.with_name("mrd_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # Model içeriğini oku
    model_buf = tflite_path.read_bytes()

    # ObjectDetector MetadataWriter
    writer = object_detector.MetadataWriter.create_for_inference(
        model_content=model_buf,
        input_norm_mean=[0.0],
        input_norm_std=[1.0],
        labels=labels,
    )
    metadata_buf = writer.populate()

    # Metadata gömülmüş modeli aynı path'e yaz
    writer_utils.save_file(metadata_buf, str(tflite_path))

    print(f"[MRD] Metadata, TFLite dosyasının içine gömüldü: {tflite_path}")
    print(f"[MRD] Label dosyası (ayrıca): {labels_path}")


def main():
    print("=" * 72)
    print("  MRD-YOLO -> TFLite (INT8) Dönüşüm Pipeline")
    print("=" * 72)

    tflite_path = export_mrd_to_tflite_int8()
    if tflite_path is None:
        print("[HATA] TFLite modeli üretilemedi.")
        return

    write_mrd_tflite_metadata(tflite_path)
    print("[OK] MRD TFLite + metadata hazır.")


if __name__ == "__main__":
    main()


