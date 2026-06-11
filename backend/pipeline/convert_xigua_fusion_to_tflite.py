"""
Xigua Full Fusion Model (Audio+Image) → ONNX → TFLite Export Pipeline

Xigua FusionModel mimarisi:
  - Audio branch:  LSTM(input=100, hidden=128, batch_first=True)
                   Girdi: (1, 160, 100) — 16000 örnek → reshape(160, 100)
  - Image branch:  ResNet50 (fc katmanı çıkarılmış) → (1, 2048)
                   Girdi: (1, 3, 224, 224) — ImageNet normalizasyonu
  - Fusion head:   Linear(2176 → 64) → ReLU → Linear(64 → 1)
                   Çıktı: Brix / olgunluk regresyon skoru

Export akışı:
  1) xigua_pytorch.pth → PyTorch FusionModel yükle
  2) torch.onnx.export (dual-input ONNX)
  3) onnx2tf veya onnxruntime → TFLite (FP16)

Kullanım:
  python -m backend.pipeline.convert_xigua_fusion_to_tflite
"""

import os
import sys
import json
import time
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import PROJECT_ROOT, MODELS_DIR

# Yollar
XIGUA_PTH_PATH = str(
    PROJECT_ROOT / "dataset" / "datasets" / "watermelon_dataset" / "xigua_pytorch.pth"
)
OUTPUT_DIR = str(MODELS_DIR)
ONNX_OUTPUT = os.path.join(OUTPUT_DIR, "xigua_fusion.onnx")
TFLITE_FP16_OUTPUT = os.path.join(OUTPUT_DIR, "xigua_fusion_fp16.tflite")
TFLITE_INT8_OUTPUT = os.path.join(OUTPUT_DIR, "xigua_fusion_int8.tflite")
METADATA_OUTPUT = os.path.join(OUTPUT_DIR, "xigua_fusion_metadata.json")

# Xigua model sabitleri
AUDIO_LENGTH = 16000       # 16k örnek (1 sn @ 16kHz veya ~0.36 sn @ 44.1kHz)
AUDIO_SEQ_LEN = 160        # LSTM zaman adımı
AUDIO_FEAT_DIM = 100       # LSTM girdi boyutu
IMAGE_SIZE = 224            # ResNet50 girdi boyutu
LSTM_HIDDEN = 128
RESNET_FEAT = 2048
FC1_OUT = 64


def _build_fusion_model():
    """Xigua FusionModel'i orijinal mimarisiyle oluşturur."""
    import torch
    import torch.nn as nn
    from torchvision import models

    class FusionModel(nn.Module):
        def __init__(self):
            super(FusionModel, self).__init__()
            self.lstm = nn.LSTM(
                input_size=AUDIO_FEAT_DIM,
                hidden_size=LSTM_HIDDEN,
                batch_first=True,
            )
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            self.fc1 = nn.Linear(LSTM_HIDDEN + RESNET_FEAT, FC1_OUT)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(FC1_OUT, 1)

        def forward(self, audio, image):
            lstm_out, _ = self.lstm(audio)
            audio_feat = lstm_out[:, -1, :]
            img_feat = self.resnet(image)
            img_feat = img_feat.view(img_feat.size(0), -1)
            merged = torch.cat((audio_feat, img_feat), dim=1)
            out = self.fc1(merged)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    return FusionModel


def load_xigua_model(model_path: str = XIGUA_PTH_PATH):
    """xigua_pytorch.pth'den FusionModel'i yükler."""
    import torch

    FusionModel = _build_fusion_model()
    model = FusionModel()

    if not os.path.exists(model_path):
        print(f"[HATA] xigua_pytorch.pth bulunamadı: {model_path}")
        return None

    print(f"[Xigua] Ağırlıklar yükleniyor: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("[Xigua] Model başarıyla yüklendi")
    return model


def export_to_onnx(model, output_path: str = ONNX_OUTPUT) -> str:
    """PyTorch FusionModel → ONNX export (dual-input)."""
    import torch

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_audio = torch.randn(1, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM)
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("[Xigua] ONNX export başlıyor...")
    torch.onnx.export(
        model,
        (dummy_audio, dummy_image),
        output_path,
        input_names=["audio_input", "image_input"],
        output_names=["brix_score"],
        opset_version=12,
        dynamic_axes={
            "audio_input": {0: "batch"},
            "image_input": {0: "batch"},
            "brix_score": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Xigua] ONNX kaydedildi: {output_path} ({size_mb:.2f} MB)")
    return output_path


def export_to_tflite_fp16(onnx_path: str = ONNX_OUTPUT,
                          output_path: str = TFLITE_FP16_OUTPUT) -> str:
    """ONNX → TFLite (FP16 quantization)."""
    import tensorflow as tf

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # onnx → saved_model → tflite akışı (onnx2tf veya onnx_tf)
    saved_model_dir = os.path.join(os.path.dirname(output_path), "_xigua_saved_model_temp")

    # Yöntem 1: onnx2tf ile
    try:
        import onnx2tf
        print("[Xigua] onnx2tf ile SavedModel oluşturuluyor...")
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            non_verbose=True,
        )
    except ImportError:
        # Yöntem 2: onnx-tf ile
        try:
            import onnx
            from onnx_tf.backend import prepare
            print("[Xigua] onnx-tf ile SavedModel oluşturuluyor...")
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(saved_model_dir)
        except ImportError:
            print("[HATA] onnx2tf veya onnx-tf bulunamadı.")
            print("  pip install onnx2tf onnx-tf")
            return ""

    print("[Xigua] TFLite FP16 dönüşümü başlıyor...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    start = time.time()
    tflite_model = converter.convert()
    elapsed = time.time() - start

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Temizlik
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Xigua] TFLite FP16 kaydedildi: {output_path} ({size_mb:.2f} MB, {elapsed:.1f}s)")
    return output_path


def write_metadata(tflite_path: str, output_path: str = METADATA_OUTPUT):
    """Xigua fusion TFLite model metadata dosyası oluşturur."""
    size_bytes = os.path.getsize(tflite_path) if os.path.exists(tflite_path) else 0
    meta = {
        "model_name": "XiguaFusion_AudioImage",
        "model_version": "1.0.0",
        "model_description": (
            "Xigua Late Fusion: LSTM(audio) + ResNet50(image) → Brix regresyon. "
            "Qilin Watermelon Dataset üzerinde eğitildi."
        ),
        "model_file": os.path.basename(tflite_path),
        "model_size_bytes": size_bytes,
        "inputs": {
            "audio_input": {
                "shape": [1, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM],
                "dtype": "float32",
                "description": (
                    f"Audio waveform: {AUDIO_LENGTH} sample → reshape({AUDIO_SEQ_LEN}, {AUDIO_FEAT_DIM}). "
                    "Normalize: [-1, 1]."
                ),
            },
            "image_input": {
                "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
                "dtype": "float32",
                "description": (
                    f"RGB image {IMAGE_SIZE}x{IMAGE_SIZE}, ImageNet normalization: "
                    "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]."
                ),
            },
        },
        "outputs": {
            "brix_score": {
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Brix / ripeness regression score",
            }
        },
        "flutter_integration": {
            "asset_path": "assets/models/xigua_fusion_fp16.tflite",
            "service": "XiguaFusionService",
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Xigua] Metadata kaydedildi: {output_path}")


def main():
    print("=" * 60)
    print("  XIGUA FULL FUSION (Audio+Image) → TFLite EXPORT")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Model yükle
    model = load_xigua_model()
    if model is None:
        print("[HATA] Model yüklenemedi, çıkılıyor.")
        return

    # 2) ONNX export
    onnx_path = export_to_onnx(model)

    # 3) TFLite FP16
    tflite_path = export_to_tflite_fp16(onnx_path)

    # 4) Metadata
    if tflite_path and os.path.exists(tflite_path):
        write_metadata(tflite_path)

    print("\n" + "=" * 60)
    print("  EXPORT TAMAMLANDI")
    print("=" * 60)
    if tflite_path:
        print(f"  TFLite: {tflite_path}")
    print(f"  ONNX:   {onnx_path}")
    print(
        "\n  Flutter'a kopyalamak için:\n"
        f"    copy {tflite_path} flutter_app/assets/models/"
    )


if __name__ == "__main__":
    main()

