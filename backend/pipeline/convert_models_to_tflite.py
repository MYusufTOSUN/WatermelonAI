"""
MRD-YOLO ve Xigua PyTorch modellerini TFLite formatına çevirir.

Flutter'da kullanmak için:
1. MRD-YOLO -> ONNX -> TFLite (veya direkt ONNX Runtime Mobile)
2. Xigua ResNet50 feature extractor -> TFLite
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    MRD_YOLO_MODEL_PATH,
    MRD_YOLO_FALLBACK_PATH,
    PROJECT_ROOT,
    MODELS_DIR,
)

# Xigua model yolu
XIGUA_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "datasets", "watermelon_dataset", "xigua_pytorch.pth"
)


def convert_mrd_yolo_to_onnx():
    """
    MRD-YOLO modelini ONNX formatına çevirir.
    Flutter'da ONNX Runtime Mobile ile kullanılabilir.
    """
    try:
        from ultralytics import YOLO
        
        # Model yolunu bul
        model_path = None
        for path in [MRD_YOLO_MODEL_PATH, MRD_YOLO_FALLBACK_PATH]:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("[HATA] MRD-YOLO modeli bulunamadı!")
            return None
        
        print(f"[MRD-YOLO] Model yükleniyor: {model_path}")
        model = YOLO(model_path)
        
        # ONNX export
        output_path = os.path.join(str(MODELS_DIR), "mrd_yolo.onnx")
        model.export(format="onnx", imgsz=640, simplify=True)
        
        # Export edilen dosyayı bul
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        exported_path = os.path.join(os.path.dirname(model_path), f"{base_name}.onnx")
        
        if os.path.exists(exported_path):
            import shutil
            shutil.move(exported_path, output_path)
            print(f"[MRD-YOLO] ONNX modeli kaydedildi: {output_path}")
            return output_path
        else:
            print("[HATA] ONNX export başarısız!")
            return None
            
    except Exception as e:
        print(f"[HATA] MRD-YOLO ONNX dönüşümü: {e}")
        return None


def convert_xigua_resnet_to_tflite():
    """
    Xigua modelindeki ResNet50 feature extractor'ı TFLite'a çevirir.
    """
    try:
        import tensorflow as tf
        from torchvision import models, transforms
        import torch.nn as nn
        
        # ResNet50 feature extractor oluştur (Xigua'daki gibi)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        
        # PyTorch modelini yükle (eğer varsa)
        if os.path.exists(XIGUA_MODEL_PATH):
            print(f"[Xigua] PyTorch modeli yükleniyor: {XIGUA_MODEL_PATH}")
            checkpoint = torch.load(XIGUA_MODEL_PATH, map_location='cpu', weights_only=False)

            # Checkpoint'tan ResNet ağırlıklarını çıkar
            state_dict = None
            if isinstance(checkpoint, dict):
                # Yaygın anahtar desenleri dene
                for key in ['resnet_state_dict', 'image_encoder', 'model_state_dict',
                            'state_dict', 'encoder']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                if state_dict is None and any(k.startswith(('resnet.', 'image_encoder.', 'layer')) for k in checkpoint):
                    # Doğrudan state dict olabilir
                    state_dict = checkpoint
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()

            if state_dict is not None:
                # Prefix'leri temizle (resnet. veya image_encoder. gibi)
                cleaned = {}
                for k, v in state_dict.items():
                    for prefix in ('resnet.', 'image_encoder.', 'backbone.'):
                        if k.startswith(prefix):
                            k = k[len(prefix):]
                            break
                    cleaned[k] = v
                try:
                    feature_extractor.load_state_dict(cleaned, strict=False)
                    print("[Xigua] ResNet ağırlıkları yüklendi (strict=False)")
                except Exception as e:
                    print(f"[Xigua] Ağırlık yükleme uyarısı: {e}")
            else:
                print("[Xigua] Checkpoint'ta uyumlu state_dict bulunamadı, ImageNet ağırlıkları kullanılıyor")
        
        # Örnek girdi ile test
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = feature_extractor(dummy_input)
            features = features.view(features.size(0), -1)
        
        print(f"[Xigua] Feature boyutu: {features.shape}")  # [1, 2048]
        
        # PyTorch -> ONNX -> TFLite dönüşümü
        onnx_path = os.path.join(str(MODELS_DIR), "xigua_resnet.onnx")
        
        print("[Xigua] ONNX'a çevriliyor...")
        torch.onnx.export(
            feature_extractor,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=['features'],
            opset_version=11,
            dynamic_axes={'image': {0: 'batch'}, 'features': {0: 'batch'}}
        )
        
        print(f"[Xigua] ONNX modeli kaydedildi: {onnx_path}")
        
        # ONNX -> TFLite
        print("[Xigua] TFLite'a çevriliyor...")
        import onnx
        
        onnx_model = onnx.load(onnx_path)
        
        # TFLite converter
        converter = tf.lite.TFLiteConverter.from_onnx_model(onnx_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(str(MODELS_DIR), "xigua_resnet_fp16.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[Xigua] TFLite modeli kaydedildi: {tflite_path}")
        return tflite_path
        
    except Exception as e:
        print(f"[HATA] Xigua TFLite dönüşümü: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("  MRD-YOLO ve Xigua Modellerini TFLite'a Çevirme")
    print("=" * 60)
    
    os.makedirs(str(MODELS_DIR), exist_ok=True)
    
    # 1. MRD-YOLO -> ONNX
    print("\n[1/2] MRD-YOLO -> ONNX dönüşümü...")
    mrd_onnx = convert_mrd_yolo_to_onnx()
    
    # 2. Xigua ResNet -> TFLite
    print("\n[2/2] Xigua ResNet50 -> TFLite dönüşümü...")
    xigua_tflite = convert_xigua_resnet_to_tflite()
    
    print("\n" + "=" * 60)
    print("  DÖNÜŞÜM TAMAMLANDI!")
    print("=" * 60)
    
    if mrd_onnx:
        print(f"  MRD-YOLO ONNX: {mrd_onnx}")
    if xigua_tflite:
        print(f"  Xigua ResNet TFLite: {xigua_tflite}")
    
    print("\nNot: MRD-YOLO için Flutter'da ONNX Runtime Mobile kullanılmalı.")
    print("     Xigua ResNet TFLite modeli direkt kullanılabilir.")


if __name__ == "__main__":
    main()

