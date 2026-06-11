"""
Master Training Script

Tum fazlari tek komutla calistirir:
  Phase 1: Gorsel siniflandirici egitimi (MobileNetV3)
  Phase 2+3: Multimodal fusion egitimi (KNN + RFC + DNN)
  Phase 4: TFLite donusumu + Flutter kopyalama
  Phase 5: E2E dogrulama

Kullanim:
    python scripts/train_all_pipeline.py
    python scripts/train_all_pipeline.py --skip-visual
    python scripts/train_all_pipeline.py --quick
    python scripts/train_all_pipeline.py --skip-dnn
"""

import os
import sys
import shutil
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FLUTTER_MODELS_DIR = PROJECT_ROOT / "flutter_app" / "assets" / "models"


def phase_1_visual(quick=False):
    """Phase 1: Gorsel siniflandirici egitimi."""
    print("\n" + "=" * 60)
    print("PHASE 1: Gorsel Siniflandirici Egitimi")
    print("=" * 60)

    from backend.pipeline.train_visual_classifier import train_visual_classifier
    result = train_visual_classifier(
        epochs=15 if quick else 40,
        batch_size=16,
        learning_rate=0.001,
        quick=quick,
    )
    print(f"\n  Phase 1 sonuc: val={result['val_accuracy']:.2%}, "
          f"test={result['test_accuracy']:.2%}")
    return result


def phase_2_3_multimodal(model_type="all", quick=False, lowo=True):
    """Phase 2+3: Multimodal fusion egitimi."""
    print("\n" + "=" * 60)
    print("PHASE 2+3: Multimodal Fusion Egitimi")
    print("=" * 60)

    # Cache temizle
    for fname in ["X_features.npy", "y_labels.npy", "groups.npy"]:
        fpath = PROCESSED_DIR / fname
        if fpath.exists():
            os.remove(str(fpath))
            print(f"  Cache temizlendi: {fname}")

    from backend.pipeline.train import train_pipeline
    results = train_pipeline(
        model_type=model_type,
        multimodal=True,
        augment=True,
        n_augment=3,
        split_strategy="random",
        do_lowo=lowo and not quick,
        use_xigua=False,
        use_smote=True,
        cross_wm_mixup=True,
        per_group_norm=True,
    )
    return results


def phase_4_copy_models():
    """Phase 4: Modelleri Flutter'a kopyala."""
    print("\n" + "=" * 60)
    print("PHASE 4: Flutter Model Kopyalama")
    print("=" * 60)

    os.makedirs(str(FLUTTER_MODELS_DIR), exist_ok=True)

    models_to_copy = [
        "fusion_model_fp16.tflite",
        "visual_classifier_fp16.tflite",
    ]

    for model_name in models_to_copy:
        src = MODELS_DIR / model_name
        dst = FLUTTER_MODELS_DIR / model_name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            size_kb = os.path.getsize(str(dst)) / 1024
            print(f"  {model_name} -> Flutter ({size_kb:.1f} KB)")
        else:
            print(f"  [!] {model_name} bulunamadi, atlaniyor")


def phase_5_validate():
    """Phase 5: E2E dogrulama."""
    print("\n" + "=" * 60)
    print("PHASE 5: End-to-End Dogrulama")
    print("=" * 60)

    from scripts.validate_pipeline import main as validate_main
    validate_main()


def main():
    parser = argparse.ArgumentParser(description="Master Training Pipeline")
    parser.add_argument("--skip-visual", action="store_true",
                        help="Phase 1 (gorsel egitim) atla")
    parser.add_argument("--skip-dnn", action="store_true",
                        help="DNN egitimini atla (sadece KNN+RFC)")
    parser.add_argument("--quick", action="store_true",
                        help="Hizli mod (az epoch, LOWO yok)")
    parser.add_argument("--no-validate", action="store_true",
                        help="E2E dogrulama atla")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("WATERMELON RIPENESS DETECTION - MASTER TRAINING")
    print(f"Quick: {args.quick} | Skip-Visual: {args.skip_visual}")
    print("=" * 60)

    # Phase 1
    if not args.skip_visual:
        phase_1_visual(quick=args.quick)
    else:
        print("\n[Phase 1 atlandi]")

    # Phase 2+3
    model_type = "both" if args.skip_dnn else "all"
    phase_2_3_multimodal(
        model_type=model_type,
        quick=args.quick,
        lowo=True,
    )

    # Phase 4
    phase_4_copy_models()

    # Phase 5
    if not args.no_validate:
        phase_5_validate()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"TUM FAZLAR TAMAMLANDI ({elapsed:.0f} saniye)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
