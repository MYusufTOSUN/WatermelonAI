"""
Model Değerlendirme Script

Eğitilmiş modelleri test verileri üzerinde değerlendirir
ve detaylı performans raporları oluşturur.

Kullanım:
    python scripts/evaluate.py --model knn
    python scripts/evaluate.py --model rfc
"""

import numpy as np
import os
import argparse
import json
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.config import MODELS_DIR, CLASS_LABELS, PROCESSED_DATA_DIR
from backend.module_e.classifier import RipenessClassifier
from backend.module_e.elasticity_index import ElasticityIndexCalculator


def evaluate_model(model_type: str = "knn"):
    """
    Model değerlendirme pipeline'ı.
    
    Args:
        model_type: 'knn' veya 'rfc'
    """
    print("=" * 60)
    print(f"Model Değerlendirme: {model_type.upper()}")
    print("=" * 60)

    # Model yükle
    classifier = RipenessClassifier(model_type=model_type)
    try:
        classifier.load_model()
    except FileNotFoundError:
        print(f"HATA: {model_type} modeli bulunamadı!")
        print("Önce eğitim yapın: python -m backend.pipeline.train")
        return

    # Test verisi yükle
    X_path = os.path.join(str(PROCESSED_DATA_DIR), "X_features.npy")
    y_path = os.path.join(str(PROCESSED_DATA_DIR), "y_labels.npy")

    if not os.path.exists(X_path):
        print("İşlenmiş test verisi bulunamadı. Sentetik veri kullanılıyor...")
        from backend.pipeline.data_loader import QilinDatasetLoader
        loader = QilinDatasetLoader()
        X, y = loader.extract_features_from_audio()
    else:
        X = np.load(X_path)
        y = np.load(y_path)

    # Tahmin
    predictions, probabilities = classifier.predict(X)

    # Metrikler
    accuracy = accuracy_score(y, predictions)
    f1_macro = f1_score(y, predictions, average='macro')
    f1_weighted = f1_score(y, predictions, average='weighted')

    print(f"\nDoğruluk (Accuracy): {accuracy:.4f}")
    print(f"F1 Makro: {f1_macro:.4f}")
    print(f"F1 Ağırlıklı: {f1_weighted:.4f}")

    print("\nSınıflandırma Raporu:")
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    print(classification_report(y, predictions, target_names=target_names))

    print("Karmaşıklık Matrisi (Confusion Matrix):")
    cm = confusion_matrix(y, predictions)
    print(f"  {'':>25}", end="")
    for name in target_names:
        print(f"  {name[:12]:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>25}", end="")
        for val in row:
            print(f"  {val:>12}", end="")
        print()

    # Fiziksel doğrulama (örnek)
    print("\n" + "=" * 60)
    print("Fiziksel Doğrulama Örnekleri:")
    ei_calc = ElasticityIndexCalculator()

    sample_cases = [
        {"f2": 120, "mag_db": 28, "mass_kg": 5.0, "expected": "Olgun"},
        {"f2": 200, "mag_db": 22, "mass_kg": 3.0, "expected": "Olgunlaşmamış"},
        {"f2": 70, "mag_db": 18, "mass_kg": 6.0, "expected": "İçi Geçmiş"},
    ]

    for case in sample_cases:
        ei = ei_calc.calculate_ei(case["f2"], case["mass_kg"])
        validation = ei_calc.validate_physical_constraints(case["f2"], case["mag_db"])
        print(f"\n  f2={case['f2']}Hz, mag={case['mag_db']}dB, m={case['mass_kg']}kg")
        print(f"  EI = {ei:.2f}")
        print(f"  Olgunluk (f2<150Hz): {'✓' if validation['f2_indicates_ripe'] else '✗'}")
        print(f"  Sinyal Kalitesi (>25dB): {'✓' if validation['magnitude_sufficient'] else '✗'}")
        print(f"  Beklenen: {case['expected']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model değerlendirme")
    parser.add_argument("--model", type=str, default="knn",
                        choices=["knn", "rfc"],
                        help="Model tipi")

    args = parser.parse_args()
    evaluate_model(args.model)

