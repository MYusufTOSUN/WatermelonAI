"""
2-Sinif (Yenir/Yenmez) LOWO Degerlendirme

Mevcut 146-D multimodal feature'lari kullanarak KNN + RFC modellerini
2-sinifli problem olarak yeniden egitir ve LOWO cross-validation yapar.

Sinif haritasi:
  0 (Olgunlasmamis) + 2 (Ici Gecmis) -> 0 (YENMEZ)
  1 (Olgun)                          -> 1 (YENIR)

Kullanim:
    python scripts/eval_binary_ripeness.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def load_or_build_features():
    X_path = PROCESSED_DIR / "X_features.npy"
    y_path = PROCESSED_DIR / "y_labels.npy"
    g_path = PROCESSED_DIR / "groups.npy"

    if X_path.exists() and y_path.exists() and g_path.exists():
        print(f"[cache] Feature'lar yukleniyor: {X_path}")
        return np.load(str(X_path)), np.load(str(y_path)), np.load(str(g_path))

    print("[build] Cache yok, feature cikarimi baslatiliyor (n_augment=2)...")
    from backend.pipeline.data_loader import QilinDatasetLoader
    loader = QilinDatasetLoader()
    X, y = loader.extract_multimodal_features(
        augment=True, n_augment=2, use_xigua=False,
    )
    groups = loader._last_groups
    groups = np.array([int(g) for g in groups])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(X_path), X)
    np.save(str(y_path), y)
    np.save(str(g_path), groups)
    return X, y, groups


def run_binary_lowo(X, y_binary, groups, model_type: str):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import confusion_matrix, classification_report
    from backend.module_e.classifier import RipenessClassifier

    unique_groups = np.unique(groups)
    logo = LeaveOneGroupOut()

    fold_accs = []
    all_true = []
    all_pred = []

    for fold_idx, (tr, te) in enumerate(logo.split(X, y_binary, groups)):
        held = np.unique(groups[te])[0]
        clf = RipenessClassifier(model_type=model_type)
        clf.train(X[tr], y_binary[tr])
        y_pred, _ = clf.predict(X[te])
        acc = float(np.mean(y_pred == y_binary[te]))
        fold_accs.append(acc)
        all_true.extend(y_binary[te].tolist())
        all_pred.extend(y_pred.tolist())
        print(f"   [{model_type}] Fold {fold_idx+1}/{len(unique_groups)} "
              f"karpuz={held} true_class={np.unique(y_binary[te]).tolist()} acc={acc:.4f}")

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    report = classification_report(
        all_true, all_pred, labels=[0, 1],
        target_names=["Yenmez", "Yenir"], output_dict=True, zero_division=0,
    )

    return {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "min_accuracy": float(np.min(fold_accs)),
        "max_accuracy": float(np.max(fold_accs)),
        "confusion_matrix": cm.tolist(),
        "report": report,
        "n_folds": len(unique_groups),
    }


def main():
    print("=" * 60)
    print("2-Sinif (Yenir/Yenmez) LOWO Degerlendirme")
    print("=" * 60)

    X, y, groups = load_or_build_features()
    print(f"X: {X.shape} | y: {y.shape} | groups: {groups.shape}")

    # 3-sinif -> 2-sinif remap
    y_bin = np.where(y == 1, 1, 0).astype(np.int64)
    u, c = np.unique(y_bin, return_counts=True)
    print(f"Binary dagilim: {dict(zip(['Yenmez','Yenir'], c.tolist()))}")

    results = {}
    for mt in ["knn", "rfc"]:
        print(f"\n--- LOWO: {mt.upper()} ---")
        results[mt] = run_binary_lowo(X, y_bin, groups, mt)
        r = results[mt]
        print(f"\n[{mt.upper()}] LOWO Mean: {r['mean_accuracy']:.4f} "
              f"(std={r['std_accuracy']:.4f}, min={r['min_accuracy']:.4f}, "
              f"max={r['max_accuracy']:.4f})")
        print(f"  Confusion Matrix:")
        for row in r["confusion_matrix"]:
            print(f"    {row}")
        for name in ["Yenmez", "Yenir"]:
            s = r["report"][name]
            print(f"  {name}: precision={s['precision']:.3f} "
                  f"recall={s['recall']:.3f} f1={s['f1-score']:.3f}")

    # Sonuclari kaydet
    out = {
        "timestamp": datetime.now().isoformat(),
        "task": "binary_edibility",
        "class_map": {"0": "Yenmez (Olgunlasmamis+IciGecmis)", "1": "Yenir (Olgun)"},
        "dataset": {"n_samples": int(len(y)), "n_groups": int(len(np.unique(groups)))},
        "lowo": results,
    }
    out_path = MODELS_DIR / "binary_lowo_results.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar: {out_path}")


if __name__ == "__main__":
    main()
