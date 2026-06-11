"""
Qilin Akustik Kol Overfit / Leakage Analizi

Amaç:
  - Aynı karpuza ait birden fazla vuruş (sample) varsa,
    bunların train ve test setine karışıp karışmadığını kontrol etmek.
  - Sadece meyve (data_id) bazlı split ile RFC / KNN doğruluğunu
    ölçerek gerçek generalizasyon seviyesini görmek.

Kullanım:
  python -m backend.pipeline.eval_acoustic_leakage
  python -m backend.pipeline.eval_acoustic_leakage --seeds 1 2 3 4 5
"""

import os
import argparse
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from backend.pipeline.data_loader import QilinDatasetLoader


def load_features_with_ids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    İşlenmiş özellikleri ve her sample'ın data_id'sini döndürür.

    NOT:
      - train.py şu an sadece X (features) ve y (labels) kaydediyor.
      - Burada leakage analizini daha doğru yapmak için
        QilinDatasetLoader üzerinden labels_df (içinde data_id var)
        ve extract_features_from_audio'yı BİRLİKTE kullanıyoruz.
    """
    loader = QilinDatasetLoader()

    # Qilin orijinal yapısını doğrudan 19_datasets altından oku
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    qilin_root = os.path.join(project_root, "dataset", "datasets", "watermelon_dataset", "19_datasets")

    discovery = loader._discover_qilin_original(qilin_root)
    samples = discovery.get("qilin_samples", [])
    if not samples:
        # Fallback: auto_discover
        labels_df = loader.load_labels()
    else:
        labels_df = loader._qilin_samples_to_dataframe(samples)

    # Qilin orijinal yapısında data_id zaten mevcut
    if "data_id" not in labels_df.columns:
        # Güvenli tarafta kalmak için, data_id yoksa her sample'ı
        # ayrı karpuz gibi kabul ediyoruz (bu durumda leakage riski düşük).
        labels_df["data_id"] = np.arange(len(labels_df)).astype(str)

    # Özellikleri yeniden çıkar (x_features.npy yerine)
    X, y = loader.extract_features_from_audio(labels_df=labels_df)

    # labels_df ile X aynı sırada olduğundan, data_id vektörünü alabiliriz
    data_ids = labels_df["data_id"].astype(str).values[: len(X)]

    return X, y, data_ids


def fruit_level_split(
    X: np.ndarray,
    y: np.ndarray,
    data_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Meyve (data_id) bazlı train/test bölmesi.

    Adımlar:
      1) Tüm sample'ları data_id'ye göre grupla
      2) data_id setini train/test diye böl
      3) X/y'yi bu bölüme göre ayır
    """
    rng = np.random.default_rng(random_state)

    # data_id -> index list
    fruit_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, fid in enumerate(data_ids):
        fruit_to_indices[str(fid)].append(idx)

    all_fruits = list(fruit_to_indices.keys())
    rng.shuffle(all_fruits)

    n_fruits = len(all_fruits)
    n_test = max(1, int(n_fruits * test_size))

    test_fruits = set(all_fruits[:n_test])
    train_fruits = set(all_fruits[n_test:])

    train_indices: list[int] = []
    test_indices: list[int] = []

    for fid, idx_list in fruit_to_indices.items():
        if fid in test_fruits:
            test_indices.extend(idx_list)
        else:
            train_indices.extend(idx_list)

    train_indices = np.array(sorted(train_indices))
    test_indices = np.array(sorted(test_indices))

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test, train_fruits, test_fruits


def evaluate_with_seeds(seeds: list[int], test_size: float = 0.2):
    """
    Birden fazla seed ile meyve-bazlı split üzerinde RFC ve KNN'yi test eder.
    """
    print("=" * 70)
    print("Qilin Akustik Kol - Meyve Bazlı Overfit / Leakage Analizi")
    print("=" * 70)

    X, y, data_ids = load_features_with_ids()
    print(f"[INFO] Toplam sample: {len(X)}, feature_dim={X.shape[1]}")
    unique_fruits = np.unique(data_ids)
    print(f"[INFO] Toplam karpuz (data_id): {len(unique_fruits)}")

    rfc_acc_list = []
    knn_acc_list = []

    for seed in seeds:
        print(f"\n--- Seed = {seed} ---")
        X_train, X_test, y_train, y_test, train_fruits, test_fruits = fruit_level_split(
            X, y, data_ids, test_size=test_size, random_state=seed
        )
        print(f"[Split] Train fruits: {len(train_fruits)}, Test fruits: {len(test_fruits)}")
        print(f"[Split] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # KNN (doğrudan sklearn ile, rapor üretmeden)
        knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        knn_model.fit(X_train, y_train)
        knn_acc = knn_model.score(X_test, y_test)
        knn_acc_list.append(knn_acc)
        print(f"[KNN] Test doğruluğu (fruit-level): {knn_acc:.4f}")

        # RFC (direkt sklearn, conservative parametrelerle)
        rfc_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=seed,
        )
        rfc_model.fit(X_train, y_train)
        rfc_acc = rfc_model.score(X_test, y_test)
        rfc_acc_list.append(rfc_acc)
        print(f"[RFC] Test doğruluğu (fruit-level): {rfc_acc:.4f}")

    print("\n" + "=" * 70)
    print("ÖZET (Fruit-level split, çoklu seed)")
    print("=" * 70)
    print(f"KNN accuracy: mean={np.mean(knn_acc_list):.4f}, std={np.std(knn_acc_list):.4f}, "
          f"min={np.min(knn_acc_list):.4f}, max={np.max(knn_acc_list):.4f}")
    print(f"RFC accuracy: mean={np.mean(rfc_acc_list):.4f}, std={np.std(rfc_acc_list):.4f}, "
          f"min={np.min(rfc_acc_list):.4f}, max={np.max(rfc_acc_list):.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Qilin akustik kol overfit/leakage analizi")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="Meyve-bazlı split için kullanılacak random seed listesi",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test seti oranı (meyve bazında)",
    )

    args = parser.parse_args()
    evaluate_with_seeds(args.seeds, test_size=args.test_size)


if __name__ == "__main__":
    main()


