"""
Birlesik Gorsel Dataset Yukleyicisi

Uc farkli kaynaktan gorsel veri yukler ve birlestirip
MobileNetV3 egitimi icin hazirlar:

  1. Roboflow Watermelon Ripeness Grading v7 (54 gorsel, 3-sinif CSV)
  2. MRD-YOLO test seti (300 gorsel, 2-sinif YOLO label)
  3. Qilin 19-datasets JPG (1557 gorsel, Brix → 3-sinif)

Sinif haritalamasi:
  0 = Olgunlasmamis (underripe / unripe / immature)
  1 = Olgun (ripe / mature)
  2 = Ici Gecmis (overripe / over-mature)
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import BRIX_IMMATURE_THRESHOLD, BRIX_OVERRIPE_THRESHOLD


PROJECT_ROOT = Path(__file__).parent.parent.parent
ROBOFLOW_DIR = PROJECT_ROOT / "watermelon ripeness grading.v7i.multiclass"
MRD_IMAGES_DIR = PROJECT_ROOT / "mrd_yolo_repo" / "dataset" / "images" / "test"
MRD_LABELS_DIR = PROJECT_ROOT / "mrd_yolo_repo" / "dataset" / "labels" / "test"
QILIN_DATASETS_DIR = PROJECT_ROOT / "dataset" / "datasets" / "watermelon_dataset" / "19_datasets"

IMAGE_SIZE = 224  # MobileNetV3 input


class UnifiedVisualDatasetLoader:
    """
    Roboflow + MRD-YOLO + Qilin gorsellerini yukler.

    Her kaynak icin ayri loader fonksiyonu bulunur.
    get_unified_dataset() tum kaynaklari birlestirip
    numpy array olarak doner.
    """

    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size

    # ------------------------------------------------------------------
    # 1) Roboflow
    # ------------------------------------------------------------------
    def load_roboflow(self) -> Tuple[List[str], List[int]]:
        """
        Roboflow CSV multi-label formatini yukler.

        Multi-label → single-label donusumu:
          - Tek sinif 1 ise → o sinif
          - Birden fazla sinif 1 ise → overripe > underripe > ripe onceligi
          - Hic sinif yoksa → atla

        Returns:
            (image_paths, labels)
        """
        image_paths: List[str] = []
        labels: List[int] = []

        for split in ["train", "valid", "test"]:
            csv_path = ROBOFLOW_DIR / split / "_classes.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row["filename"]
                    overripe = int(row.get("overripe", 0))
                    ripe = int(row.get("ripe", 0))
                    underripe = int(row.get("underripe", 0))

                    # Single-label donusumu
                    active = overripe + ripe + underripe
                    if active == 0:
                        continue

                    if active == 1:
                        if underripe:
                            label = 0
                        elif ripe:
                            label = 1
                        else:
                            label = 2
                    else:
                        # Multi-label → oncelik: overripe > underripe > ripe
                        if overripe:
                            label = 2
                        elif underripe:
                            label = 0
                        else:
                            label = 1

                    img_path = str(ROBOFLOW_DIR / split / filename)
                    if os.path.exists(img_path):
                        image_paths.append(img_path)
                        labels.append(label)

        print(f"  [Roboflow] {len(image_paths)} gorsel yuklendi")
        self._print_dist(labels, "Roboflow")
        return image_paths, labels

    # ------------------------------------------------------------------
    # 2) MRD-YOLO
    # ------------------------------------------------------------------
    def load_mrd_yolo(self) -> Tuple[List[str], List[int]]:
        """
        MRD-YOLO test setinden karpuz bolgesini crop ederek siniflandirma
        dataseti olusturur.

        YOLO label formati: class_id cx cy w h (normalized)
        class 0 = unripe → sinif 0
        class 1 = ripe   → sinif 1

        Her gorselden en buyuk bounding box secilir.

        Returns:
            (image_paths, labels)  — image_paths orijinal dosya yollari,
            crop islemi get_unified_dataset() icinde yapilir.
        """
        image_paths: List[str] = []
        labels: List[int] = []
        bbox_data: List[Dict] = []

        if not MRD_LABELS_DIR.exists() or not MRD_IMAGES_DIR.exists():
            print("  [MRD-YOLO] Dataset bulunamadi, atlanıyor")
            return image_paths, labels

        for label_file in sorted(MRD_LABELS_DIR.glob("*.txt")):
            stem = label_file.stem
            # Eslesen gorsel bul
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = MRD_IMAGES_DIR / f"{stem}{ext}"
                if candidate.exists():
                    img_path = str(candidate)
                    break
            if img_path is None:
                continue

            # Label dosyasini oku — en buyuk bbox'u sec
            best_area = 0
            best_cls = -1
            best_box = None
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_cls = cls_id
                        best_box = (cx, cy, w, h)

            if best_cls < 0 or best_box is None:
                continue

            # MRD class → proje class haritalamasi
            # MRD class 0=unripe → 0, class 1=ripe → 1
            label = min(best_cls, 1)  # Guvenlik: 0 veya 1

            image_paths.append(img_path)
            labels.append(label)
            bbox_data.append({
                "cx": best_box[0], "cy": best_box[1],
                "w": best_box[2], "h": best_box[3],
            })

        self._mrd_bbox_data = bbox_data
        print(f"  [MRD-YOLO] {len(image_paths)} gorsel yuklendi")
        self._print_dist(labels, "MRD-YOLO")
        return image_paths, labels

    # ------------------------------------------------------------------
    # 3) Qilin
    # ------------------------------------------------------------------
    def load_qilin_images(self) -> Tuple[List[str], List[int], List[int]]:
        """
        Qilin 19-datasets'ten JPG gorsellerini ve Brix-derived etiketlerini yukler.

        Returns:
            (image_paths, labels, watermelon_ids)
        """
        image_paths: List[str] = []
        labels: List[int] = []
        watermelon_ids: List[int] = []

        if not QILIN_DATASETS_DIR.exists():
            print(f"  [Qilin] Dataset dizini bulunamadi: {QILIN_DATASETS_DIR}")
            return image_paths, labels, watermelon_ids

        for subdir in sorted(os.listdir(str(QILIN_DATASETS_DIR))):
            subdir_path = QILIN_DATASETS_DIR / subdir
            if not subdir_path.is_dir():
                continue
            try:
                parts = subdir.rsplit("_", 1)
                data_id = int(parts[0])
                brix = float(parts[1])
            except (ValueError, IndexError):
                continue

            # Brix → kategori
            if brix < BRIX_IMMATURE_THRESHOLD:
                label = 0
            elif brix > BRIX_OVERRIPE_THRESHOLD:
                label = 2
            else:
                label = 1

            chu_dir = subdir_path / "chu"
            if not chu_dir.is_dir():
                continue

            for folder in sorted(os.listdir(str(chu_dir))):
                folder_path = chu_dir / folder
                if not folder_path.is_dir():
                    continue
                jpgs = [f for f in os.listdir(str(folder_path))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if jpgs:
                    img_path = str(folder_path / jpgs[0])
                    image_paths.append(img_path)
                    labels.append(label)
                    watermelon_ids.append(data_id)

        print(f"  [Qilin] {len(image_paths)} gorsel yuklendi")
        self._print_dist(labels, "Qilin")
        return image_paths, labels, watermelon_ids

    # ------------------------------------------------------------------
    # Birlestirme
    # ------------------------------------------------------------------
    def get_unified_dataset(
        self,
        augment_roboflow: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Tum kaynaklari birlestirip (images, labels, groups) doner.

        Args:
            augment_roboflow: Her Roboflow gorseli icin augment kopyasi

        Returns:
            X_images: (N, image_size, image_size, 3) uint8
            y_labels: (N,) int
            groups:   (N,) int — watermelon ID (Qilin icin gercek, diger icin -1/-2)
        """
        print("\n[UnifiedVisualDatasetLoader] Birlestirme basladi...")

        all_images: List[np.ndarray] = []
        all_labels: List[int] = []
        all_groups: List[int] = []

        # --- 1) Roboflow ---
        rf_paths, rf_labels = self.load_roboflow()
        for i, (path, label) in enumerate(zip(rf_paths, rf_labels)):
            img = self._load_and_resize(path)
            if img is None:
                continue
            all_images.append(img)
            all_labels.append(label)
            all_groups.append(-1)  # Roboflow grubu

            # Augmentation
            for _ in range(augment_roboflow):
                aug_img = self._augment_image(img)
                all_images.append(aug_img)
                all_labels.append(label)
                all_groups.append(-1)

        # --- 2) MRD-YOLO (crop) ---
        mrd_paths, mrd_labels = self.load_mrd_yolo()
        bbox_data = getattr(self, "_mrd_bbox_data", [{}] * len(mrd_paths))
        for i, (path, label) in enumerate(zip(mrd_paths, mrd_labels)):
            img = self._safe_imread(path)
            if img is None:
                continue
            # Bbox ile crop
            if i < len(bbox_data) and bbox_data[i]:
                bb = bbox_data[i]
                h_img, w_img = img.shape[:2]
                x1 = max(0, int((bb["cx"] - bb["w"] / 2) * w_img))
                y1 = max(0, int((bb["cy"] - bb["h"] / 2) * h_img))
                x2 = min(w_img, int((bb["cx"] + bb["w"] / 2) * w_img))
                y2 = min(h_img, int((bb["cy"] + bb["h"] / 2) * h_img))
                if x2 > x1 + 10 and y2 > y1 + 10:
                    img = img[y1:y2, x1:x2]

            img = cv2.resize(img, (self.image_size, self.image_size))
            all_images.append(img)
            all_labels.append(label)
            all_groups.append(-2)  # MRD grubu

        # --- 3) Qilin ---
        q_paths, q_labels, q_ids = self.load_qilin_images()
        for path, label, wm_id in zip(q_paths, q_labels, q_ids):
            img = self._load_and_resize(path)
            if img is None:
                continue
            all_images.append(img)
            all_labels.append(label)
            all_groups.append(wm_id)

        X = np.array(all_images, dtype=np.uint8)
        y = np.array(all_labels, dtype=np.int64)
        groups = np.array(all_groups, dtype=np.int64)

        print(f"\n  [Toplam] {len(X)} gorsel")
        self._print_dist(y.tolist(), "Birlesik")
        return X, y, groups

    # ------------------------------------------------------------------
    # Yardimci
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_imread(path: str) -> Optional[np.ndarray]:
        """Turkce/ozel karakterli path'lerde cv2.imread bozulur, imdecode kullan."""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _load_and_resize(self, path: str) -> Optional[np.ndarray]:
        """Gorseli yukle ve image_size x image_size boyutuna getir."""
        img = self._safe_imread(path)
        if img is None:
            return None
        return cv2.resize(img, (self.image_size, self.image_size))

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Basit gorsel augmentasyon (OpenCV tabanli, ek paket gerektirmez)."""
        aug = img.copy()
        h, w = aug.shape[:2]

        # 1) Random horizontal flip (%50)
        if np.random.rand() < 0.5:
            aug = cv2.flip(aug, 1)

        # 2) Random rotation (-20, +20 derece)
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 3) Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Kontrast
        beta = np.random.uniform(-20, 20)     # Parlaklik
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # 4) Random crop + resize
        crop_pct = np.random.uniform(0.80, 1.0)
        new_h, new_w = int(h * crop_pct), int(w * crop_pct)
        y1 = np.random.randint(0, h - new_h + 1)
        x1 = np.random.randint(0, w - new_w + 1)
        aug = aug[y1:y1 + new_h, x1:x1 + new_w]
        aug = cv2.resize(aug, (w, h))

        # 5) Random Gaussian blur (%30)
        if np.random.rand() < 0.3:
            ksize = np.random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

        return aug

    @staticmethod
    def _print_dist(labels, name: str):
        """Sinif dagilimini yazdirir."""
        from collections import Counter
        c = Counter(labels)
        names = {0: "underripe", 1: "ripe", 2: "overripe"}
        parts = [f"{names.get(k, k)}={v}" for k, v in sorted(c.items())]
        print(f"    Dagilim: {', '.join(parts)}")


if __name__ == "__main__":
    loader = UnifiedVisualDatasetLoader()
    X, y, groups = loader.get_unified_dataset(augment_roboflow=8)
    print(f"\nSon: X.shape={X.shape}, y.shape={y.shape}, groups.shape={groups.shape}")
