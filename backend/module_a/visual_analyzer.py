"""
MODULE_A: Gorsel Analiz Modulu - MRD-YOLO Entegrasyonu

MRD-YOLO (Melon Ripeness Detection YOLO) modeli kullanarak
karpuzun gorsel olgunluk tespiti yapar.

Kaynak Repo: https://github.com/XuebinJing/Melon-Ripeness-Detection

MRD-YOLO Mimarisi:
  - Backbone: MobileNetV3 (Lightweight, SE Block + Inverted Residual)
  - Neck: VoVGSCSP + GSConv (Ghost Shuffle Conv)
  - Head: Detect (P3/P4/P5) + CoordAtt (Coordinate Attention)
  - Siniflar: 2 (unripe, ripe)
  - Dataset: 300 test goruntusu (YOLO format)

Ek Analizler:
  - Tarla lekesi (field spot) renk analizi
  - Kabuk dokusu (rind texture) degerlendirmesi
  - Gorsel ozellik vektoru cikarimi (Late Fusion icin)

Matematiksel Altyapi (Koc & Akbalik, 2025):
  - Kayip: L_total = lambda_loc*L_CIoU + lambda_cls*L_focal + lambda_conf*L_BCE
  - Basari: %96.04 mAP@0.5
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import glob

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    YOLO_MODEL_PATH,
    MRD_YOLO_MODEL_PATH,
    MRD_YOLO_FALLBACK_PATH,
    MRD_ULTRALYTICS_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_NMS_IOU_THRESHOLD,
    IMAGE_INPUT_SIZE,
    MRD_CLASS_NAMES,
    MRD_NUM_CLASSES,
    MRD_DATASET_DIR,
    MRD_DATASET_YAML,
    MRD_TEST_IMAGES,
    LOSS_LAMBDA_LOCALIZATION,
    LOSS_LAMBDA_CLASSIFICATION,
    LOSS_LAMBDA_CONFIDENCE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    CLASS_ALPHA_WEIGHTS,
    OPTIMAL_CONFIDENCE_THRESHOLD
)


# =================================================================
# MRD-YOLO MODEL YUKLEME
# =================================================================

def _resolve_mrd_model_path() -> Optional[str]:
    """MRD.pt model dosyasinin konumunu bulur."""
    candidates = [
        MRD_YOLO_MODEL_PATH,           # model/MRD.pt
        MRD_YOLO_FALLBACK_PATH,        # mrd_yolo_repo/weights/MRD.pt
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_mrd_yolo(model_path: str):
    """
    MRD-YOLO modelini yukler.
    
    MRD-YOLO ozel ultralytics modifikasyonlari gerektirir:
      - MobileNetV3_InvertedResidual (backbone)
      - CoordAtt (attention)
      - VoVGSCSP, GSConv (neck)
      - EMA (training)
    
    Bu modüller mrd_yolo_repo/ultralytics/ icinde tanimlidir.
    """
    # Oncelikle modifiye ultralytics'i sys.path'e ekle
    # Bu sayede ozel modüller (MobileNetV3, CoordAtt vb.) import edilebilir
    if os.path.exists(MRD_ULTRALYTICS_PATH):
        if MRD_ULTRALYTICS_PATH not in sys.path:
            sys.path.insert(0, MRD_ULTRALYTICS_PATH)
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"[MRD-YOLO] Model yukleme hatasi: {e}")
        return None


# =================================================================
# ANA SINIF: VisualAnalyzer
# =================================================================

class VisualAnalyzer:
    """
    Karpuz gorsel olgunluk analizi.
    
    MRD-YOLO modelini kullanarak:
      1. Goruntude karpuz tespiti (object detection)
      2. Olgunluk siniflandirmasi (unripe/ripe)
      3. Tarla lekesi ve kabuk dokusu analizi
      4. Gorsel skor hesaplama (Late Fusion icin)
    """

    def __init__(
        self,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        nms_iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
        auto_load: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.mrd_model = None
        self.model_path = None
        self.model_loaded = False
        
        if auto_load:
            self._load_model()

    def _load_model(self):
        """MRD-YOLO modelini yukler."""
        model_path = _resolve_mrd_model_path()
        
        if model_path is None:
            print("[MRD-YOLO] UYARI: MRD.pt model dosyasi bulunamadi!")
            print("  Beklenen konumlar:")
            print(f"    1. {MRD_YOLO_MODEL_PATH}")
            print(f"    2. {MRD_YOLO_FALLBACK_PATH}")
            return
        
        print(f"[MRD-YOLO] Model yukleniyor: {model_path}")
        self.mrd_model = _load_mrd_yolo(model_path)
        
        if self.mrd_model is not None:
            self.model_path = model_path
            self.model_loaded = True
            print(f"[MRD-YOLO] Model basariyla yuklendi!")
            print(f"  Siniflar: {MRD_CLASS_NAMES}")
            print(f"  Confidence esigi: {self.confidence_threshold}")
        else:
            print("[MRD-YOLO] Model yuklenemedi!")

    # =============================================================
    # TESPIT VE SINIFLANDIRMA
    # =============================================================

    def detect_and_classify(
        self,
        image: np.ndarray,
        conf: float = None,
        iou: float = None
    ) -> List[Dict]:
        """
        Goruntude karpuz tespiti ve olgunluk siniflandirmasi yapar.
        
        MRD-YOLO cikisi: her tespit icin
          - bbox (x1, y1, x2, y2)
          - confidence (0-1)
          - class_id (0=unripe, 1=ripe)
          - class_name
        
        Args:
            image: BGR formatinda giris goruntusu
            conf: Confidence esigi (None ise varsayilan)
            iou: NMS IoU esigi
            
        Returns:
            Tespit listesi
        """
        if not self.model_loaded:
            return []
        
        conf = conf or self.confidence_threshold
        iou = iou or self.nms_iou_threshold
        
        try:
            results = self.mrd_model(
                image,
                conf=conf,
                iou=iou,
                verbose=False
            )
        except Exception as e:
            print(f"[MRD-YOLO] Tespit hatasi: {e}")
            return []
        
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # Sinif adini al
                class_name = MRD_CLASS_NAMES.get(
                    class_id,
                    result.names.get(class_id, f"class_{class_id}")
                )
                
                detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "is_ripe": class_id == 1,
                    "bbox_width": bbox[2] - bbox[0],
                    "bbox_height": bbox[3] - bbox[1],
                    "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                }
                detections.append(detection)
        
        # Confidence'a gore sirala
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def predict_single(
        self,
        image: np.ndarray
    ) -> Dict:
        """
        Tek bir karpuz goruntusu icin sonuc dondurur.
        
        En yuksek confidence'li tespiti dondurur.
        Tespit yoksa gorsel analiz (field spot + texture) yapar.
        
        Args:
            image: BGR giris goruntusu
            
        Returns:
            Olgunluk tahmini
        """
        detections = self.detect_and_classify(image)
        
        if detections:
            best = detections[0]
            return {
                "source": "mrd_yolo",
                "class_id": best["class_id"],
                "class_name": best["class_name"],
                "confidence": best["confidence"],
                "is_ripe": best["is_ripe"],
                "bbox": best["bbox"],
                "n_detections": len(detections),
                "all_detections": detections
            }
        else:
            # Model tespiti yoksa fallback gorsel analiz
            visual_result = self.get_visual_ripeness_score(image)
            return {
                "source": "visual_heuristic",
                "class_id": visual_result["visual_class"],
                "class_name": MRD_CLASS_NAMES.get(
                    visual_result["visual_class"], "Bilinmiyor"
                ),
                "confidence": visual_result["visual_score"],
                "is_ripe": visual_result["visual_class"] == 1,
                "bbox": None,
                "n_detections": 0,
                "visual_analysis": visual_result
            }

    # =============================================================
    # BATCH TESPIT (300 Test Goruntusu)
    # =============================================================

    def evaluate_test_set(
        self,
        test_dir: str = None,
        max_images: int = None
    ) -> Dict:
        """
        MRD-YOLO test seti uzerinde degerlendirme yapar.
        
        Args:
            test_dir: Test goruntuleri dizini
            max_images: Maksimum goruntu sayisi (None=hepsi)
            
        Returns:
            Degerlendirme sonuclari
        """
        if test_dir is None:
            test_dir = str(MRD_TEST_IMAGES)
        
        if not os.path.exists(test_dir):
            return {"error": f"Test dizini bulunamadi: {test_dir}"}
        
        # Goruntuleri bul
        image_paths = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        
        if not image_paths:
            return {"error": f"Test goruntusu bulunamadi: {test_dir}"}
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"\n[MRD-YOLO] Test seti degerlendirmesi: {len(image_paths)} goruntu")
        
        results = {
            "total_images": len(image_paths),
            "total_detections": 0,
            "class_counts": {0: 0, 1: 0},
            "confidences": [],
            "no_detection_count": 0,
            "detections_per_image": []
        }
        
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            dets = self.detect_and_classify(image)
            results["total_detections"] += len(dets)
            results["detections_per_image"].append(len(dets))
            
            if not dets:
                results["no_detection_count"] += 1
            else:
                for d in dets:
                    cls_id = d["class_id"]
                    if cls_id in results["class_counts"]:
                        results["class_counts"][cls_id] += 1
                    results["confidences"].append(d["confidence"])
            
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(image_paths)}] islendi...")
        
        # Istatistikler
        if results["confidences"]:
            confs = np.array(results["confidences"])
            results["mean_confidence"] = float(np.mean(confs))
            results["min_confidence"] = float(np.min(confs))
            results["max_confidence"] = float(np.max(confs))
        
        results["avg_detections_per_image"] = (
            results["total_detections"] / results["total_images"]
            if results["total_images"] > 0 else 0
        )
        
        return results

    def validate_model(self) -> Optional[Dict]:
        """
        MRD-YOLO modelini YOLO val() API ile resmi degerlendirme yapar.
        
        Returns:
            mAP@0.5, mAP@0.5:0.95, mAP@0.75 metrikleri
        """
        if not self.model_loaded:
            print("[MRD-YOLO] Model yuklu degil!")
            return None
        
        yaml_path = str(MRD_DATASET_YAML)
        if not os.path.exists(yaml_path):
            print(f"[MRD-YOLO] Dataset YAML bulunamadi: {yaml_path}")
            return None
        
        try:
            print(f"[MRD-YOLO] Resmi degerlendirme baslatiliyor...")
            print(f"  Dataset: {yaml_path}")
            
            metrics = self.mrd_model.val(
                data=yaml_path,
                split='test',
                batch=1,
                verbose=False
            )
            
            result = {
                "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75),
                "map50_95": float(metrics.box.map),
                "maps_per_class": [float(m) for m in metrics.box.maps],
                "class_names": MRD_CLASS_NAMES
            }
            
            print(f"  mAP@0.5:    {result['map50']:.4f}")
            print(f"  mAP@0.75:   {result['map75']:.4f}")
            print(f"  mAP@0.5:0.95: {result['map50_95']:.4f}")
            
            return result
        except Exception as e:
            print(f"[MRD-YOLO] Degerlendirme hatasi: {e}")
            return None

    # =============================================================
    # GORSEL ANALIZ (Field Spot + Texture)
    # =============================================================

    def analyze_field_spot(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, float]:
        """
        Tarla lekesi (field spot) renk analizi.
        
        Karpuzun yere temas eden kismindaki sari lekenin rengi
        olgunluk gostergesidir:
        - Beyaz/acik yesil -> olgunlasmamis
        - Kremsi sari -> olgun
        - Koyu sari/turuncu -> asiri olgun
        """
        if roi is None:
            h, w = image.shape[:2]
            roi = (0, int(h * 0.7), w, h)

        x1, y1, x2, y2 = roi
        field_spot = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(field_spot, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        yellow_mask = cv2.inRange(hsv, (15, 50, 50), (45, 255, 255))
        yellow_ratio = np.sum(yellow_mask > 0) / (yellow_mask.size + 1e-10)

        mean_hue = float(np.mean(h_channel))
        mean_saturation = float(np.mean(s_channel))
        mean_value = float(np.mean(v_channel))

        ripeness_score = min(1.0, yellow_ratio * 3.0 + mean_saturation / 255 * 0.3)

        return {
            "field_spot_yellow_ratio": yellow_ratio,
            "field_spot_mean_hue": mean_hue,
            "field_spot_mean_saturation": mean_saturation,
            "field_spot_mean_value": mean_value,
            "field_spot_ripeness_score": ripeness_score
        }

    def analyze_rind_texture(self, image: np.ndarray) -> Dict[str, float]:
        """
        Kabuk dokusu (rind texture) analizi.
        
        Olgun karpuzlarin kabugu daha mat ve puruzludur.
        Parlak kabuk olgunlasmamis karpuzu gosterir.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        mean_gradient = float(np.mean(gradient_magnitude))
        std_gradient = float(np.std(gradient_magnitude))
        brightness_std = float(np.std(gray))

        return {
            "texture_variance": texture_variance,
            "mean_gradient": mean_gradient,
            "std_gradient": std_gradient,
            "brightness_std": brightness_std,
            "matteness_score": min(1.0, texture_variance / 1000)
        }

    def get_visual_ripeness_score(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, object]:
        """
        Gorsel olgunluk skorunu hesaplar (Late Fusion icin).
        
        MRD-YOLO varsa model cikisi kullanilir,
        yoksa field spot + texture heristik kullanilir.
        """
        # MRD-YOLO ile tespit dene
        mrd_result = None
        if self.model_loaded:
            detections = self.detect_and_classify(image)
            if detections:
                best = detections[0]
                mrd_result = {
                    "mrd_class_id": best["class_id"],
                    "mrd_class_name": best["class_name"],
                    "mrd_confidence": best["confidence"],
                    "mrd_is_ripe": best["is_ripe"],
                    "mrd_bbox": best["bbox"]
                }
        
        # Heristik analiz
        field_spot = self.analyze_field_spot(image, roi)
        texture = self.analyze_rind_texture(image)

        heuristic_score = (
            field_spot["field_spot_ripeness_score"] * 0.6 +
            texture["matteness_score"] * 0.4
        )

        # MRD-YOLO varsa model cikisini agirlikla birlestir
        if mrd_result:
            mrd_score = mrd_result["mrd_confidence"] if mrd_result["mrd_is_ripe"] else (1 - mrd_result["mrd_confidence"])
            # Model %70, Heristik %30
            visual_score = 0.70 * mrd_score + 0.30 * heuristic_score
        else:
            visual_score = heuristic_score

        # Siniflandirma (2 sinif: 0=unripe, 1=ripe)
        if visual_score < 0.5:
            visual_class = 0  # Olgunlasmamis
        else:
            visual_class = 1  # Olgun

        result = {
            "visual_score": visual_score,
            "visual_class": visual_class,
            "visual_class_name": MRD_CLASS_NAMES.get(visual_class, "Bilinmiyor"),
            "field_spot_analysis": field_spot,
            "texture_analysis": texture,
            "heuristic_score": heuristic_score
        }
        
        if mrd_result:
            result["mrd_yolo"] = mrd_result
        
        return result

    def extract_visual_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Gorsel ozellikleri duz vektore donusturur (fusion modeli icin).
        """
        result = self.get_visual_ripeness_score(image)

        fs = result["field_spot_analysis"]
        tx = result["texture_analysis"]

        vector = np.array([
            fs["field_spot_yellow_ratio"],
            fs["field_spot_mean_hue"],
            fs["field_spot_mean_saturation"],
            fs["field_spot_mean_value"],
            fs["field_spot_ripeness_score"],
            tx["texture_variance"],
            tx["mean_gradient"],
            tx["std_gradient"],
            tx["brightness_std"],
            tx["matteness_score"],
            result["visual_score"]
        ])

        return vector

    # =============================================================
    # GORSEL CIKTI
    # =============================================================

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict] = None,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Tespit sonuclarini goruntu uzerine cizer.
        
        Args:
            image: Orijinal goruntu
            detections: Tespit listesi (None ise otomatik tespit)
            show_labels: Etiketleri goster
            
        Returns:
            Uzerine cizimlenmis goruntu
        """
        output = image.copy()
        
        if detections is None:
            detections = self.detect_and_classify(image)
        
        colors = {
            0: (0, 0, 255),    # Kirmizi: Unripe
            1: (0, 255, 0),    # Yesil: Ripe
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            class_id = det["class_id"]
            conf = det["confidence"]
            name = det["class_name"]
            
            color = colors.get(class_id, (255, 255, 0))
            
            # Bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                label = f"{name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    output,
                    (x1, y1 - label_size[1] - 6),
                    (x1 + label_size[0], y1),
                    color, -1
                )
                cv2.putText(
                    output, label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1
                )
        
        return output

    # =============================================================
    # MODEL & MIMARI BILGI
    # =============================================================

    def get_model_info(self) -> Dict:
        """MRD-YOLO model ve mimari bilgilerini dondurur."""
        info = {
            "model_name": "MRD-YOLO",
            "source": "https://github.com/XuebinJing/Melon-Ripeness-Detection",
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "num_classes": MRD_NUM_CLASSES,
            "class_names": MRD_CLASS_NAMES,
            "confidence_threshold": self.confidence_threshold,
            "architecture": {
                "backbone": "MobileNetV3 (Lightweight - SE Block + Inverted Residual)",
                "neck": "VoVGSCSP + GSConv (Ghost Shuffle Convolution)",
                "head": "Detect P3/P4/P5",
                "attention": "CoordAtt (Coordinate Attention)",
                "detail": {
                    "backbone_blocks": [
                        "conv_bn_hswish(3, 16, stride=2)",
                        "MobileNetV3_InvertedResidual x11",
                        "SE Block (Squeeze & Excitation)"
                    ],
                    "neck_blocks": [
                        "VoVGSCSP (Varied on Versatile GSConv CSP)",
                        "GSConv 3x3 stride=2 (downsample)",
                        "CoordAtt (her seviyede)"
                    ],
                    "detection_scales": ["P3/8 (small)", "P4/16 (medium)", "P5/32 (large)"]
                }
            },
            "dataset": {
                "test_images": 300,
                "format": "YOLO (images/ + labels/)",
                "yaml": str(MRD_DATASET_YAML)
            }
        }
        return info

    def get_architecture_summary(self) -> str:
        """MRD-YOLO mimari ozetini metin olarak dondurur."""
        lines = [
            "=" * 60,
            "  MRD-YOLO Mimari Ozeti",
            "  (Melon Ripeness Detection)",
            "=" * 60,
            "",
            "  BACKBONE: MobileNetV3",
            "    - conv_bn_hswish(3, 16, stride=2)       # P1/2",
            "    - InvertedResidual(16, 16, k=3, s=2)     # P2/4",
            "    - InvertedResidual(24, 72, k=3, s=2)     # P3/8",
            "    - InvertedResidual(24, 88, k=3, s=1)",
            "    - InvertedResidual(40, 96, k=5, s=2, SE) # P4/16",
            "    - InvertedResidual x3 (40->48 ch, SE)",
            "    - InvertedResidual(96, 288, k=5, s=2, SE)# P5/32",
            "    - InvertedResidual x2 (96 ch, SE)",
            "",
            "  NECK (FPN-up + PAN-down):",
            "    - Upsample + Concat(P4)",
            "    - VoVGSCSP(512) + CoordAtt(512)          # P4'",
            "    - Upsample + Concat(P3)",
            "    - VoVGSCSP(256) + CoordAtt(256)          # P3' (small)",
            "    - GSConv(256, s=2) + Concat(P4')",
            "    - VoVGSCSP(512) + CoordAtt(512)          # P4'' (medium)",
            "    - GSConv(512, s=2) + Concat(P5)",
            "    - VoVGSCSP(1024) + CoordAtt(1024)        # P5'' (large)",
            "",
            "  HEAD: Detect([P3', P4'', P5''], nc=2)",
            "    - Siniflar: unripe (0), ripe (1)",
            "",
            "  DIKKAT MEKANIZMASI: CoordAtt (Coordinate Attention)",
            "    - Y ve X yonunde ayri ayri pooling",
            "    - Pozisyon bilgisi koruyan dikkat",
            "    - Her FPN/PAN seviyesinde uygulanir",
            "",
            "  EK MODULLER:",
            "    - EMA (Efficient Multi-Scale Attention)",
            "    - GAM_Attention (Global Attention Mechanism)",
            "    - SEAttention (Squeeze & Excitation)",
            "    - ShuffleAttention",
            "=" * 60
        ]
        return "\n".join(lines)

    # =============================================================
    # KOC & AKBALIK (2025) - MATEMATIKSEL BILESENLER
    # =============================================================

    def compute_detection_loss(
        self,
        box_pred: np.ndarray,
        box_gt: np.ndarray,
        cls_prob: float,
        cls_target: int,
        obj_score: float,
        obj_target: int = 1
    ) -> Dict[str, float]:
        """
        YOLOv11 toplam kayip fonksiyonunu hesaplar.
        
        L_total = lambda_loc * L_CIoU + lambda_cls * L_focal + lambda_conf * L_BCE
        """
        from backend.module_a.math_foundations import TotalLoss
        
        loss_fn = TotalLoss(
            lambda_loc=LOSS_LAMBDA_LOCALIZATION,
            lambda_cls=LOSS_LAMBDA_CLASSIFICATION,
            lambda_conf=LOSS_LAMBDA_CONFIDENCE,
            focal_alpha=FOCAL_LOSS_ALPHA,
            focal_gamma=FOCAL_LOSS_GAMMA
        )
        
        return loss_fn.compute(
            box_pred, box_gt,
            cls_prob, cls_target,
            obj_score, obj_target
        )

    @staticmethod
    def apply_cbam_attention(feature_map: np.ndarray) -> np.ndarray:
        """
        CBAM (Convolutional Block Attention Module) simulasyonu.
        Not: MRD-YOLO gercekte CoordAtt kullanir, bu referans icin saklanir.
        """
        if len(feature_map.shape) != 3:
            return feature_map
        
        H, W, C = feature_map.shape
        
        channel_avg = np.mean(feature_map, axis=(0, 1), keepdims=True)
        channel_max = np.max(feature_map, axis=(0, 1), keepdims=True)
        channel_attention = 1.0 / (1.0 + np.exp(-(channel_avg + channel_max)))
        F_prime = feature_map * channel_attention
        
        spatial_avg = np.mean(F_prime, axis=2, keepdims=True)
        spatial_max = np.max(F_prime, axis=2, keepdims=True)
        spatial_attention = 1.0 / (1.0 + np.exp(-(spatial_avg + spatial_max)))
        F_refined = F_prime * spatial_attention
        
        return F_refined

    def get_model_architecture_info(self) -> Dict:
        """YOLOv11 + MRD-YOLO mimari bilgilerini dondurur."""
        return {
            "paper": "Koc & Akbalik (2025)",
            "target_accuracy": "96.04% mAP@0.5",
            "mrd_yolo": {
                "backbone": "MobileNetV3 (SE Block + Inverted Residual)",
                "neck": "VoVGSCSP + GSConv",
                "head": "Detect P3/P4/P5",
                "attention": "CoordAtt (Coordinate Attention)",
            },
            "loss_function": {
                "total": f"L = {LOSS_LAMBDA_LOCALIZATION}*L_CIoU + {LOSS_LAMBDA_CLASSIFICATION}*L_focal + {LOSS_LAMBDA_CONFIDENCE}*L_BCE",
                "localization": "CIoU: 1 - IoU + rho^2/c^2 + alpha*v",
                "classification": f"Focal: -alpha_t*(1-p_t)^{FOCAL_LOSS_GAMMA}*log(p_t)",
                "confidence": "BCE: -[y*log(sigma(x)) + (1-y)*log(1-sigma(x))]"
            }
        }
