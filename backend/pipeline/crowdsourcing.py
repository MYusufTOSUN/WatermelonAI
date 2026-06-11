"""
Crowdsourcing / Kullanici Geri Bildirim Modulu

Kullanicilarin karpuz olgunluk tahminlerini gercek sonucla
karsilastirarak geri bildirim toplamasini saglar.

Bu modul su islevleri saglar:
  1. Tahmin sonuclari + kullanici geri bildirimi kaydi (FeedbackLogger)
  2. Geri bildirim verisini analiz ve ozet raporlama
  3. Yeniden egitim icin veri hazirlama (feedback -> training data)
  4. Flutter API icin JSON formatinda veri alisverisi

Veri Akisi (Crowdsourcing Loop):
  Kullanici karpuzu test eder
    -> Sistem tahmin uretir (class, confidence)
    -> Kullanici gercek sonucu girer (ground truth)
    -> FeedbackLogger kaydeder
    -> Yeterli veri biriktiginde yeniden egitim tetiklenir
    -> Model guncellenir

Dosya Yapisi:
  data/feedback/
    feedback_log.json       # Ana geri bildirim deposu
    feedback_export.csv     # CSV export (egitim icin)
    sessions/               # Oturum bazli kayitlar

Kullanim:
    from backend.pipeline.crowdsourcing import FeedbackLogger

    logger = FeedbackLogger()

    # Tahmin sonrasinda geri bildirim kaydet
    logger.log_feedback(
        prediction_class=1,
        prediction_confidence=0.87,
        user_label=2,              # Kullanicinin verdigi gercek sinif
        features=feature_vector,    # 120-dim akustik ozellikler
        hh_features=hh_vector,     # 8-dim HH gostergeleri
        session_id="user123",
        notes="Karpuz kesildikten sonra ici bosluklu cikti"
    )

    # Ozet rapor
    logger.print_summary()

    # Yeniden egitim icin veri hazirla
    X, y = logger.prepare_training_data()
"""

import numpy as np
import json
import csv
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    FEEDBACK_DIR,
    FEEDBACK_DB_PATH,
    FEEDBACK_EXPORT_PATH,
    FEEDBACK_CLASS_OPTIONS,
    FEEDBACK_MIN_CONFIDENCE_FOR_PROMPT,
    FEEDBACK_MAX_SAMPLES_PER_SESSION,
    FEEDBACK_RETRAIN_THRESHOLD,
    CLASS_LABELS,
    N_FEATURES,
    TFLITE_HH_INPUT_DIM,
)


class FeedbackLogger:
    """
    Kullanici geri bildirimi toplama ve yonetim sinifi.

    Her geri bildirim kaydi su bilgileri icerir:
      - Sistem tahmini (sinif, guven skoru, olasiliklar)
      - Kullanici geri bildirimi (gercek sinif, notlar)
      - Ozellik vektoru (120 akustik + 8 HH)
      - Metadata (zaman, oturum, cihaz bilgisi)

    Veriler JSON formatinda saklanir ve CSV olarak export edilir.
    """

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: Geri bildirim veritabani dosya yolu
        """
        self.db_path = db_path or FEEDBACK_DB_PATH
        self.feedback_dir = os.path.dirname(self.db_path)
        self.sessions_dir = os.path.join(self.feedback_dir, "sessions")

        # Dizinleri olustur
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)

        # Mevcut verileri yukle
        self._entries: List[Dict] = []
        self._load()

    # =========================================
    # VERI YUKLEME / KAYDETME
    # =========================================

    def _load(self):
        """Mevcut geri bildirim verilerini yukler."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
            except (json.JSONDecodeError, KeyError):
                self._entries = []
        else:
            self._entries = []

    def _save(self):
        """Geri bildirim verilerini diske kaydeder."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_entries": len(self._entries),
            "entries": self._entries
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    # =========================================
    # GERI BILDIRIM KAYDI
    # =========================================

    def log_feedback(
        self,
        prediction_class: int,
        prediction_confidence: float,
        user_label: int,
        features: Optional[np.ndarray] = None,
        hh_features: Optional[np.ndarray] = None,
        prediction_probabilities: Optional[Dict] = None,
        hh_score: Optional[float] = None,
        session_id: Optional[str] = None,
        device_info: Optional[str] = None,
        notes: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict:
        """
        Geri bildirim kaydeder.

        Args:
            prediction_class: Sistemin tahmin ettigi sinif (0, 1, 2)
            prediction_confidence: Sistemin guven skoru (0-1)
            user_label: Kullanicinin bildirdigi gercek sinif (0, 1, 2)
            features: 120-dim akustik ozellik vektoru (opsiyonel)
            hh_features: 8-dim HH gosterge vektoru (opsiyonel)
            prediction_probabilities: Sinif olasiliklari (opsiyonel)
            hh_score: Hollow Heart skoru (opsiyonel)
            session_id: Kullanici/oturum kimligi (opsiyonel)
            device_info: Cihaz bilgisi (opsiyonel)
            notes: Kullanici notlari (opsiyonel)
            image_path: Karpuz goruntusu yolu (opsiyonel)
            audio_path: Vurus sesi yolu (opsiyonel)

        Returns:
            Kaydedilen geri bildirim kaydi
        """
        entry_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        is_correct = (prediction_class == user_label)

        entry = {
            "id": entry_id,
            "timestamp": timestamp,
            "session_id": session_id or "anonymous",
            "prediction": {
                "class_id": int(prediction_class),
                "class_label": CLASS_LABELS.get(prediction_class, "Bilinmeyen"),
                "confidence": float(prediction_confidence),
                "probabilities": prediction_probabilities,
                "hh_score": float(hh_score) if hh_score is not None else None,
            },
            "user_feedback": {
                "class_id": int(user_label),
                "class_label": CLASS_LABELS.get(user_label, "Bilinmeyen"),
                "is_correct": is_correct,
                "notes": notes,
            },
            "features": {
                "acoustic_vector": features.tolist() if features is not None else None,
                "hh_vector": hh_features.tolist() if hh_features is not None else None,
            },
            "media": {
                "image_path": image_path,
                "audio_path": audio_path,
            },
            "device_info": device_info,
        }

        self._entries.append(entry)
        self._save()

        # Oturum dosyasina da kaydet
        self._save_to_session(entry)

        # Yeniden egitim kontrolu
        total = len(self._entries)
        if total % FEEDBACK_RETRAIN_THRESHOLD == 0 and total > 0:
            print(f"\n  [Crowdsourcing] {total} geri bildirim birikti!")
            print(f"  [Crowdsourcing] Yeniden egitim oneriliyor.")
            print(f"  Komut: python main.py --retrain-with-feedback")

        return entry

    def _save_to_session(self, entry: Dict):
        """Geri bildirimi oturum dosyasina da kaydeder."""
        session_id = entry.get("session_id", "anonymous")
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")

        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                session_data = {"entries": []}
        else:
            session_data = {"session_id": session_id, "entries": []}

        session_data["entries"].append(entry)
        session_data["last_updated"] = datetime.now().isoformat()

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)

    # =========================================
    # GERI BILDIRIM AL (FLUTTER API ICIN)
    # =========================================

    def should_request_feedback(self, confidence: float) -> bool:
        """
        Sistem tahmininin guvenine gore geri bildirim istenip
        istenmeyecegini belirler.

        Dusuk guvenli tahminlerde kullaniciya sormak daha faydali.

        Args:
            confidence: Sistemin guven skoru (0-1)

        Returns:
            True ise geri bildirim iste
        """
        return confidence < FEEDBACK_MIN_CONFIDENCE_FOR_PROMPT

    def get_feedback_prompt(self, prediction_class: int, confidence: float) -> Dict:
        """
        Flutter UI icin geri bildirim istegi olusturur.

        Returns:
            Kullaniciya gosterilecek soru ve secenekler (JSON)
        """
        prompt = {
            "title_tr": "Geri Bildiriminiz",
            "title_en": "Your Feedback",
            "question_tr": (
                f"Sistem bu karpuzun '{CLASS_LABELS.get(prediction_class, '')}' "
                f"oldugunu tahmin etti (guven: %{confidence*100:.0f}). "
                f"Sizce dogru mu?"
            ),
            "question_en": (
                f"The system predicted this watermelon as "
                f"'{CLASS_LABELS.get(prediction_class, '')}' "
                f"(confidence: {confidence*100:.0f}%). "
                f"Do you agree?"
            ),
            "options": [
                {
                    "class_id": k,
                    "label_tr": v["tr"],
                    "label_en": v["en"]
                }
                for k, v in FEEDBACK_CLASS_OPTIONS.items()
            ],
            "allow_notes": True,
            "predicted_class": prediction_class,
            "confidence": confidence,
        }
        return prompt

    # =========================================
    # ISTATISTIK VE RAPOR
    # =========================================

    def get_statistics(self) -> Dict:
        """
        Toplanan geri bildirim istatistiklerini dondurur.

        Returns:
            Istatistik ozeti
        """
        if not self._entries:
            return {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "class_distribution": {},
                "confusion_counts": {},
            }

        total = len(self._entries)
        correct = sum(1 for e in self._entries if e["user_feedback"]["is_correct"])
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0

        # Sinif dagilimi (kullanici etiketleri)
        class_dist = {}
        for e in self._entries:
            c = e["user_feedback"]["class_id"]
            class_dist[c] = class_dist.get(c, 0) + 1

        # Confusion: tahmin vs gercek
        confusion = {}
        for e in self._entries:
            pred = e["prediction"]["class_id"]
            true = e["user_feedback"]["class_id"]
            key = f"{pred}->{true}"
            confusion[key] = confusion.get(key, 0) + 1

        # Oturum sayisi
        sessions = set(e.get("session_id", "anonymous") for e in self._entries)

        # Zaman analizi
        timestamps = [e["timestamp"] for e in self._entries]
        first = min(timestamps) if timestamps else None
        last = max(timestamps) if timestamps else None

        # Guven dagilimi
        confidences = [e["prediction"]["confidence"] for e in self._entries]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # HH spesifik istatistikler
        hh_entries = [e for e in self._entries if e["user_feedback"]["class_id"] == 2]
        hh_detected_correctly = sum(
            1 for e in hh_entries if e["prediction"]["class_id"] == 2
        )

        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "class_distribution": {
                CLASS_LABELS.get(k, f"class_{k}"): v
                for k, v in sorted(class_dist.items())
            },
            "confusion_counts": confusion,
            "n_sessions": len(sessions),
            "first_entry": first,
            "last_entry": last,
            "avg_confidence": avg_confidence,
            "hh_total": len(hh_entries),
            "hh_detected_correctly": hh_detected_correctly,
            "hh_recall": (hh_detected_correctly / len(hh_entries)
                         if hh_entries else 0.0),
            "retrain_threshold": FEEDBACK_RETRAIN_THRESHOLD,
            "retrain_ready": total >= FEEDBACK_RETRAIN_THRESHOLD,
        }

    def print_summary(self):
        """Geri bildirim ozet raporunu yazdirir."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("  CROWDSOURCING GERI BILDIRIM RAPORU")
        print("=" * 60)

        if stats["total"] == 0:
            print("  Henuz geri bildirim toplanmadi.")
            print("=" * 60)
            return

        print(f"\n  Toplam geri bildirim: {stats['total']}")
        print(f"  Dogru tahmin:        {stats['correct']} ({stats['accuracy']:.1%})")
        print(f"  Yanlis tahmin:       {stats['incorrect']}")
        print(f"  Ort. guven skoru:    {stats['avg_confidence']:.2f}")
        print(f"  Oturum sayisi:       {stats['n_sessions']}")
        print(f"  Ilk kayit:           {stats['first_entry']}")
        print(f"  Son kayit:           {stats['last_entry']}")

        print(f"\n  Sinif Dagilimi (Kullanici Etiketleri):")
        for label, count in stats["class_distribution"].items():
            pct = count / stats["total"] * 100
            bar = "#" * int(pct / 2)
            print(f"    {label}: {count} ({pct:.1f}%) {bar}")

        if stats["hh_total"] > 0:
            print(f"\n  Hollow Heart Tespiti:")
            print(f"    Gercek HH ornekleri:  {stats['hh_total']}")
            print(f"    Dogru tespit edilen:  {stats['hh_detected_correctly']}")
            print(f"    HH Recall:            {stats['hh_recall']:.1%}")

        print(f"\n  Confusion Akisi (Tahmin -> Gercek):")
        for flow, count in sorted(stats["confusion_counts"].items()):
            pred, true = flow.split("->")
            pred_label = CLASS_LABELS.get(int(pred), pred)[:15]
            true_label = CLASS_LABELS.get(int(true), true)[:15]
            print(f"    {pred_label} -> {true_label}: {count}")

        if stats["retrain_ready"]:
            print(f"\n  [!] Yeniden egitim icin yeterli veri mevcut!")
            print(f"      python main.py --retrain-with-feedback")
        else:
            remaining = FEEDBACK_RETRAIN_THRESHOLD - stats["total"]
            print(f"\n  Yeniden egitim icin {remaining} geri bildirim daha gerekli")
            print(f"  (Esik: {FEEDBACK_RETRAIN_THRESHOLD})")

        print("=" * 60)

    # =========================================
    # EGITIM VERISI HAZIRLAMA
    # =========================================

    def prepare_training_data(
        self,
        include_hh: bool = True,
        min_entries: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Geri bildirim verilerinden egitim verisi hazirlar.

        Yalnizca ozellik vektoru mevcut olan kayitlari kullanir.

        Args:
            include_hh: HH ozelliklerini dahil et
            min_entries: Minimum kayit sayisi

        Returns:
            (X_acoustic, X_hh, y) tuple'i veya (None, None, None)
        """
        # Ozellik vektoru olan kayitlari filtrele
        valid_entries = [
            e for e in self._entries
            if e.get("features", {}).get("acoustic_vector") is not None
        ]

        if len(valid_entries) < min_entries:
            print(f"  [Crowdsourcing] Yetersiz veri: {len(valid_entries)}/{min_entries}")
            return None, None, None

        X_acoustic_list = []
        X_hh_list = []
        y_list = []

        for entry in valid_entries:
            acoustic = np.array(entry["features"]["acoustic_vector"])

            # Boyut kontrolu
            if len(acoustic) != N_FEATURES:
                continue

            X_acoustic_list.append(acoustic)
            y_list.append(entry["user_feedback"]["class_id"])

            if include_hh:
                hh = entry["features"].get("hh_vector")
                if hh is not None and len(hh) == TFLITE_HH_INPUT_DIM:
                    X_hh_list.append(np.array(hh))
                else:
                    X_hh_list.append(np.zeros(TFLITE_HH_INPUT_DIM))

        if not X_acoustic_list:
            return None, None, None

        X_acoustic = np.array(X_acoustic_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        if include_hh and X_hh_list:
            X_hh = np.array(X_hh_list, dtype=np.float32)
        else:
            X_hh = None

        print(f"  [Crowdsourcing] Egitim verisi hazir:")
        print(f"    Ornekler: {len(y)}")
        print(f"    Akustik boyut: {X_acoustic.shape}")
        if X_hh is not None:
            print(f"    HH boyut: {X_hh.shape}")
        print(f"    Sinif dagilimi: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X_acoustic, X_hh, y

    # =========================================
    # CSV EXPORT
    # =========================================

    def export_to_csv(self, output_path: str = None) -> str:
        """
        Geri bildirim verilerini CSV formatinda export eder.

        Args:
            output_path: Cikis dosya yolu

        Returns:
            CSV dosya yolu
        """
        if output_path is None:
            output_path = FEEDBACK_EXPORT_PATH

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = [
            "id", "timestamp", "session_id",
            "pred_class", "pred_label", "pred_confidence",
            "user_class", "user_label", "is_correct",
            "hh_score", "notes",
            "has_features", "has_hh_features"
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self._entries:
                row = {
                    "id": entry["id"],
                    "timestamp": entry["timestamp"],
                    "session_id": entry.get("session_id", ""),
                    "pred_class": entry["prediction"]["class_id"],
                    "pred_label": entry["prediction"]["class_label"],
                    "pred_confidence": entry["prediction"]["confidence"],
                    "user_class": entry["user_feedback"]["class_id"],
                    "user_label": entry["user_feedback"]["class_label"],
                    "is_correct": entry["user_feedback"]["is_correct"],
                    "hh_score": entry["prediction"].get("hh_score", ""),
                    "notes": entry["user_feedback"].get("notes", ""),
                    "has_features": entry["features"].get("acoustic_vector") is not None,
                    "has_hh_features": entry["features"].get("hh_vector") is not None,
                }
                writer.writerow(row)

        print(f"  [Crowdsourcing] CSV export: {output_path} ({len(self._entries)} kayit)")
        return output_path

    # =========================================
    # FLUTTER JSON API
    # =========================================

    def get_recent_entries(self, n: int = 20) -> List[Dict]:
        """Son N geri bildirimi dondurur (Flutter API icin)."""
        entries = self._entries[-n:] if n < len(self._entries) else self._entries
        # Ozellik vektorlerini cikart (boyut azaltma)
        clean_entries = []
        for e in entries:
            clean = {k: v for k, v in e.items() if k != "features"}
            clean_entries.append(clean)
        return list(reversed(clean_entries))

    def get_session_entries(self, session_id: str) -> List[Dict]:
        """Belirli bir oturumun geri bildirimlerini dondurur."""
        return [
            e for e in self._entries
            if e.get("session_id") == session_id
        ]

    def get_model_performance_trend(self, window: int = 20) -> List[Dict]:
        """
        Model performansinin zaman icerisindeki trendini hesaplar.

        Returns:
            Her pencere icin dogruluk oranlari
        """
        if len(self._entries) < window:
            return []

        trend = []
        for i in range(0, len(self._entries) - window + 1, max(1, window // 4)):
            chunk = self._entries[i:i + window]
            correct = sum(1 for e in chunk if e["user_feedback"]["is_correct"])
            accuracy = correct / len(chunk)
            avg_conf = np.mean([e["prediction"]["confidence"] for e in chunk])

            trend.append({
                "window_start": chunk[0]["timestamp"],
                "window_end": chunk[-1]["timestamp"],
                "accuracy": float(accuracy),
                "avg_confidence": float(avg_conf),
                "n_samples": len(chunk),
            })

        return trend

    # =========================================
    # YENIDEN EGITIM ENTEGRASYONU
    # =========================================

    def merge_with_existing_data(
        self,
        existing_X: np.ndarray,
        existing_y: np.ndarray,
        weight_feedback: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mevcut egitim verisiyle geri bildirim verisini birlestirir.

        Geri bildirim verisine daha yuksek agirlik verilir cunku
        kullanici dogrulamali olgunluk bilgisi icerir.

        Args:
            existing_X: Mevcut akustik ozellik matrisi
            existing_y: Mevcut etiketler
            weight_feedback: Geri bildirim verisi agirligi

        Returns:
            (X_merged, y_merged, sample_weights)
        """
        feedback_X, _, feedback_y = self.prepare_training_data(include_hh=False)

        if feedback_X is None or len(feedback_X) == 0:
            weights = np.ones(len(existing_y))
            return existing_X, existing_y, weights

        # Boyut uyumu kontrolu
        if existing_X.shape[1] != feedback_X.shape[1]:
            print(f"  [!] Boyut uyumsuzlugu: mevcut={existing_X.shape[1]}, "
                  f"feedback={feedback_X.shape[1]}")
            weights = np.ones(len(existing_y))
            return existing_X, existing_y, weights

        # Birlestir
        X_merged = np.vstack([existing_X, feedback_X])
        y_merged = np.concatenate([existing_y, feedback_y])

        # Agirliklar: mevcut veri = 1.0, geri bildirim verisi = weight_feedback
        weights = np.concatenate([
            np.ones(len(existing_y)),
            np.full(len(feedback_y), weight_feedback)
        ])

        print(f"  [Crowdsourcing] Veri birlestirildi:")
        print(f"    Mevcut: {len(existing_y)} ornek (agirlik: 1.0)")
        print(f"    Geri bildirim: {len(feedback_y)} ornek (agirlik: {weight_feedback})")
        print(f"    Toplam: {len(y_merged)} ornek")

        return X_merged, y_merged, weights

    # =========================================
    # VERI TEMIZLEME
    # =========================================

    def clear_all(self):
        """Tum geri bildirim verilerini siler."""
        self._entries = []
        self._save()
        print("  [Crowdsourcing] Tum geri bildirim verileri silindi.")

    def remove_entry(self, entry_id: str) -> bool:
        """Belirli bir kaydi siler."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["id"] != entry_id]
        if len(self._entries) < before:
            self._save()
            return True
        return False

    @property
    def total_entries(self) -> int:
        """Toplam geri bildirim sayisi."""
        return len(self._entries)


# ============================================
# DEMO: SENTETIK GERI BILDIRIM URETICI
# ============================================

def generate_demo_feedback(n_samples: int = 30) -> FeedbackLogger:
    """
    Demo amacli sentetik geri bildirim uretir.

    Returns:
        Doldurulmus FeedbackLogger nesnesi
    """
    np.random.seed(42)
    logger = FeedbackLogger()

    print(f"\n[Demo] {n_samples} sentetik geri bildirim uretiliyor...")

    for i in range(n_samples):
        # Rastgele gercek sinif
        true_class = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])

        # Sistem tahmini (%80 dogru, %20 yanlis)
        if np.random.rand() < 0.80:
            pred_class = true_class
        else:
            other_classes = [c for c in [0, 1, 2] if c != true_class]
            pred_class = np.random.choice(other_classes)

        confidence = np.random.uniform(0.55, 0.98)

        # Sentetik ozellik vektoru
        features = np.random.randn(N_FEATURES).astype(np.float32)
        hh_features = np.random.rand(TFLITE_HH_INPUT_DIM).astype(np.float32) * 0.5
        if true_class == 2:
            hh_features[0] = np.random.uniform(0.5, 0.9)

        hh_score = float(hh_features[0])

        logger.log_feedback(
            prediction_class=pred_class,
            prediction_confidence=confidence,
            user_label=true_class,
            features=features,
            hh_features=hh_features,
            hh_score=hh_score,
            session_id=f"demo_user_{i % 5}",
            notes=f"Demo ornek #{i+1}" if np.random.rand() < 0.3 else None,
        )

    return logger


if __name__ == "__main__":
    print("=" * 60)
    print("  CROWDSOURCING DEMO")
    print("=" * 60)

    logger = generate_demo_feedback(n_samples=30)
    logger.print_summary()

    # CSV export
    logger.export_to_csv()

    # Egitim verisi hazirlama
    X, X_hh, y = logger.prepare_training_data()

    # Performans trendi
    trend = logger.get_model_performance_trend(window=10)
    if trend:
        print(f"\n  Performans trendi ({len(trend)} pencere):")
        for t in trend[-3:]:
            print(f"    {t['window_start'][:10]}: dogruluk={t['accuracy']:.2%}")

    print("\n  Demo tamamlandi!")

