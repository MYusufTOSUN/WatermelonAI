"""
MODULE_E: Sınıflandırma Modülleri

KNN (K-Nearest Neighbors) ve RFC (Random Forest Classifier)
modelleri ile karpuz olgunluk sınıflandırması.

Sınıflar:
  0: Olgunlaşmamış (Immature)
  1: Olgun (Mature/Ripe)
  2: İçi Geçmiş (Over-mature/Hollow Heart)
"""

import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    KNN_N_NEIGHBORS,
    KNN_WEIGHTS,
    KNN_METRIC,
    RFC_N_ESTIMATORS,
    RFC_MAX_DEPTH,
    RFC_MIN_SAMPLES_LEAF,
    RFC_MIN_SAMPLES_SPLIT,
    RFC_RANDOM_STATE,
    CLASS_LABELS,
    MODELS_DIR
)


class RipenessClassifier:
    """
    Karpuz olgunluk sınıflandırıcısı.
    
    KNN ve Random Forest modellerini destekler.
    Qilin Watermelon Dataset özellikleri ile eğitilir.
    """

    def __init__(self, model_type: str = "knn"):
        """
        Args:
            model_type: Model tipi ('knn' veya 'rfc')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None

        self._init_model()

    def _init_model(self):
        """Model nesnesini başlatır."""
        if self.model_type == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=KNN_N_NEIGHBORS,
                weights=KNN_WEIGHTS,
                metric=KNN_METRIC,
                n_jobs=-1
            )
        elif self.model_type == "rfc":
            self.model = RandomForestClassifier(
                n_estimators=RFC_N_ESTIMATORS,
                max_depth=RFC_MAX_DEPTH,
                min_samples_leaf=RFC_MIN_SAMPLES_LEAF,
                min_samples_split=RFC_MIN_SAMPLES_SPLIT,
                random_state=RFC_RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt',
            )
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list = None
    ) -> Dict[str, object]:
        """
        Modeli eğitir.
        
        Args:
            X_train: Eğitim özellik matrisi (n_samples, n_features)
            y_train: Eğitim etiketleri (n_samples,)
            feature_names: Özellik isimleri
            
        Returns:
            Eğitim metrikleri
        """
        self.feature_names = feature_names

        # Özellik standardizasyonu
        X_scaled = self.scaler.fit_transform(X_train)

        # Model eğitimi
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # Çapraz doğrulama
        cv_scores = cross_val_score(self.model, X_scaled, y_train, cv=5, scoring='accuracy')

        # Eğitim seti performansı
        train_predictions = self.model.predict(X_scaled)
        all_labels = sorted(CLASS_LABELS.keys())
        train_report = classification_report(
            y_train, train_predictions,
            labels=all_labels,
            target_names=[CLASS_LABELS[i] for i in all_labels],
            output_dict=True,
            zero_division=0,
        )

        return {
            "model_type": self.model_type,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "train_accuracy": float(np.mean(train_predictions == y_train)),
            "classification_report": train_report
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tahmin yapar.
        
        Args:
            X: Özellik matrisi (n_samples, n_features)
            
        Returns:
            (predictions, probabilities): Tahminler ve olasılıklar
        """
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmedi!")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Olasılık tahmini
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = np.zeros((len(X), len(CLASS_LABELS)))

        return predictions, probabilities

    def predict_single(self, features: np.ndarray) -> Dict[str, object]:
        """
        Tek bir örnek için tahmin yapar.
        
        Args:
            features: 1D özellik vektörü
            
        Returns:
            Tahmin sonuçları
        """
        X = features.reshape(1, -1)
        predictions, probabilities = self.predict(X)

        class_id = int(predictions[0])
        probs = probabilities[0]

        return {
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "confidence": float(np.max(probs)),
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i])
                for i in range(len(probs))
            }
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, object]:
        """
        Model değerlendirmesi yapar.
        
        Args:
            X_test: Test özellik matrisi
            y_test: Test etiketleri
            
        Returns:
            Değerlendirme metrikleri
        """
        predictions, _ = self.predict(X_test)

        all_labels = sorted(CLASS_LABELS.keys())
        report = classification_report(
            y_test, predictions,
            labels=all_labels,
            target_names=[CLASS_LABELS[i] for i in all_labels],
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, predictions, labels=all_labels)

        return {
            "accuracy": float(np.mean(predictions == y_test)),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Grid search ile hiperparametre optimizasyonu.
        
        Returns:
            En iyi parametreler ve skor
        """
        X_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == "knn":
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        else:  # rfc
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }

        grid_search = GridSearchCV(
            self.model, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y_train)

        self.model = grid_search.best_estimator_
        self.is_trained = True

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_)
        }

    def save_model(self, path: str = None):
        """Modeli diske kaydeder."""
        if path is None:
            path = os.path.join(str(MODELS_DIR), f"{self.model_type}_ripeness_model.joblib")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        joblib.dump(save_data, path)
        print(f"[MODULE_E] Model kaydedildi: {path}")

    def load_model(self, path: str = None):
        """Modeli diskten yükler."""
        if path is None:
            path = os.path.join(str(MODELS_DIR), f"{self.model_type}_ripeness_model.joblib")

        save_data = joblib.load(path)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.model_type = save_data['model_type']
        self.feature_names = save_data['feature_names']
        self.is_trained = True
        print(f"[MODULE_E] Model yüklendi: {path}")

