"""
Veri Yukleyici

Qilin Watermelon Dataset'i yukler ve isler.

Qilin Dataset Yapisi (watermelon_eval):
  datasets/
    {id}_{brix_degeri}/       # orn: "001_8.5" -> Brix=8.5
      chu/
        {alt_klasor}/
          *.wav               # 2 kanalli ses (sag kanal kullanilir, 16000 ornek)
          *.jpg               # gorsel dosya

Brix -> Kategori Donusumu (Qilin 19-datasets'e göre kalibre):
  Brix < 10.0   -> 0 (Olgunlasmamis / Immature)
  10.0 <= Brix <= 11.5 -> 1 (Olgun / Mature / Ripe)
  Brix > 11.5   -> 2 (Ici Gecmis / Over-mature)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_LABELS,
    BRIX_IMMATURE_THRESHOLD, BRIX_OVERRIPE_THRESHOLD,
)
from backend.module_d.feature_extractor import AcousticFeatureExtractor
from backend.utils.audio_utils import load_wav, load_wav_right_channel, normalize_audio, trim_silence

# Lazy import: VisualAnalyzer + HollowHeartDetector icin dis bagimliliklar (cv2/librosa)
# yuklenme sirasi gerekebilir — fonksiyon icinde import ediyoruz.


class QilinDatasetLoader:
    """
    Qilin Watermelon Dataset yukleyicisi.
    
    Gercek Qilin yapisini (watermelon_eval) ve alternatif yapilari destekler.
    
    Desteklenen yapilar:
      Yapi 1 (Qilin Orijinal - watermelon_eval):
        datasets/
          {id}_{brix}/chu/{alt_klasor}/{*.wav, *.jpg}
          
      Yapi 2 (Hazir labels.csv):
        qilin/
          audio/           # WAV dosyalari
          labels.csv       # filename, brix, category
          
      Yapi 3 (Klasor bazli etiketleme):
        qilin/
          immature/
          mature/
          overmature/
          
      Yapi 4 (Excel metadata):
        qilin/
          audio/           # WAV dosyalari
          metadata.xlsx
    """

    def __init__(
        self,
        data_dir: str = None,
        sample_rate: int = 44100
    ):
        self.data_dir = data_dir or str(RAW_DATA_DIR / "qilin")
        self.sample_rate = sample_rate
        self.feature_extractor = AcousticFeatureExtractor(sample_rate=sample_rate)
        
        # Qilin dataset dizini (dataset/datasets)
        project_root = Path(__file__).parent.parent.parent
        self.qilin_datasets_dir = str(project_root / "dataset" / "datasets")

    def auto_discover_structure(self) -> dict:
        """
        Veri seti dizinini tarar ve yapiyi otomatik kesfeder.
        
        Oncelikle Qilin orijinal yapisini (datasets/{id}_{brix}/chu/...)
        kontrol eder, bulamazsa alternatif yapilari arar.
        
        Returns:
            Kesif sonuclari: structure_type, samples listesi, istatistikler
        """
        result = {
            "wav_dir": None,
            "label_file": None,
            "structure_type": None,
            "wav_count": 0,
            "image_dir": None,
            "image_count": 0,
            "qilin_samples": [],  # Qilin orijinal yapisi icin
            "brix_range": None
        }

        # ==============================================
        # ONCELIK 1: Qilin Orijinal Yapi Kontrolu
        # datasets/{id}_{brix}/chu/{folder}/{wav,jpg}
        # ==============================================
        if os.path.exists(self.qilin_datasets_dir):
            qilin_result = self._discover_qilin_original(self.qilin_datasets_dir)
            if qilin_result["qilin_samples"]:
                result.update(qilin_result)
                result["structure_type"] = "qilin_original"
                self._print_discovery(result)
                return result

        # ==============================================
        # ONCELIK 2: data/raw/qilin icinde Qilin yapisi
        # ==============================================
        alt_qilin_dir = os.path.join(self.data_dir, "datasets")
        if os.path.exists(alt_qilin_dir):
            qilin_result = self._discover_qilin_original(alt_qilin_dir)
            if qilin_result["qilin_samples"]:
                result.update(qilin_result)
                result["structure_type"] = "qilin_original"
                self._print_discovery(result)
                return result

        # data_dir icinde direkt Qilin yapisi kontrol
        if os.path.exists(self.data_dir):
            qilin_result = self._discover_qilin_original(self.data_dir)
            if qilin_result["qilin_samples"]:
                result.update(qilin_result)
                result["structure_type"] = "qilin_original"
                self._print_discovery(result)
                return result

        # ==============================================
        # ONCELIK 3: Genel WAV/Etiket Arama
        # ==============================================
        search_dirs = [self.data_dir, self.qilin_datasets_dir]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            # WAV dosyalarini ara
            all_wavs = list(Path(search_dir).rglob("*.wav"))
            all_wavs += list(Path(search_dir).rglob("*.WAV"))
            result["wav_count"] = len(all_wavs)

            if all_wavs:
                wav_dirs = {}
                for w in all_wavs:
                    d = str(w.parent)
                    wav_dirs[d] = wav_dirs.get(d, 0) + 1
                result["wav_dir"] = max(wav_dirs, key=wav_dirs.get)

            # Etiket dosyalarini ara
            for name in ["labels.csv", "label.csv", "metadata.csv", "data.csv",
                          "annotations.csv", "brix.csv", "dataset.csv"]:
                fpath = os.path.join(search_dir, name)
                if os.path.exists(fpath):
                    result["label_file"] = fpath
                    result["structure_type"] = "csv"
                    break

            # Excel dosyasi ara
            if result["label_file"] is None:
                for ext in ["*.xlsx", "*.xls"]:
                    excels = list(Path(search_dir).rglob(ext))
                    if excels:
                        result["label_file"] = str(excels[0])
                        result["structure_type"] = "excel"
                        break

            # Klasor bazli etiketleme
            if result["label_file"] is None:
                category_dirs = {}
                category_keywords = {
                    0: ["immature", "unripe", "green", "raw", "olgunlasmamis"],
                    1: ["mature", "ripe", "ready", "olgun"],
                    2: ["overmature", "overripe", "hollow", "ici_gecmis"]
                }
                for cat_id, keywords in category_keywords.items():
                    for kw in keywords:
                        check_dir = os.path.join(search_dir, kw)
                        if os.path.isdir(check_dir):
                            wavs_in = list(Path(check_dir).rglob("*.wav"))
                            if wavs_in:
                                category_dirs[cat_id] = check_dir
                                break
                if len(category_dirs) >= 2:
                    result["label_file"] = category_dirs
                    result["structure_type"] = "folder_based"

            # Goruntu dosyalarini ara
            all_images = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                all_images += list(Path(search_dir).rglob(ext))
            result["image_count"] = len(all_images)
            if all_images:
                img_dirs = {}
                for img in all_images:
                    d = str(img.parent)
                    img_dirs[d] = img_dirs.get(d, 0) + 1
                result["image_dir"] = max(img_dirs, key=img_dirs.get)

            if result["wav_count"] > 0 or result["structure_type"]:
                break

        self._print_discovery(result)
        return result

    def _discover_qilin_original(self, datasets_dir: str) -> dict:
        """
        Qilin orijinal yapisini kesfeder.

        Yapi: datasets/{id}_{brix}/chu/{folder}/{*.wav, *.jpg}
        veya:  datasets/watermelon_dataset/19_datasets/{id}_{brix}/chu/...

        Klasor adi {id}_{brix} formatinda:
          - id: karpuz numarasi
          - brix: seker degeri (float)
        """
        result = {
            "qilin_samples": [],
            "wav_count": 0,
            "image_count": 0,
            "brix_range": None
        }

        if not os.path.exists(datasets_dir):
            return result

        # {id}_{brix} formatindaki klasorleri bul (ust ve alt dizinlerde)
        subdirs = []
        candidate_dirs = [datasets_dir]

        # Alt dizinlerde de ara (watermelon_dataset/19_datasets gibi)
        try:
            for d in os.listdir(datasets_dir):
                d_path = os.path.join(datasets_dir, d)
                if os.path.isdir(d_path):
                    candidate_dirs.append(d_path)
                    # Bir seviye daha derine bak
                    for dd in os.listdir(d_path):
                        dd_path = os.path.join(d_path, dd)
                        if os.path.isdir(dd_path):
                            candidate_dirs.append(dd_path)
        except Exception:
            pass

        # Her aday dizinde {id}_{brix} formatini ara
        found_parent = None
        for cdir in candidate_dirs:
            try:
                entries = [d for d in os.listdir(cdir)
                           if os.path.isdir(os.path.join(cdir, d))]
            except Exception:
                continue

            # Bu dizindeki {id}_{brix} formatindaki klasorleri say
            brix_count = 0
            for entry in entries:
                try:
                    parts = entry.rsplit("_", 1)
                    if len(parts) == 2:
                        float(parts[1])
                        brix_count += 1
                except (ValueError, IndexError):
                    continue

            if brix_count > 0 and (found_parent is None or brix_count > len(subdirs)):
                found_parent = cdir
                subdirs = entries

        if not found_parent:
            return result

        brix_values = []

        for subdir in subdirs:
            subdir_path = os.path.join(found_parent, subdir)

            # {id}_{brix} formatini parse et
            try:
                parts = subdir.rsplit("_", 1)
                if len(parts) == 2:
                    data_id = parts[0]
                    brix = float(parts[1])
                else:
                    # Alternatif: tum sayisal kismi dene
                    continue
            except (ValueError, IndexError):
                # Bu format degilse atla
                continue

            brix_values.append(brix)
            category = self._brix_to_category(brix)

            # chu/ klasorunu kontrol et
            chu_dir = os.path.join(subdir_path, "chu")
            if not os.path.isdir(chu_dir):
                # chu olmadan direkt icerik de olabilir
                chu_dir = subdir_path

            # Alt klasorleri tara
            try:
                folders = [f for f in os.listdir(chu_dir)
                           if os.path.isdir(os.path.join(chu_dir, f))]
            except Exception:
                folders = []

            if not folders:
                # Alt klasor yoksa direkt dosyalari ara
                wav_files = [f for f in os.listdir(chu_dir) if f.lower().endswith('.wav')]
                jpg_files = [f for f in os.listdir(chu_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for wf in wav_files:
                    # En yakin jpg'yi esle
                    paired_jpg = None
                    if jpg_files:
                        paired_jpg = os.path.join(chu_dir, jpg_files[0])
                    
                    result["qilin_samples"].append({
                        "data_id": data_id,
                        "brix": brix,
                        "category": category,
                        "wav_path": os.path.join(chu_dir, wf),
                        "jpg_path": paired_jpg,
                        "folder": subdir
                    })
                    result["wav_count"] += 1
                    if paired_jpg:
                        result["image_count"] += 1
                continue

            for folder in folders:
                folder_path = os.path.join(chu_dir, folder)
                
                try:
                    files = os.listdir(folder_path)
                except Exception:
                    continue

                wav_files = [f for f in files if f.lower().endswith('.wav')]
                jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                for wf in wav_files:
                    wav_full = os.path.join(folder_path, wf)
                    
                    # Eslesen gorsel dosyayi bul
                    jpg_full = None
                    if jpg_files:
                        jpg_full = os.path.join(folder_path, jpg_files[0])

                    result["qilin_samples"].append({
                        "data_id": data_id,
                        "brix": brix,
                        "category": category,
                        "wav_path": wav_full,
                        "jpg_path": jpg_full,
                        "folder": f"{subdir}/chu/{folder}"
                    })
                    result["wav_count"] += 1
                    if jpg_full:
                        result["image_count"] += 1

        if brix_values:
            result["brix_range"] = (min(brix_values), max(brix_values))

        return result

    def _print_discovery(self, result: dict):
        """Kesif sonuclarini yazdirir."""
        print(f"\n{'='*60}")
        print(f"  [AutoDiscover] Veri Seti Analizi")
        print(f"{'='*60}")
        print(f"  Yapi tipi: {result.get('structure_type', 'bulunamadi')}")
        print(f"  WAV dosyasi: {result.get('wav_count', 0)} adet")
        print(f"  Goruntu dosyasi: {result.get('image_count', 0)} adet")
        
        if result.get("structure_type") == "qilin_original":
            samples = result.get("qilin_samples", [])
            brix_range = result.get("brix_range")
            categories = [s["category"] for s in samples]
            cat_counts = {0: categories.count(0), 1: categories.count(1), 2: categories.count(2)}
            
            print(f"  Qilin ornekleri: {len(samples)} adet")
            if brix_range:
                print(f"  Brix araligi: {brix_range[0]:.1f} - {brix_range[1]:.1f}")
            print(f"  Sinif dagilimi:")
            print(f"    0 (Olgunlasmamis): {cat_counts[0]}")
            print(f"    1 (Olgun):         {cat_counts[1]}")
            print(f"    2 (Ici Gecmis):    {cat_counts[2]}")
        else:
            if result.get("wav_dir"):
                print(f"  WAV dizini: {result['wav_dir']}")
            if result.get("image_dir"):
                print(f"  Goruntu dizini: {result['image_dir']}")
            if result.get("label_file"):
                lf = result["label_file"]
                if isinstance(lf, dict):
                    print(f"  Etiket: Klasor bazli ({len(lf)} sinif)")
                else:
                    print(f"  Etiket dosyasi: {lf}")
        print(f"{'='*60}\n")

    def load_labels(self, labels_file: str = None) -> pd.DataFrame:
        """
        Etiket dosyasini yukler. Oncelikle Qilin orijinal yapisini dener.
        
        Returns:
            DataFrame: filename, brix, category, wav_path, jpg_path sutunlari
        """
        discovery = self.auto_discover_structure()

        # Qilin orijinal yapisi
        if discovery.get("structure_type") == "qilin_original":
            return self._qilin_samples_to_dataframe(discovery["qilin_samples"])

        # Digerleri
        if labels_file is None and discovery["label_file"] is not None:
            labels_file = discovery["label_file"]
        elif labels_file is None:
            labels_file = os.path.join(self.data_dir, "labels.csv")

        # Klasor bazli etiketleme
        if isinstance(labels_file, dict):
            return self._load_folder_based_labels(labels_file)

        if not os.path.exists(labels_file):
            wav_dir = discovery.get("wav_dir")
            if wav_dir:
                return self._create_labels_from_wav_files(wav_dir)
            print(f"[DataLoader] UYARI: Etiket dosyasi bulunamadi: {labels_file}")
            print("[DataLoader] Sentetik veri ile devam ediliyor...")
            return self._generate_synthetic_labels()

        # CSV veya Excel yukle
        ext = Path(labels_file).suffix.lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(labels_file)
            print(f"[DataLoader] Excel yuklendi: {labels_file}")
        else:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(labels_file, sep=sep)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                df = pd.read_csv(labels_file)
            print(f"[DataLoader] CSV yuklendi: {labels_file}")

        df = self._normalize_columns(df)
        print(f"[DataLoader] {len(df)} satir, sutunlar: {list(df.columns)}")
        return df

    def _qilin_samples_to_dataframe(self, samples: list) -> pd.DataFrame:
        """Qilin orijinal orneklerini DataFrame'e donusturur."""
        rows = []
        for s in samples:
            rows.append({
                "filename": os.path.basename(s["wav_path"]),
                "data_id": s["data_id"],
                "brix": s["brix"],
                "category": s["category"],
                "wav_path": s["wav_path"],
                "jpg_path": s["jpg_path"],
                "folder": s["folder"]
            })
        
        df = pd.DataFrame(rows)
        print(f"[DataLoader] Qilin orijinal: {len(df)} ornek yuklendi")
        print(f"[DataLoader] Brix araligi: {df['brix'].min():.1f} - {df['brix'].max():.1f}")
        print(f"[DataLoader] Sinif dagilimi: {dict(df['category'].value_counts().sort_index())}")
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Farkli CSV formatlarindaki sutun adlarini standart forma donusturur."""
        col_lower = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_lower)
        cols = list(df.columns)

        # filename sutununu bul
        filename_candidates = ["filename", "file", "name", "wav", "audio",
                               "file_name", "dosya", "sample", "id"]
        for fc in filename_candidates:
            matches = [c for c in cols if fc in c.lower()]
            if matches:
                df = df.rename(columns={matches[0]: "filename"})
                break

        # brix sutununu bul
        brix_candidates = ["brix", "sugar", "sweetness", "seker", "bx",
                           "sugar_content", "tss"]
        for bc in brix_candidates:
            matches = [c for c in cols if bc in c.lower()]
            if matches:
                df = df.rename(columns={matches[0]: "brix"})
                break

        # category sutununu bul veya brix'ten olustur
        cat_candidates = ["category", "class", "label", "sinif", "etiket",
                          "ripeness", "maturity", "olgunluk"]
        cat_found = False
        for cc in cat_candidates:
            matches = [c for c in df.columns if cc in c.lower()]
            if matches:
                df = df.rename(columns={matches[0]: "category"})
                cat_found = True
                break

        if not cat_found and "brix" in df.columns:
            df["brix"] = pd.to_numeric(df["brix"], errors='coerce')
            df["category"] = df["brix"].apply(self._brix_to_category)
            print(f"[DataLoader] Brix'ten otomatik kategori olusturuldu")

        if "category" in df.columns and df["category"].dtype == object:
            cat_map = {
                "immature": 0, "unripe": 0, "green": 0, "olgunlasmamis": 0, "ham": 0,
                "mature": 1, "ripe": 1, "olgun": 1, "ready": 1,
                "overmature": 2, "overripe": 2, "hollow": 2, "ici_gecmis": 2
            }
            df["category"] = df["category"].str.lower().str.strip().map(cat_map)
            df["category"] = df["category"].fillna(1).astype(int)

        if "filename" in df.columns:
            df["filename"] = df["filename"].apply(
                lambda x: str(x) if str(x).endswith('.wav') else f"{x}.wav"
            )

        return df

    @staticmethod
    def _brix_to_category(brix_value: float) -> int:
        """
        Brix degerinden olgunluk kategorisi.

        Qilin 19-datasets Brix araligi: 8.7 - 12.7
        Esikler veri setine ve karpuz literaturune gore ayarlanmistir:
          - Brix < 10.0  → 0 (Olgunlasmamis / az tatli)
          - 10.0 <= Brix <= 11.5 → 1 (Olgun / ideal)
          - Brix > 11.5  → 2 (Asiri olgun / ici gecmis riski)
        """
        if pd.isna(brix_value):
            return 1
        if brix_value < BRIX_IMMATURE_THRESHOLD:
            return 0   # Olgunlasmamis
        elif brix_value <= BRIX_OVERRIPE_THRESHOLD:
            return 1   # Olgun
        else:
            return 2   # Ici gecmis

    def _load_folder_based_labels(self, category_dirs: dict) -> pd.DataFrame:
        """Klasor bazli etiketleme: Her klasor bir sinif."""
        rows = []
        for cat_id, dir_path in category_dirs.items():
            wavs = list(Path(dir_path).rglob("*.wav"))
            wavs += list(Path(dir_path).rglob("*.WAV"))
            for wav in wavs:
                rows.append({
                    "filename": wav.name,
                    "brix": -1,
                    "category": cat_id,
                    "wav_path": str(wav)
                })
        df = pd.DataFrame(rows)
        print(f"[DataLoader] Klasor bazli etiketleme: {len(df)} dosya, "
              f"{len(category_dirs)} sinif")
        return df

    def _create_labels_from_wav_files(self, wav_dir: str) -> pd.DataFrame:
        """WAV dosyalari var ama etiket dosyasi yok."""
        wavs = list(Path(wav_dir).glob("*.wav"))
        wavs += list(Path(wav_dir).glob("*.WAV"))

        print(f"\n{'='*60}")
        print(f"  UYARI: {len(wavs)} WAV dosyasi bulundu ama etiket dosyasi yok!")
        print(f"  Lutfen asagidaki formatta bir labels.csv olusturun:")
        print(f"  ")
        print(f"  filename,brix,category")
        print(f"  ornek1.wav,10.5,1")
        print(f"  ornek2.wav,6.2,0")
        print(f"  ")
        print(f"  Kategori: 0=Olgunlasmamis, 1=Olgun, 2=Ici Gecmis")
        print(f"{'='*60}\n")

        rows = []
        for wav in wavs:
            rows.append({
                "filename": wav.name,
                "brix": -1,
                "category": 1,
                "wav_path": str(wav)
            })

        return pd.DataFrame(rows)

    def _generate_synthetic_labels(self, n_samples: int = 300) -> pd.DataFrame:
        """Test amacli sentetik etiketler olusturur."""
        np.random.seed(42)

        filenames = [f"sample_{i:04d}.wav" for i in range(n_samples)]
        brix_values = np.concatenate([
            np.random.uniform(4, 8, n_samples // 3),
            np.random.uniform(8, 12, n_samples // 3),
            np.random.uniform(12, 16, n_samples - 2 * (n_samples // 3))
        ])
        np.random.shuffle(brix_values)

        categories = np.where(brix_values < 8, 0,
                              np.where(brix_values <= 12, 1, 2))

        return pd.DataFrame({
            "filename": filenames,
            "brix": brix_values,
            "category": categories.astype(int)
        })

    # =================================================================
    # MULTIMODAL (AUDIO + IMAGE + HH) FEATURE EXTRACTION
    # =================================================================

    def _augment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Ses sinyali icin 6 augmentasyon varyasyonu uretir.

        1) Time stretch (±10%)
        2) Pitch shift (±1 semitone)
        3) Gaussian gurultu (SNR ~25 dB)
        4) Volume perturbation (±6 dB)
        5) Time shift (circular, ±%20)
        6) SpecAugment tarzi frekans maskeleme
        """
        import librosa
        augmented = []
        audio_f = audio.astype(np.float32)

        # 1) Time stretch
        try:
            rate = np.random.uniform(0.90, 1.10)
            a1 = librosa.effects.time_stretch(y=audio_f, rate=rate)
            augmented.append(a1)
        except Exception:
            augmented.append(audio_f.copy())

        # 2) Pitch shift
        try:
            n_steps = np.random.uniform(-1.0, 1.0)
            a2 = librosa.effects.pitch_shift(y=audio_f, sr=sr, n_steps=n_steps)
            augmented.append(a2)
        except Exception:
            augmented.append(audio_f.copy())

        # 3) Gauss gurultu
        rms = np.sqrt(np.mean(audio_f ** 2) + 1e-12)
        noise = np.random.randn(len(audio_f)).astype(np.float32) * rms * 0.06
        augmented.append(audio_f + noise)

        # 4) Volume perturbation (±6 dB)
        gain_db = np.random.uniform(-6.0, 6.0)
        a4 = audio_f * (10.0 ** (gain_db / 20.0))
        augmented.append(a4)

        # 5) Time shift (circular, ±%20)
        max_shift = max(1, len(audio_f) // 5)
        shift = np.random.randint(-max_shift, max_shift)
        a5 = np.roll(audio_f, shift)
        augmented.append(a5)

        # 6) SpecAugment tarzi: mel spektrogram uzerinde frekans bandi maskeleme
        try:
            S = librosa.feature.melspectrogram(y=audio_f, sr=sr, n_mels=64)
            n_mels, n_frames = S.shape
            # Rastgele 4-12 mel bandi sifirla
            f_start = np.random.randint(0, max(1, n_mels - 12))
            f_width = np.random.randint(4, min(13, n_mels - f_start))
            S_masked = S.copy()
            S_masked[f_start:f_start + f_width, :] = 0
            # Griffin-Lim ile geri donustur
            a6 = librosa.feature.inverse.mel_to_audio(S_masked, sr=sr, n_iter=16)
            if len(a6) >= len(audio_f) // 2:
                augmented.append(a6[:len(audio_f)] if len(a6) >= len(audio_f) else
                                 np.pad(a6, (0, len(audio_f) - len(a6))))
            else:
                augmented.append(audio_f.copy())
        except Exception:
            augmented.append(audio_f.copy())

        return augmented

    def _extract_visual_11d(self, image_path: str) -> np.ndarray:
        """Gorsel dosyadan 11-D ozellik vektoru cikarir (VisualAnalyzer)."""
        if not hasattr(self, "_visual_analyzer"):
            from backend.module_a.visual_analyzer import VisualAnalyzer
            self._visual_analyzer = VisualAnalyzer()
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(11, dtype=np.float32)
        try:
            return self._visual_analyzer.extract_visual_feature_vector(img).astype(np.float32)
        except Exception:
            return np.zeros(11, dtype=np.float32)

    def _extract_xigua_64d(self, audio: np.ndarray, sr: int,
                           image_path: str) -> np.ndarray:
        """
        Xigua FusionModel'in 64-D ortak embedding'ini cikarir.

        Image yoksa veya model yuklenemiyorsa sifir vektor doner.
        Lazy load: ilk cagirimda model ve resampler hazirlanir.
        """
        if not hasattr(self, "_xigua_model"):
            from backend.module_e.xigua_model import XiguaPytorchModel
            from pathlib import Path as _P
            project_root = _P(__file__).parent.parent.parent
            xigua_path = str(project_root / "dataset" / "datasets"
                             / "watermelon_dataset" / "xigua_pytorch.pth")
            try:
                self._xigua_model = XiguaPytorchModel(xigua_path)
            except Exception as e:
                print(f"[DataLoader] Xigua yuklenemedi: {e}")
                self._xigua_model = None

        if self._xigua_model is None or not self._xigua_model.is_loaded:
            return np.zeros(64, dtype=np.float32)
        if not image_path or not os.path.exists(str(image_path)):
            return np.zeros(64, dtype=np.float32)

        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return np.zeros(64, dtype=np.float32)

            # Xigua model 16000 Hz bekler — gerekirse resample
            if sr != 16000:
                import librosa
                audio_16k = librosa.resample(audio.astype(np.float32),
                                             orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio.astype(np.float32)

            return self._xigua_model.extract_embedding(audio_16k, img)
        except Exception:
            return np.zeros(64, dtype=np.float32)

    def _extract_hh_8d(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Ses sinyalinden 8-D hollow-heart ozellik vektoru cikarir."""
        if not hasattr(self, "_hh_detector"):
            from backend.module_e.hollow_heart_detector import HollowHeartDetector
            self._hh_detector = HollowHeartDetector(sample_rate=sr)
        try:
            result = self._hh_detector.detect(audio, sr=sr, verbose=False)
            indicators = result.get("indicators", {})
            dp = indicators.get("dual_peak", {}).get("score", 0.0)
            dm = indicators.get("damping", {}).get("score", 0.0)
            sp = indicators.get("spectral", {}).get("score", 0.0)
            cp = indicators.get("cepstral", {}).get("score", 0.0)
            hnr = indicators.get("hnr", {}).get("score", 0.0)
            hh_score = result.get("hh_score", 0.0)
            confidence = result.get("confidence", 0.0)
            active_n = result.get("active_indicators", 0) / 5.0
            return np.array([dp, dm, sp, cp, hnr, hh_score, confidence, active_n],
                            dtype=np.float32)
        except Exception:
            return np.zeros(8, dtype=np.float32)

    def extract_multimodal_features(
        self,
        labels_df: pd.DataFrame = None,
        augment: bool = True,
        n_augment: int = 2,
        use_xigua: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ses + gorsel + HH kaynaklarindan 146-D fusion vektoru cikarir.

        Vektor yapisi:
          [0:120]   Akustik (AcousticFeatureExtractor)
          [120:131] Gorsel (VisualAnalyzer, 11-D)
          [131:138] Haptik (sifir — Qilin'de IMU yok)
          [138:146] Hollow heart (HollowHeartDetector, 8-D)

        Args:
            labels_df: filename, brix, category, wav_path, jpg_path sutunlariyla DF
            augment: Veri artirma aktif mi (her orijinal ornek icin n_augment varyasyon)
            n_augment: Orijinal basina eklenecek augment varyasyon sayisi (0..3)

        Returns:
            (X, y): (n, 146) ozellik matrisi ve (n,) etiket vektoru
        """
        if labels_df is None:
            labels_df = self.load_labels()

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        groups_list: List[str] = []
        errors: List[str] = []

        total = len(labels_df)
        print(f"\n[DataLoader] Multimodal ozellik cikariliyor: {total} ornek "
              f"(augment={augment}, x{1 + (n_augment if augment else 0)})")

        for idx, row in tqdm(labels_df.iterrows(), total=total,
                             desc="Multimodal"):
            wav_path = row.get("wav_path")
            jpg_path = row.get("jpg_path") if "jpg_path" in row else None
            if not wav_path or not os.path.exists(wav_path):
                errors.append(f"WAV yok: {row.get('filename', idx)}")
                continue

            try:
                audio, sr = load_wav_right_channel(wav_path, target_sr=self.sample_rate)
                audio = normalize_audio(audio)
                audio = trim_silence(audio, sr)
                if len(audio) < sr * 0.05:
                    errors.append(f"Cok kisa: {row.get('filename', idx)}")
                    continue
            except Exception as e:
                errors.append(f"{row.get('filename', idx)} audio: {e}")
                continue

            # Gorsel: gorsel her augment varyasyonunda ayni (fotograf sabit)
            visual_vec = (self._extract_visual_11d(jpg_path)
                          if jpg_path and os.path.exists(str(jpg_path))
                          else np.zeros(11, dtype=np.float32))

            haptic_vec = np.zeros(7, dtype=np.float32)  # Qilin'de IMU yok

            # Orijinal + augment varyasyonlar
            audio_variants = [audio]
            if augment and n_augment > 0:
                try:
                    augs = self._augment_audio(audio, sr)[:n_augment]
                    audio_variants.extend(augs)
                except Exception as e:
                    errors.append(f"{row.get('filename', idx)} augment: {e}")

            # Xigua embedding orijinal sesle bir kez hesaplanir (augment'larda paylasilir)
            xigua_vec = None
            if use_xigua:
                xigua_vec = self._extract_xigua_64d(audio, sr, jpg_path)

            for v_idx, av in enumerate(audio_variants):
                try:
                    acoustic_vec = self.feature_extractor.extract_feature_vector(av, sr)
                    hh_vec = self._extract_hh_8d(av, sr)
                    parts = [acoustic_vec.astype(np.float32),
                             visual_vec, haptic_vec, hh_vec]
                    if use_xigua:
                        parts.append(xigua_vec if xigua_vec is not None
                                     else np.zeros(64, dtype=np.float32))
                    full = np.concatenate(parts)
                    X_list.append(full)
                    y_list.append(int(row["category"]))
                    groups_list.append(str(row.get("data_id", idx)))
                except Exception as e:
                    errors.append(f"{row.get('filename', idx)}#{v_idx}: {e}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        self._last_groups = np.array(groups_list)

        print(f"\n[DataLoader] Multimodal sonuc: {len(X)} ornek, "
              f"{X.shape[1] if len(X) > 0 else 0} ozellik")
        if len(y) > 0:
            print(f"[DataLoader] Sinif dagilimi: "
                  f"{dict(zip(*np.unique(y, return_counts=True)))}")
        if errors:
            print(f"[DataLoader] {len(errors)} hata:")
            for e in errors[:5]:
                print(f"  - {e}")
            if len(errors) > 5:
                print(f"  ... ve {len(errors)-5} hata daha")

        return X, y

    def extract_features_from_audio(
        self,
        audio_dir: str = None,
        labels_df: pd.DataFrame = None,
        use_right_channel: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tum ses dosyalarindan ozellik cikarir.
        
        Qilin dataset icin ozel:
          - 2 kanalli WAV dosyalari
          - Sag kanal (channel 1) kullanilir
          - 16000 ornek uzunlugunda kesilir
        
        Args:
            audio_dir: Ses dosyalari dizini (None ise otomatik kesif)
            labels_df: Etiket DataFrame'i (None ise otomatik yukleme)
            use_right_channel: Sag kanali kullan (Qilin icin True)
            
        Returns:
            (X, y): Ozellik matrisi ve etiketler
        """
        if labels_df is None:
            labels_df = self.load_labels()

        # audio_dir otomatik kesif (Qilin yapisi icin gerek yok)
        if audio_dir is None and "wav_path" not in labels_df.columns:
            discovery = self.auto_discover_structure()
            audio_dir = discovery.get("wav_dir") or os.path.join(self.data_dir, "audio")

        X_list = []
        y_list = []
        groups_list = []
        real_count = 0
        synthetic_count = 0
        errors = []

        total = len(labels_df)
        print(f"\n[DataLoader] {total} ornekten ozellik cikariliyor...")

        for idx, row in tqdm(labels_df.iterrows(), total=total,
                             desc="Ozellik cikarma"):
            # Dosya yolunu belirle
            if "wav_path" in row and pd.notna(row.get("wav_path")):
                wav_path = row["wav_path"]
            elif audio_dir:
                wav_path = os.path.join(audio_dir, row["filename"])
            else:
                wav_path = None

            # Dosya yoksa alternatif yollari dene
            if wav_path and not os.path.exists(wav_path):
                candidates = list(Path(self.data_dir).rglob(row["filename"]))
                if not candidates:
                    # dataset klasorunde de ara
                    candidates = list(Path(self.qilin_datasets_dir).rglob(row["filename"])) if os.path.exists(self.qilin_datasets_dir) else []
                if candidates:
                    wav_path = str(candidates[0])

            if wav_path and os.path.exists(wav_path):
                try:
                    # Qilin dataset: 2 kanalli -> sag kanal kullan
                    if use_right_channel:
                        audio, sr = load_wav_right_channel(wav_path, target_sr=self.sample_rate)
                    else:
                        audio, sr = load_wav(wav_path, target_sr=self.sample_rate)
                    
                    audio = normalize_audio(audio)
                    audio = trim_silence(audio, sr)

                    if len(audio) < sr * 0.05:
                        errors.append(f"Cok kisa: {row.get('filename', idx)}")
                        continue

                    features = self.feature_extractor.extract_feature_vector(audio, sr)
                    X_list.append(features)
                    y_list.append(int(row["category"]))
                    groups_list.append(str(row.get("data_id", idx)))
                    real_count += 1
                except Exception as e:
                    errors.append(f"{row.get('filename', idx)}: {e}")
            else:
                # Sentetik ozellik olustur (veri dosyasi yoksa)
                features = self._generate_synthetic_features(int(row["category"]))
                X_list.append(features)
                y_list.append(int(row["category"]))
                groups_list.append(str(row.get("data_id", f"synth_{idx}")))
                synthetic_count += 1

        X = np.array(X_list)
        y = np.array(y_list)
        self._last_groups = np.array(groups_list)

        print(f"\n[DataLoader] Sonuc: {len(X)} ornek, {X.shape[1] if len(X) > 0 else 0} ozellik")
        print(f"[DataLoader] Gercek: {real_count}, Sentetik: {synthetic_count}")
        if len(y) > 0:
            print(f"[DataLoader] Sinif dagilimi: {dict(zip(*np.unique(y, return_counts=True)))}")
        if errors:
            print(f"[DataLoader] {len(errors)} hata:")
            for e in errors[:5]:
                print(f"  - {e}")
            if len(errors) > 5:
                print(f"  ... ve {len(errors)-5} hata daha")

        return X, y

    def extract_features_with_images(
        self,
        labels_df: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Qilin dataset'ten ses + gorsel eslestirmeli ozellik cikarir.
        
        Returns:
            (X_audio, y, image_paths): Ses ozellikleri, etiketler, gorsel yollari
        """
        if labels_df is None:
            labels_df = self.load_labels()
        
        X_audio, y = self.extract_features_from_audio(labels_df=labels_df)
        
        image_paths = []
        if "jpg_path" in labels_df.columns:
            for _, row in labels_df.iterrows():
                jpg = row.get("jpg_path")
                if jpg and os.path.exists(str(jpg)):
                    image_paths.append(str(jpg))
                else:
                    image_paths.append(None)
        
        print(f"[DataLoader] {len([p for p in image_paths if p])} gorsel eslesti")
        return X_audio, y, image_paths

    def _generate_synthetic_features(self, category: int) -> np.ndarray:
        """
        Koç & Akbalık (2025) uyumlu 120 boyutlu sentetik özellik vektörü.

        Her sınıf için fiziksel olarak tutarlı özellikler:
        - Olgunlaşmamış: Yüksek f2, yüksek ZCR, hızlı decay
        - Olgun: Düşük f2 (<150Hz), orta ZCR, yavaş decay
        - İçi geçmiş: Çok düşük f2, düşük ZCR, çok hızlı decay

        120 boyut dağılımı:
          [0:52]    MFCC istatistikleri (mean/std/min/max × 13)
          [52:78]   Delta MFCC (mean/std × 13)
          [78:80]   ZCR (mean, std)
          [80:95]   Spektral (centroid/bw/rolloff/flatness/contrast)
          [95:99]   Enerji (rms/log_energy)
          [99:111]  Chroma (12 pitch sınıfı)
          [111:114] Frekans & rezonans (f2, f2_db, entropy)
          [114:120] Zaman alanı (peak/crest/centroid/attack/decay/duration)
        """
        np.random.seed(None)

        # --- Sınıf bazlı fiziksel parametreler ---
        if category == 0:  # Olgunlasmamis
            f2 = np.random.uniform(160, 250)
            zcr_mean = np.random.uniform(0.08, 0.15)
            f2_db = np.random.uniform(20, 30)
            spectral_entropy = np.random.uniform(0.6, 0.8)
            decay_rate = np.random.uniform(15, 30)  # Hızlı sönüm (sert)
            crest_factor = np.random.uniform(3.0, 5.0)
            attack_time = np.random.uniform(0.001, 0.005)
        elif category == 1:  # Olgun
            f2 = np.random.uniform(80, 145)
            zcr_mean = np.random.uniform(0.04, 0.09)
            f2_db = np.random.uniform(26, 35)
            spectral_entropy = np.random.uniform(0.3, 0.6)
            decay_rate = np.random.uniform(5, 15)  # Yavaş sönüm (olgun)
            crest_factor = np.random.uniform(4.0, 7.0)
            attack_time = np.random.uniform(0.002, 0.008)
        else:  # Ici gecmis
            f2 = np.random.uniform(50, 90)
            zcr_mean = np.random.uniform(0.01, 0.05)
            f2_db = np.random.uniform(15, 25)
            spectral_entropy = np.random.uniform(0.7, 0.95)
            decay_rate = np.random.uniform(20, 50)  # Çok hızlı sönüm
            crest_factor = np.random.uniform(2.0, 4.0)
            attack_time = np.random.uniform(0.005, 0.015)

        # Grup 1: MFCC İstatistikleri (52) - mean/std/min/max × 13
        mfcc_mean = np.random.randn(13) * 5
        mfcc_std = np.abs(np.random.randn(13) * 2) + 0.5
        mfcc_min = mfcc_mean - mfcc_std * 2 + np.random.randn(13) * 0.5
        mfcc_max = mfcc_mean + mfcc_std * 2 + np.random.randn(13) * 0.5
        mfcc_stats = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])

        # Grup 2: Delta MFCC (26) - mean/std × 13
        delta_mean = np.random.randn(13) * 1.0
        delta_std = np.abs(np.random.randn(13) * 0.5) + 0.1
        delta_stats = np.concatenate([delta_mean, delta_std])

        # Grup 3: ZCR (2)
        zcr_stats = np.array([zcr_mean, np.random.uniform(0.005, 0.03)])

        # Grup 4: Spektral (15)
        spectral_centroid_mean = f2 * np.random.uniform(1.5, 3.0)
        spectral_centroid_std = spectral_centroid_mean * np.random.uniform(0.1, 0.3)
        spectral_bw_mean = np.random.uniform(500, 2000)
        spectral_bw_std = spectral_bw_mean * np.random.uniform(0.1, 0.3)
        spectral_rolloff_mean = np.random.uniform(1000, 5000)
        spectral_rolloff_std = spectral_rolloff_mean * np.random.uniform(0.05, 0.2)
        spectral_flatness_mean = np.random.uniform(0.01, 0.3)
        spectral_flatness_std = spectral_flatness_mean * np.random.uniform(0.1, 0.5)
        spectral_contrast = np.random.uniform(10, 40, 7)
        spectral_vec = np.concatenate([
            [spectral_centroid_mean, spectral_centroid_std,
             spectral_bw_mean, spectral_bw_std,
             spectral_rolloff_mean, spectral_rolloff_std,
             spectral_flatness_mean, spectral_flatness_std],
            spectral_contrast
        ])

        # Grup 5: Enerji (4)
        rms_mean = np.random.uniform(0.01, 0.2)
        rms_std = rms_mean * np.random.uniform(0.3, 0.8)
        log_energy_mean = np.log(rms_mean + 1e-10)
        log_energy_std = np.random.uniform(0.5, 2.0)
        energy_vec = np.array([rms_mean, rms_std, log_energy_mean, log_energy_std])

        # Grup 6: Chroma (12)
        chroma_vec = np.random.uniform(0.1, 0.6, 12)

        # Grup 7: Frekans & Rezonans (3)
        freq_vec = np.array([f2, f2_db, spectral_entropy])

        # Grup 8: Zaman Alanı (6)
        peak_amp = np.random.uniform(0.3, 0.9)
        temporal_centroid = np.random.uniform(0.05, 0.2)
        duration_norm = np.random.uniform(0.4, 0.9)
        time_vec = np.array([
            peak_amp, crest_factor, temporal_centroid,
            attack_time, decay_rate, duration_norm
        ])

        # Birleştir: 52 + 26 + 2 + 15 + 4 + 12 + 3 + 6 = 120
        vector = np.concatenate([
            mfcc_stats,     # 52
            delta_stats,    # 26
            zcr_stats,      #  2
            spectral_vec,   # 15
            energy_vec,     #  4
            chroma_vec,     # 12
            freq_vec,       #  3
            time_vec,       #  6
        ])

        assert len(vector) == 120, f"Sentetik vektör boyut hatası: {len(vector)}"
        return vector

    def save_processed_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = None
    ):
        """Islenmis veriyi diske kaydeder."""
        if output_dir is None:
            output_dir = str(PROCESSED_DATA_DIR)

        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "X_features.npy"), X)
        np.save(os.path.join(output_dir, "y_labels.npy"), y)
        groups = getattr(self, "_last_groups", None)
        if groups is not None and len(groups) == len(X):
            np.save(os.path.join(output_dir, "groups.npy"), groups)
        print(f"[DataLoader] Veri kaydedildi: {output_dir}")

    def load_processed_data(
        self,
        data_dir: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Islenmis veriyi diskten yukler."""
        if data_dir is None:
            data_dir = str(PROCESSED_DATA_DIR)

        X = np.load(os.path.join(data_dir, "X_features.npy"))
        y = np.load(os.path.join(data_dir, "y_labels.npy"))
        groups_path = os.path.join(data_dir, "groups.npy")
        if os.path.exists(groups_path):
            self._last_groups = np.load(groups_path, allow_pickle=True)
        else:
            self._last_groups = None
        print(f"[DataLoader] Veri yuklendi: {X.shape[0]} ornek, {X.shape[1]} ozellik")

        return X, y

    def get_dataset_info(self) -> Dict:
        """Dataset hakkinda ozet bilgi dondurur."""
        discovery = self.auto_discover_structure()
        
        info = {
            "structure_type": discovery.get("structure_type", "bulunamadi"),
            "wav_count": discovery.get("wav_count", 0),
            "image_count": discovery.get("image_count", 0),
            "qilin_datasets_dir": self.qilin_datasets_dir,
            "data_dir": self.data_dir,
        }
        
        if discovery.get("structure_type") == "qilin_original":
            samples = discovery["qilin_samples"]
            categories = [s["category"] for s in samples]
            info["sample_count"] = len(samples)
            info["brix_range"] = discovery.get("brix_range")
            info["class_distribution"] = {
                "Olgunlasmamis (0)": categories.count(0),
                "Olgun (1)": categories.count(1),
                "Ici Gecmis (2)": categories.count(2)
            }
            # Ornek gosterimi
            if samples:
                info["sample_example"] = samples[0]
        
        return info
