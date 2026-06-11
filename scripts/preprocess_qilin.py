"""
Qilin Watermelon Dataset On Isleme Script

Qilin veri setindeki WAV + JPG dosyalarini isleyerek
ozellik matrislerine donusturur.

Qilin Dataset Yapisi:
  dataset/datasets/{id}_{brix}/chu/{folder}/{*.wav, *.jpg}

Kullanim:
    python scripts/preprocess_qilin.py
    python scripts/preprocess_qilin.py --input dataset/datasets --output data/processed
"""

import numpy as np
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.config import (
    PROCESSED_DATA_DIR, AUDIO_SAMPLE_RATE,
    QILIN_DATASETS_DIR
)
from backend.utils.audio_utils import load_wav, normalize_audio, trim_silence
from backend.module_d.feature_extractor import AcousticFeatureExtractor
from backend.pipeline.data_loader import QilinDatasetLoader


def preprocess_dataset(
    input_dir: str = None,
    output_dir: str = None,
    sample_rate: int = AUDIO_SAMPLE_RATE
):
    """
    Qilin veri setini isler ve ozellik matrislerini olusturur.
    
    Islemler:
    1. Qilin yapisini otomatik kesfet ({id}_{brix}/chu/{folder}/{wav,jpg})
    2. Her WAV dosyasini yukle (sag kanal kullanilir)
    3. Normalize et ve sessizlik kirp
    4. MFCC, ZCR, f2 ozelliklerini cikar
    5. .npy olarak kaydet
    6. Gorsel dosya yollarini kaydet
    """
    if input_dir is None:
        input_dir = str(QILIN_DATASETS_DIR)
    if output_dir is None:
        output_dir = str(PROCESSED_DATA_DIR)

    os.makedirs(output_dir, exist_ok=True)
    npy_dir = os.path.join(output_dir, "npy_audio")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    print("=" * 60)
    print("  Qilin Watermelon Dataset On Isleme")
    print(f"  Giris: {input_dir}")
    print(f"  Cikis: {output_dir}")
    print("=" * 60)

    # Qilin yapisini kullanarak veri yukle
    loader = QilinDatasetLoader()
    discovery = loader.auto_discover_structure()

    if discovery.get("structure_type") != "qilin_original":
        print("\n[!] Qilin orijinal yapisi bulunamadi!")
        print("    Beklenen yapi: dataset/datasets/{id}_{brix}/chu/{folder}/{*.wav, *.jpg}")
        print("    Alternatif olarak genel WAV arama yapiliyor...\n")
        _preprocess_generic(input_dir, output_dir, sample_rate)
        return

    samples = discovery["qilin_samples"]
    print(f"\n{len(samples)} Qilin ornegi bulundu.")

    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

    all_features = []
    all_labels = []
    all_brix = []
    all_filenames = []
    all_jpg_paths = []
    errors = []

    for sample in tqdm(samples, desc="Ozellik cikarma"):
        wav_path = sample["wav_path"]
        try:
            # 1) Ses yukle
            audio, sr = load_wav(wav_path, target_sr=sample_rate)

            # 2) Qilin: 2 kanalli -> sag kanal
            if len(audio.shape) > 1 and audio.shape[0] >= 2:
                audio = audio[1]  # Sag kanal
            elif len(audio.shape) > 1:
                audio = audio[0]  # Mono

            # 3) Normalize et
            audio = normalize_audio(audio)

            # 4) Sessizlik kirp
            audio = trim_silence(audio, sr)

            # Cok kisa sinyalleri atla
            if len(audio) < sr * 0.05:
                errors.append(f"Cok kisa: {os.path.basename(wav_path)}")
                continue

            # 5) .npy olarak kaydet
            npy_name = f"{sample['data_id']}_{os.path.basename(wav_path).replace('.wav', '')}.npy"
            npy_path = os.path.join(npy_dir, npy_name)
            np.save(npy_path, audio)

            # 6) Ozellik cikar
            feature_vector = extractor.extract_feature_vector(audio, sr)
            all_features.append(feature_vector)
            all_labels.append(sample["category"])
            all_brix.append(sample["brix"])
            all_filenames.append(os.path.basename(wav_path))
            all_jpg_paths.append(sample.get("jpg_path", ""))

        except Exception as e:
            errors.append(f"{os.path.basename(wav_path)}: {e}")

    if all_features:
        X = np.array(all_features)
        y = np.array(all_labels)
        brix = np.array(all_brix)

        # Ozellik matrisi kaydet
        np.save(os.path.join(features_dir, "X_acoustic_features.npy"), X)
        np.save(os.path.join(features_dir, "y_labels.npy"), y)
        np.save(os.path.join(features_dir, "brix_values.npy"), brix)

        # Ayrica ana processed dizinine de kaydet (train.py uyumu)
        np.save(os.path.join(output_dir, "X_features.npy"), X)
        np.save(os.path.join(output_dir, "y_labels.npy"), y)

        # Metadata kaydet
        metadata = {
            "total_samples": len(X),
            "feature_dim": int(X.shape[1]),
            "brix_range": [float(brix.min()), float(brix.max())],
            "class_distribution": {
                "0_olgunlasmamis": int(np.sum(y == 0)),
                "1_olgun": int(np.sum(y == 1)),
                "2_ici_gecmis": int(np.sum(y == 2))
            },
            "filenames": all_filenames,
            "jpg_paths": all_jpg_paths
        }

        with open(os.path.join(features_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Dosya adlarini kaydet
        with open(os.path.join(features_dir, "filenames.txt"), 'w', encoding='utf-8') as f:
            for name in all_filenames:
                f.write(f"{name}\n")

        print(f"\n{'='*60}")
        print(f"  Tamamlandi!")
        print(f"  Ham ses (.npy): {npy_dir}")
        print(f"  Ozellik matrisi: {features_dir}/X_acoustic_features.npy")
        print(f"  Boyut: {X.shape} ({X.shape[0]} ornek, {X.shape[1]} ozellik)")
        print(f"  Brix araligi: {brix.min():.1f} - {brix.max():.1f}")
        print(f"  Sinif dagilimi:")
        print(f"    0 (Olgunlasmamis): {np.sum(y == 0)}")
        print(f"    1 (Olgun):         {np.sum(y == 1)}")
        print(f"    2 (Ici Gecmis):    {np.sum(y == 2)}")
        if errors:
            print(f"  Hatalar: {len(errors)}")
            for e in errors[:5]:
                print(f"    - {e}")
        print(f"{'='*60}")
    else:
        print("Hicbir dosya islenemedi!")
        if errors:
            print("Hatalar:")
            for e in errors:
                print(f"  - {e}")


def _preprocess_generic(input_dir, output_dir, sample_rate):
    """Genel WAV dosyasi isleme (Qilin yapisi bulunamazsa)."""
    npy_dir = os.path.join(output_dir, "npy_audio")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

    wav_files = list(Path(input_dir).rglob("*.wav"))
    wav_files += list(Path(input_dir).rglob("*.WAV"))

    if not wav_files:
        print(f"UYARI: {input_dir} dizininde WAV dosyasi bulunamadi!")
        print("Qilin veri setini indirip dataset/datasets dizinine yerlestirin.")
        return

    print(f"Toplam {len(wav_files)} WAV dosyasi bulundu.")

    all_features = []
    all_filenames = []

    for wav_path in tqdm(wav_files, desc="WAV -> Ozellik"):
        try:
            audio, sr = load_wav(str(wav_path), target_sr=sample_rate)

            if len(audio.shape) > 1:
                audio = audio[1] if audio.shape[0] >= 2 else audio[0]

            audio = normalize_audio(audio)
            audio = trim_silence(audio, sr)

            npy_path = os.path.join(npy_dir, f"{wav_path.stem}.npy")
            np.save(npy_path, audio)

            feature_vector = extractor.extract_feature_vector(audio, sr)
            all_features.append(feature_vector)
            all_filenames.append(wav_path.stem)

        except Exception as e:
            print(f"\nHata ({wav_path.name}): {e}")

    if all_features:
        X = np.array(all_features)
        np.save(os.path.join(features_dir, "X_acoustic_features.npy"), X)

        with open(os.path.join(features_dir, "filenames.txt"), 'w', encoding='utf-8') as f:
            for name in all_filenames:
                f.write(f"{name}\n")

        print(f"\nTamamlandi!")
        print(f"  Boyut: {X.shape}")
    else:
        print("Hicbir dosya islenemedi!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qilin veri seti on isleme")
    parser.add_argument("--input", type=str, default=None,
                        help="Dataset dizini (varsayilan: dataset/datasets)")
    parser.add_argument("--output", type=str, default=None,
                        help="Cikis dizini (varsayilan: data/processed)")
    parser.add_argument("--sr", type=int, default=AUDIO_SAMPLE_RATE,
                        help="Ornekleme hizi")

    args = parser.parse_args()
    preprocess_dataset(args.input, args.output, args.sr)
