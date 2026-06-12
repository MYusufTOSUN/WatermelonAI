"""precomputed_features.json'daki hh_8 vektorlerini EGITIMLE BIREBIR ayni
kod yolundan yeniden uretir.

Onceki uretim extract_hollow_heart_features() kullanmisti:
    [hh_score, dual_peak, damping, spectral, cepstral, hnr_score, hnr_db, damping_ratio]
Egitim pipeline'i (backend/pipeline/data_loader.py::_extract_hh_8d) ise:
    [dp, dm, sp, cp, hnr, hh_score, confidence, active_n]
kullaniyor — USTELIK 'spectral' anahtari detektorde 'spectral_spread' oldugu
icin egitimde dim2 HEP 0.0 kalmis (anahtar-adi bug'i). Model bu dagilimla
egitildi; referans da ayni yoldan uretilmeli (bug dahil).

Calistirma (flutter_app/ icinden):  python tool/regen_hh_reference.py
"""
import json
import sys

sys.path.insert(0, "..")
import numpy as np
from backend.module_e.hollow_heart_detector import HollowHeartDetector
import librosa

SR = 44100
det = HollowHeartDetector(sample_rate=SR)


def hh_training_exact(audio):
    """data_loader._extract_hh_8d ile birebir ayni (bug dahil)."""
    result = det.detect(audio, sr=SR, verbose=False)
    ind = result.get("indicators", {})
    dp = ind.get("dual_peak", {}).get("score", 0.0)
    dm = ind.get("damping", {}).get("score", 0.0)
    sp = ind.get("spectral", {}).get("score", 0.0)   # anahtar yok -> 0.0 (egitim bug'i korunuyor)
    cp = ind.get("cepstral", {}).get("score", 0.0)
    hnr = ind.get("hnr", {}).get("score", 0.0)
    hh_score = result.get("hh_score", 0.0)
    confidence = result.get("confidence", 0.0)
    active_n = result.get("active_indicators", 0) / 5.0
    return [float(x) for x in (dp, dm, sp, cp, hnr, hh_score, confidence, active_n)]


with open("assets/samples/precomputed_features.json") as f:
    data = json.load(f)

with open("assets/samples/manifest.json") as f:
    manifest = json.load(f)

names = ["dp", "dm", "sp", "cp", "hnr", "hh", "conf", "act"]
rows = []
for s in manifest["samples"]:
    asset = s["asset"]
    fname = asset.split("/")[-1]
    audio, _ = librosa.load(asset, sr=SR, mono=True)
    hh8 = hh_training_exact(audio)
    data["features"][fname]["hh_8"] = hh8
    rows.append(hh8)
    print(f"{fname:26s} " + " ".join(f"{v:6.3f}" for v in hh8))

data["note"] = ("Python-extracted features for bundled WAVs. acoustic_120: "
                "module_d extract_feature_vector. hh_8: TRAINING-EXACT layout "
                "[dp,dm,sp,cp,hnr,hh_score,confidence,active_n] via "
                "data_loader._extract_hh_8d code path (spectral-key bug dahil).")
with open("assets/samples/precomputed_features.json", "w") as f:
    json.dump(data, f)

R = np.array(rows)
print("\nuretilen dagilim   : ort=" + " ".join(f"{v:5.3f}" for v in R.mean(0)))
print("egitim dagilimi ref: ort= 0.470 0.287 0.000 0.218 0.997 0.433 0.843 0.299")
print("\nGuncellendi: assets/samples/precomputed_features.json")
