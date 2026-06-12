"""Telefonun verecegi tahminleri masaustunde onizler.

parity_check.dart'in urettigi Dart-hesapli featurelari fusion TFLite'a verir;
ayni ornekler icin Python-hesapli featurelarla (precomputed_features.json)
yan yana karsilastirir. Boylelikle sahaya cikmadan "telefon ne diyecek"
sorusunun cevabini gorur, v13 tarzi dagitim kaymasi varsa YAKALAR.

Calistirma (flutter_app/ icinden):
    python tool/parity_predict.py
"""
import json
import numpy as np
import tensorflow as tf

CLASSES = ["Olgunlasmamis", "Olgun", "IciGecmis"]

# --- Modeli yukle ---
with open("assets/models/fusion_model_fp16.tflite", "rb") as f:
    interp = tf.lite.Interpreter(model_content=f.read())
interp.allocate_tensors()
inputs = interp.get_input_details()
out_det = interp.get_output_details()[0]

# --- Self-test'in kullandigi training-ortalama visual/haptic vektorler ---
with open("assets/samples/feature_means.json") as f:
    means = json.load(f)
visual_mean = np.array(means["visual_mean"], dtype=np.float32).reshape(1, 11)
haptic_mean = np.array(means["haptic_mean"], dtype=np.float32).reshape(1, 7)


def predict(acoustic120, hh8):
    feed = {
        120: np.asarray(acoustic120, dtype=np.float32).reshape(1, 120),
        11: visual_mean,
        7: haptic_mean,
        8: np.asarray(hh8, dtype=np.float32).reshape(1, 8),
    }
    for ip in inputs:
        interp.set_tensor(ip["index"], feed[ip["shape"][1]])
    interp.invoke()
    p = interp.get_tensor(out_det["index"])[0]
    return int(np.argmax(p)), p


# --- Dart features (canli pipeline simulasyonu) ---
with open("tool/dart_features_all_samples.json") as f:
    dart_feats = json.load(f)

# --- Python features (backend referansi) ---
with open("assets/samples/precomputed_features.json") as f:
    py_feats = json.load(f)["features"]

print(f"{'ornek':28s} {'gercek':14s} {'DART tahmin':22s} {'PYTHON tahmin':22s} uyum")
print("-" * 100)
n_total = n_dart_ok = n_py_ok = n_agree = 0
n_bin_dart = n_bin_py = 0
for fname, d in dart_feats.items():
    label = int(d["label"])
    cd, pd_ = predict(d["acoustic_120"], d["hh_8"])
    pf = py_feats[fname]
    cp, pp = predict(pf["acoustic_120"], pf["hh_8"])

    n_total += 1
    n_dart_ok += int(cd == label)
    n_py_ok += int(cp == label)
    n_agree += int(cd == cp)
    n_bin_dart += int((cd == 1) == (label == 1))
    n_bin_py += int((cp == 1) == (label == 1))

    mark = "AYNI" if cd == cp else ">>> FARKLI <<<"
    print(f"{fname:28s} {CLASSES[label]:14s} "
          f"{CLASSES[cd]:13s}({pd_.max():.2f}) "
          f"{CLASSES[cp]:13s}({pp.max():.2f})  {mark}")

print("-" * 100)
print(f"3-sinif dogruluk : DART {n_dart_ok}/{n_total}   PYTHON {n_py_ok}/{n_total}")
print(f"2-sinif dogruluk : DART {n_bin_dart}/{n_total}   PYTHON {n_bin_py}/{n_total}")
print(f"Dart-Python tahmin uyumu: {n_agree}/{n_total}"
      f"  ({'PARITY TAM' if n_agree == n_total else 'KAYMA VAR - incele!'})")

# --- Hibrit teshis: kalan farklar akustikten mi HH'den mi geliyor? ---
n_hyb_acoustic = n_hyb_hh = 0
for fname, d in dart_feats.items():
    pf = py_feats[fname]
    cp, _ = predict(pf["acoustic_120"], pf["hh_8"])
    # Dart akustik + Python HH
    ca, _ = predict(d["acoustic_120"], pf["hh_8"])
    # Python akustik + Dart HH
    ch, _ = predict(pf["acoustic_120"], d["hh_8"])
    n_hyb_acoustic += int(ca == cp)
    n_hyb_hh += int(ch == cp)
print(f"\nHibrit teshis (Python tahminiyle uyum):")
print(f"  Dart akustik + Python HH : {n_hyb_acoustic}/{n_total}  -> akustik kaynakli fark: {n_total-n_hyb_acoustic}")
print(f"  Python akustik + Dart HH : {n_hyb_hh}/{n_total}  -> HH kaynakli fark: {n_total-n_hyb_hh}")
