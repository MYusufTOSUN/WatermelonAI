"""Yeniden egitim karsilastirma deneyi — eski modellere DOKUNMAZ.

Mevcut cache'li featurelar (data/processed/X_features.npy, 4671x146) ile:

  --stage rfc   : RFC varyantlarini tam 19-fold LOWO ile kiyasla (hizli)
  --stage dnn   : DNN varyantlarini GroupKFold(5) ile kiyasla (~15 dk)
  --stage final : Secilen DNN varyantini tum veriyle egit + FP16 TFLite uret
                  (fusion_model_fp16_v2.tflite — ESKISININ USTUNE YAZMAZ)

Cikti: data/models/retrain_compare_results.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

SEED = 42
np.random.seed(SEED)

ROOT = os.path.join(os.path.dirname(__file__), "..")
X = np.load(os.path.join(ROOT, "data/processed/X_features.npy")).astype(np.float32)
y = np.load(os.path.join(ROOT, "data/processed/y_labels.npy"))
groups = np.load(os.path.join(ROOT, "data/processed/groups.npy"))
RESULTS_PATH = os.path.join(ROOT, "data/models/retrain_compare_results.json")

# Egitim HH ortalamalari (telefon serve-time'da DNN'e bunu veriyor)
HH_TRAIN_MEAN = np.array([0.470, 0.287, 0.0, 0.218, 0.997, 0.433, 0.843, 0.299],
                         dtype=np.float32)

def binary(labels):
    return (labels == 1).astype(int)  # yenir=1, yenmez=0

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"timestamp": datetime.now().isoformat()}

def save_results(res):
    res["updated"] = datetime.now().isoformat()
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)

# =====================================================================
# RFC LOWO
# =====================================================================

def lowo_rfc(make_clf):
    from sklearn.base import clone
    uniq = np.unique(groups)
    acc3_list, acc2_list = [], []
    y3_all, p3_all = [], []
    for g in uniq:
        te = groups == g
        tr = ~te
        clf = clone(make_clf)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc3_list.append(float((pred == y[te]).mean()))
        acc2_list.append(float((binary(pred) == binary(y[te])).mean()))
        y3_all.append(y[te]); p3_all.append(pred)
    y3 = np.concatenate(y3_all); p3 = np.concatenate(p3_all)
    return {
        "lowo_acc3_mean": float(np.mean(acc3_list)),
        "lowo_acc3_std": float(np.std(acc3_list)),
        "lowo_acc2_mean": float(np.mean(acc2_list)),
        "lowo_acc2_agg": float((binary(p3) == binary(y3)).mean()),
        "lowo_acc3_agg": float((p3 == y3).mean()),
    }

def stage_rfc():
    from sklearn.ensemble import RandomForestClassifier
    res = load_results()
    res.setdefault("rfc", {})
    variants = {
        "R0_mevcut": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            min_samples_split=5, random_state=SEED, n_jobs=-1),
        "R1_balanced": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            min_samples_split=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "R2_bal_subsample_d10": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=2,
            min_samples_split=4, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1),
    }
    for name, clf in variants.items():
        print(f"[RFC] {name} LOWO calisiyor...")
        m = lowo_rfc(clf)
        res["rfc"][name] = m
        print(f"  3-sinif: {m['lowo_acc3_agg']:.3f}  2-sinif: {m['lowo_acc2_agg']:.3f}")
        save_results(res)
    print("\nRFC stage tamam.")

# =====================================================================
# DNN GroupKFold(5)
# =====================================================================

def split_inputs(Xa, serve_aligned=False):
    ac = Xa[:, :120]
    vi = Xa[:, 120:131]
    ha = Xa[:, 131:138]
    hh = Xa[:, 138:146]
    if serve_aligned:
        hh = np.tile(HH_TRAIN_MEAN, (len(Xa), 1))
    return [ac, vi, ha, hh]

def build_dnn():
    import tensorflow as tf
    from backend.pipeline.tflite_converter import create_fusion_keras_model
    tf.keras.utils.set_random_seed(SEED)
    return create_fusion_keras_model()

def dnn_groupcv(variant, use_class_weight=False, serve_aligned=False,
                epochs=80, batch=64):
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold
    from sklearn.utils.class_weight import compute_class_weight

    gkf = GroupKFold(n_splits=5)
    acc3_list, acc2_list = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        tf.keras.backend.clear_session()
        model = build_dnn()
        cw = None
        if use_class_weight:
            w = compute_class_weight("balanced", classes=np.array([0, 1, 2]),
                                     y=y[tr])
            cw = {i: float(w[i]) for i in range(3)}
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True)
        model.fit(split_inputs(X[tr], serve_aligned), y[tr],
                  validation_split=0.15, epochs=epochs, batch_size=batch,
                  class_weight=cw, callbacks=[es], verbose=0)
        prob = model.predict(split_inputs(X[te], serve_aligned), verbose=0)
        pred = prob.argmax(1)
        a3 = float((pred == y[te]).mean())
        a2 = float((binary(pred) == binary(y[te])).mean())
        acc3_list.append(a3); acc2_list.append(a2)
        print(f"  [{variant}] fold{fold+1}: 3s={a3:.3f} 2s={a2:.3f}")
    return {
        "groupcv_acc3_mean": float(np.mean(acc3_list)),
        "groupcv_acc2_mean": float(np.mean(acc2_list)),
        "folds_acc3": acc3_list,
        "folds_acc2": acc2_list,
    }

def stage_dnn():
    res = load_results()
    res.setdefault("dnn", {})
    experiments = [
        ("D0_baseline", dict(use_class_weight=False, serve_aligned=False)),
        ("D1_class_weight", dict(use_class_weight=True, serve_aligned=False)),
        ("D2_serve_aligned_cw", dict(use_class_weight=True, serve_aligned=True)),
    ]
    for name, kw in experiments:
        print(f"[DNN] {name} GroupKFold(5) calisiyor...")
        m = dnn_groupcv(name, **kw)
        res["dnn"][name] = m
        print(f"  ORT: 3-sinif={m['groupcv_acc3_mean']:.3f} "
              f"2-sinif={m['groupcv_acc2_mean']:.3f}")
        save_results(res)
    print("\nDNN stage tamam.")

# =====================================================================
# FINAL: secilen varyanti tum veriyle egit + TFLite v2
# =====================================================================

def stage_final(variant):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    serve_aligned = variant == "D2_serve_aligned_cw"
    use_cw = variant in ("D1_class_weight", "D2_serve_aligned_cw")

    print(f"[FINAL] {variant} tum veriyle egitiliyor...")
    tf.keras.backend.clear_session()
    model = build_dnn()
    cw = None
    if use_cw:
        w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
        cw = {i: float(w[i]) for i in range(3)}
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True)
    model.fit(split_inputs(X, serve_aligned), y, epochs=150, batch_size=64,
              class_weight=cw, callbacks=[es], verbose=0)
    prob = model.predict(split_inputs(X, serve_aligned), verbose=0)
    acc = float((prob.argmax(1) == y).mean())
    print(f"  train acc: {acc:.3f}")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    out = os.path.join(ROOT, "data/models/fusion_model_fp16_v2.tflite")
    with open(out, "wb") as f:
        f.write(tfl)
    print(f"  TFLite v2 yazildi: {out} ({len(tfl)/1024:.1f} KB)")

    res = load_results()
    res["final"] = {"variant": variant, "train_acc": acc,
                    "tflite": "fusion_model_fp16_v2.tflite",
                    "size_kb": round(len(tfl) / 1024, 1)}
    save_results(res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rfc", "dnn", "final"])
    ap.add_argument("--variant", default="D2_serve_aligned_cw")
    a = ap.parse_args()
    print(f"Veri: X={X.shape} y={y.shape} gruplar={len(np.unique(groups))}")
    if a.stage == "rfc":
        stage_rfc()
    elif a.stage == "dnn":
        stage_dnn()
    else:
        stage_final(a.variant)
