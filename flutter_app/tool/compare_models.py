"""Eski fusion_model_fp16.tflite ile yeni v2'yi TELEFON-BESLEME kosullarinda
yan yana kiyaslar: Dart-hesapli akustik + egitim-ortalamasi visual/haptic/HH.

Calistirma (flutter_app/ icinden):  python tool/compare_models.py
"""
import json
import numpy as np
import tensorflow as tf

CLASSES = ["Olgunlasmamis", "Olgun", "IciGecmis"]
HH_TRAIN_MEAN = np.array([0.470, 0.287, 0.0, 0.218, 0.997, 0.433, 0.843, 0.299],
                         dtype=np.float32).reshape(1, 8)

with open("assets/samples/feature_means.json") as f:
    means = json.load(f)
VM = np.array(means["visual_mean"], dtype=np.float32).reshape(1, 11)
HM = np.array(means["haptic_mean"], dtype=np.float32).reshape(1, 7)


def make_predictor(path):
    with open(path, "rb") as f:
        interp = tf.lite.Interpreter(model_content=f.read())
    interp.allocate_tensors()
    inputs = interp.get_input_details()
    outd = interp.get_output_details()[0]

    def predict(ac):
        feed = {120: np.asarray(ac, dtype=np.float32).reshape(1, 120),
                11: VM, 7: HM, 8: HH_TRAIN_MEAN}
        for ip in inputs:
            interp.set_tensor(ip["index"], feed[ip["shape"][1]])
        interp.invoke()
        p = interp.get_tensor(outd["index"])[0]
        return int(np.argmax(p)), p
    return predict


old_p = make_predictor("assets/models/fusion_model_fp16.tflite")
new_p = make_predictor("../data/models/fusion_model_fp16_v2.tflite")

with open("tool/dart_features_all_samples.json") as f:
    dart = json.load(f)

print(f"{'ornek':26s} {'gercek':14s} {'ESKI model':20s} {'YENI model (v2)':20s}")
print("-" * 88)
n = o3 = n3 = o2 = n2 = 0
for fname, d in dart.items():
    label = int(d["label"])
    co, po = old_p(d["acoustic_120"])
    cn, pn = new_p(d["acoustic_120"])
    n += 1
    o3 += co == label; n3 += cn == label
    o2 += (co == 1) == (label == 1); n2 += (cn == 1) == (label == 1)
    print(f"{fname:26s} {CLASSES[label]:14s} "
          f"{CLASSES[co]:13s}({po.max():.2f}) {CLASSES[cn]:13s}({pn.max():.2f})")
print("-" * 88)
print(f"3-sinif: ESKI {o3}/{n}  YENI {n3}/{n}")
print(f"2-sinif: ESKI {o2}/{n}  YENI {n2}/{n}")
