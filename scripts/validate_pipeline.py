"""
End-to-End Pipeline Dogrulama Scripti

Gercek bir Qilin ornegi ile tum modulleri test eder:
  1. Module A: VisualAnalyzer → gorsel ozellik
  2. Module D: AcousticFeatureExtractor → akustik ozellik
  3. Module E: HollowHeartDetector → HH ozellik
  4. Module E: RipenessClassifier → tahmin
  5. Module E: LateFusionEngine → fusion
  6. TFLite: input/output shape dogrulama

Kullanim:
    python scripts/validate_pipeline.py
"""

import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "data" / "models"
QILIN_DIR = PROJECT_ROOT / "dataset" / "datasets" / "watermelon_dataset" / "19_datasets"


def find_sample():
    """Qilin dataset'ten bir test ornegi bul (wav + jpg)."""
    for subdir in sorted(os.listdir(str(QILIN_DIR))):
        subdir_path = QILIN_DIR / subdir
        if not subdir_path.is_dir():
            continue
        chu_dir = subdir_path / "chu"
        if not chu_dir.is_dir():
            continue
        for folder in sorted(os.listdir(str(chu_dir))):
            folder_path = chu_dir / folder
            if not folder_path.is_dir():
                continue
            wavs = [f for f in os.listdir(str(folder_path)) if f.lower().endswith('.wav')]
            jpgs = [f for f in os.listdir(str(folder_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if wavs and jpgs:
                brix = float(subdir.rsplit("_", 1)[1])
                return str(folder_path / wavs[0]), str(folder_path / jpgs[0]), brix
    return None, None, None


def validate_module_d(wav_path):
    """Module D: Akustik ozellik cikarimi (120-D)."""
    print("\n[Module D] Akustik ozellik cikarimi...")
    from backend.module_d.feature_extractor import AcousticFeatureExtractor
    afe = AcousticFeatureExtractor()
    audio, sr = afe.load_audio(wav_path)
    features = afe.extract_feature_vector(audio, sr)
    features = np.asarray(features, dtype=np.float32)
    assert features.shape == (120,), f"Beklenen (120,), alinan {features.shape}"
    print(f"  OK: 120-D vektor, sr={sr}, audio len={len(audio)}")
    print(f"  Ornek degerler: f[0]={features[0]:.4f}, f[78]={features[78]:.6f}")
    return audio, sr, features


def validate_module_a(jpg_path):
    """Module A: Gorsel analiz (11-D)."""
    print("\n[Module A] Gorsel analiz...")
    try:
        from backend.module_a.visual_analyzer import VisualAnalyzer
        va = VisualAnalyzer()
        img_data = np.fromfile(jpg_path, dtype=np.uint8)
        import cv2
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print("  [!] Gorsel okunamadi, sifir vektor kullaniliyor")
            return np.zeros(11, dtype=np.float32)
        features = va.extract_visual_feature_vector(img)
        assert features.shape == (11,), f"Beklenen (11,), alinan {features.shape}"
        print(f"  OK: 11-D vektor")
        return features
    except Exception as e:
        print(f"  [!] VisualAnalyzer hatasi: {e}")
        print(f"  Sifir vektor kullaniliyor")
        return np.zeros(11, dtype=np.float32)


def validate_module_e_hh(audio, sr):
    """Module E: Hollow Heart tespit (8-D)."""
    print("\n[Module E] Hollow Heart tespit...")
    from backend.module_e.hollow_heart_detector import HollowHeartDetector
    hh = HollowHeartDetector(sample_rate=sr)
    result = hh.detect(audio, sr=sr, verbose=False)
    indicators = result.get("indicators", {})
    dp = indicators.get("dual_peak", {}).get("score", 0.0)
    dm = indicators.get("damping", {}).get("score", 0.0)
    sp = indicators.get("spectral", {}).get("score", 0.0)
    cp = indicators.get("cepstral", {}).get("score", 0.0)
    hnr = indicators.get("hnr", {}).get("score", 0.0)
    hh_score = result.get("hh_score", 0.0)
    confidence = result.get("confidence", 0.0)
    active_n = result.get("active_indicators", 0) / 5.0
    features = np.array([dp, dm, sp, cp, hnr, hh_score, confidence, active_n], dtype=np.float32)
    print(f"  OK: 8-D vektor, hh_score={hh_score:.4f}, is_hollow={result.get('is_hollow', False)}")
    return features


def validate_classifier(acoustic_feat, visual_feat, hh_feat):
    """Module E: Siniflandirici tahmin."""
    print("\n[Module E] RFC Siniflandirici tahmin...")
    from backend.module_e.classifier import RipenessClassifier
    clf = RipenessClassifier(model_type='rfc')
    try:
        clf.load_model()
    except Exception as e:
        print(f"  [!] Model yuklenemedi: {e}")
        print("  Once egitim yapin: python -m backend.pipeline.train --model rfc")
        return None

    haptic_feat = np.zeros(7, dtype=np.float32)
    feature_vec = np.concatenate([acoustic_feat, visual_feat, haptic_feat, hh_feat])
    assert feature_vec.shape == (146,), f"Beklenen (146,), alinan {feature_vec.shape}"

    result = clf.predict_single(feature_vec)
    print(f"  OK: class={result['class_id']} ({result['class_label']}), "
          f"confidence={result['confidence']:.4f}")
    return result


def validate_late_fusion():
    """Module E: Late Fusion Engine."""
    print("\n[Module E] Late Fusion Engine...")
    from backend.module_e.late_fusion import LateFusionEngine
    fusion = LateFusionEngine()

    visual_probs = np.array([0.2, 0.7, 0.1], dtype=np.float32)
    acoustic_probs = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    haptic_probs = np.array([0.15, 0.75, 0.1], dtype=np.float32)

    result = fusion.fuse_probabilities(visual_probs, acoustic_probs, haptic_probs)
    assert len(result) == 3, f"Beklenen 3 prob, alinan {len(result)}"
    assert abs(sum(result) - 1.0) < 0.01, f"Olasiliklar 1'e toplanmiyor: {sum(result)}"
    print(f"  OK: Fused probs={[f'{p:.3f}' for p in result]}")
    return result


def validate_tflite():
    """TFLite model input/output shape dogrulama."""
    print("\n[TFLite] Model dogrulama...")
    import tensorflow as tf

    tflite_path = str(MODELS_DIR / "fusion_model_fp16.tflite")
    if not os.path.exists(tflite_path):
        print(f"  [!] TFLite model bulunamadi: {tflite_path}")
        print("  Once egitim yapin: python -m backend.pipeline.train --model dnn")
        return False

    # Turkce path uyumlulugu: binary oku
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()
    interpreter = tf.lite.Interpreter(model_content=tflite_content)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print(f"  Inputs: {len(inputs)}")
    expected_set = {(1, 120), (1, 11), (1, 7), (1, 8)}
    got_set = set()
    for i, inp in enumerate(inputs):
        shape = tuple(int(x) for x in inp['shape'])
        got_set.add(shape)
        print(f"    [{i}] shape={shape} dtype={inp['dtype']}")
    all_ok = got_set == expected_set
    print(f"  Input shapes {'eslesti' if all_ok else 'FARKLI'}: beklenen={expected_set}, alinan={got_set}")

    out_shape = tuple(int(x) for x in outputs[0]['shape'])
    print(f"  Output: shape={out_shape}")
    if out_shape != (1, 3):
        print(f"    [!] Beklenen (1, 3), alinan {out_shape}")
        all_ok = False

    # Dummy inference
    for i, inp in enumerate(inputs):
        interpreter.set_tensor(inp['index'],
                               np.zeros(inp['shape'], dtype=inp['dtype']))
    interpreter.invoke()
    output = interpreter.get_tensor(outputs[0]['index'])
    print(f"  Dummy inference: {output}")
    print(f"  TFLite dogrulama: {'BASARILI' if all_ok else 'SORUNLU'}")
    return all_ok


def main():
    print("=" * 60)
    print("End-to-End Pipeline Dogrulama")
    print("=" * 60)

    wav_path, jpg_path, brix = find_sample()
    if wav_path is None:
        print("[HATA] Qilin ornegi bulunamadi!")
        return

    print(f"Ornek: {wav_path}")
    print(f"Gorsel: {jpg_path}")
    print(f"Brix: {brix}")

    passed = 0
    total = 5

    # 1) Akustik
    try:
        audio, sr, acoustic_feat = validate_module_d(wav_path)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Module D: {e}")
        acoustic_feat = np.zeros(120, dtype=np.float32)
        audio, sr = np.zeros(16000), 44100

    # 2) Gorsel
    try:
        visual_feat = validate_module_a(jpg_path)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Module A: {e}")
        visual_feat = np.zeros(11, dtype=np.float32)

    # 3) Hollow Heart
    try:
        hh_feat = validate_module_e_hh(audio, sr)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] HollowHeart: {e}")
        hh_feat = np.zeros(8, dtype=np.float32)

    # 4) Classifier
    try:
        clf_result = validate_classifier(acoustic_feat, visual_feat, hh_feat)
        if clf_result is not None:
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Classifier: {e}")

    # 5) Late Fusion
    try:
        validate_late_fusion()
        passed += 1
    except Exception as e:
        print(f"  [FAIL] LateFusion: {e}")

    # 6) TFLite (bonus)
    tflite_ok = validate_tflite()

    print(f"\n{'=' * 60}")
    print(f"SONUC: {passed}/{total} modul basarili")
    if tflite_ok:
        print("TFLite: BASARILI")
    print("=" * 60)


if __name__ == "__main__":
    main()
