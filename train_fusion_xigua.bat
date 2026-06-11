@echo off
setlocal

echo ============================================
echo  Karpuz Fusion Modeli Egitimi (XIGUA Transfer)
echo  Dataset: Qilin 19_datasets
echo  Model:   KNN + RFC + DNN
echo  Feature: 146-D + Xigua 64-D embedding = 210-D
echo ============================================

cd /d c:\Users\tosun\Desktop\watermelon

REM Xigua varyantinda farkli feature dim oldugu icin cache'i temizle
if exist "data\processed\X_features.npy" (
    echo [0/3] Eski islenmis veriler temizleniyor...
    del /Q "data\processed\X_features.npy" 2>nul
    del /Q "data\processed\y_labels.npy" 2>nul
    del /Q "data\processed\groups.npy" 2>nul
)

echo.
echo [1/3] Feature extraction + KNN/RFC/DNN egitimi (Xigua transfer)...
echo       Multimodal: ses + gorsel + HH + Xigua64 (210-D)
echo       Augment: 2x (time-stretch + pitch-shift + noise)
echo       Split: random
echo       LOWO: aktif (gercek dogruluk)
python -m backend.pipeline.train --model all --split random --n-augment 2 --xigua --lowo
if errorlevel 1 (
    echo.
    echo [HATA] Egitim basarisiz! Loglari kontrol edin.
    pause
    exit /b 1
)

echo.
echo [2/3] TFLite modeli Flutter assets'e kopyalaniyor...
if not exist "flutter_app\assets\models" mkdir "flutter_app\assets\models"
if exist "data\models\fusion_model_fp16.tflite" (
    copy /Y "data\models\fusion_model_fp16.tflite" "flutter_app\assets\models\" >nul
    echo       fusion_model_fp16.tflite kopyalandi.
)

echo.
echo [3/3] Egitim tamamlandi!
echo ============================================
echo   Sonuclar:  data\models\training_results.json
echo   Modeller:  data\models\*.joblib, *.tflite
echo   NOT: DNN dali xigua-64'u kullanmaz, sadece KNN/RFC kullanir.
echo ============================================
pause
endlocal
