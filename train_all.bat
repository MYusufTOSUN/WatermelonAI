@echo off
echo ==============================================
echo 1. MRD-YOLO Egitimi Baslatiliyor...
echo ==============================================
cd /d c:\Users\tosun\Desktop\watermelon\mrd_yolo_repo
python train.py

echo.
echo ==============================================
echo 2. Qilin (Watermelon_Dataset) Egitimi Baslatiliyor...
echo ==============================================
cd /d c:\Users\tosun\Desktop\watermelon\dataset\datasets\watermelon_dataset
python main_pytorch.py

echo.
echo TUM EGITIMLER TAMAMLANDI.
pause
