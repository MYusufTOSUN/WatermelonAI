from ultralytics import YOLO
import os

model_path = r"c:\Users\tosun\Desktop\watermelon\mrd_yolo_repo\weights\MRD.pt"
print(f"Loading YOLO model from {model_path}...")
model = YOLO(model_path)

print("Exporting to TFLite format...")
# Exporting for Edge AI devices efficiently (int8 or fp16) - by default it uses fp32 or fp16 depending on the system
model.export(format="tflite", optimize=True, simplify=False)
print("Export complete.")
