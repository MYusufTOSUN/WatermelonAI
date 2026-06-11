from ultralytics import YOLO

model_path = r"c:\Users\tosun\Desktop\watermelon\mrd_yolo_repo\weights\MRD.pt"
print(f"Loading YOLO model from {model_path}...")
model = YOLO(model_path)

print("Exporting to TensorFlow SavedModel format...")
model.export(format="onnx", simplify=False)
print("Export complete.")
