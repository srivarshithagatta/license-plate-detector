from ultralytics import YOLO

# Load trained model
model = YOLO("best (1).pt")

# Export to ONNX
model.export(format="onnx")

print("Model converted to ONNX successfully!")