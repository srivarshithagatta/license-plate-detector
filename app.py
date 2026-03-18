from flask import Flask, render_template, request
from ultralytics import YOLO
import easyocr
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best (1).onnx")

# Load OCR
reader = easyocr.Reader(['en'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)

        if img is None:
            return "Image load error"

        results = model(img)   # 🔥 IMPORTANT: no imgsz here

        plate_text = "No plate detected"

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate = img[y1:y2, x1:x2]

                # OCR
                ocr_result = reader.readtext(plate)

                if len(ocr_result) > 0:
                    plate_text = ocr_result[0][1]

        return render_template(
            "index.html",
            plate=plate_text,
            image=file.filename
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)