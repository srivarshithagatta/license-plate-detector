from flask import Flask, render_template, request
from ultralytics import YOLO
import easyocr
import cv2
import os

# Windows path for tesseract


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best (1).onnx")
reader = easyocr.Reader(['en'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    results = model(img, imgsz=320, device="cpu", half=False)

    plate_text = "No plate detected"

    for r in results:
        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]

            result = reader.readtext(plate)

            if len(result) > 0:
                plate_text = result[0][1]

    return render_template(
        "index.html",
        plate=plate_text,
        image=file.filename
    )

if __name__ == "__main__":
    app.run(debug=True)