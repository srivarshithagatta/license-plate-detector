from flask import Flask, render_template, request
from ultralytics import YOLO
import pytesseract
import cv2
import os

# Windows path for tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best (1).pt")

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

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_string(gray, config="--psm 7")

            if text.strip() != "":
                plate_text = text.strip()

    return render_template(
        "index.html",
        plate=plate_text,
        image=file.filename
    )

if __name__ == "__main__":
    app.run(debug=True)