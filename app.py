from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as rt
import cv2
import io
import base64
import os
import logging
from pathlib import Path
from threading import Thread
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "best (1).onnx"
CONFIDENCE_THRESHOLD = 0.5
EXPECTED_SIZE = 640

# Global session variable
session = None
input_name = None
output_names = None
model_loaded = False

def load_model():
    """Load ONNX model"""
    global session, input_name, output_names, model_loaded
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            model_loaded = False
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        session = rt.InferenceSession(
            MODEL_PATH, 
            providers=['CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Input: {input_name}")
        logger.info(f"  Outputs: {output_names}")
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        model_loaded = False
        return False

def preprocess_image(image):
    """Preprocess image for YOLO model inference"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize to 640x640
        img_resized = cv2.resize(
            img_array, 
            (EXPECTED_SIZE, EXPECTED_SIZE), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to 0-1 range
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert HWC to NCHW format
        img_nchw = np.transpose(img_normalized, (2, 0, 1))
        img_nchw = np.expand_dims(img_nchw, axis=0)
        
        return img_nchw, img_resized
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def postprocess_detections(outputs, conf_threshold=CONFIDENCE_THRESHOLD):
    """Postprocess YOLO model outputs"""
    try:
        detections = []
        
        if len(outputs) == 0:
            return detections
        
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]
        
        for detection in output:
            x, y, w, h = detection[:4]
            conf = detection[4]
            
            if conf > conf_threshold:
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                detection_data = {
                    'confidence': float(conf),
                    'coordinates': {
                        'x1': max(0, x1),
                        'y1': max(0, y1),
                        'x2': x2,
                        'y2': y2,
                        'width': int(w),
                        'height': int(h)
                    }
                }
                detections.append(detection_data)
        
        return detections
    except Exception as e:
        logger.error(f"Error postprocessing detections: {e}")
        return []

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    try:
        img_with_boxes = image.copy()
        
        for detection in detections:
            coords = detection['coordinates']
            conf = detection['confidence']
            
            cv2.rectangle(
                img_with_boxes,
                (coords['x1'], coords['y1']),
                (coords['x2'], coords['y2']),
                (0, 255, 0),
                2
            )
            
            label = f"LP: {conf:.2f}"
            cv2.putText(
                img_with_boxes,
                label,
                (coords['x1'], coords['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return img_with_boxes
    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        return image

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return "Error loading page", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict license plates in uploaded image"""
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read and open image
        try:
            image = Image.open(io.BytesIO(file.read()))
            original_size = image.size
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400
        
        # Preprocess image
        img_array, img_resized = preprocess_image(image)
        
        # Run inference
        input_feed = {input_name: img_array}
        outputs = session.run(output_names, input_feed)
        
        # Postprocess detections
        detections = postprocess_detections(outputs)
        
        # Draw detections on image
        img_with_boxes = draw_detections(img_resized, detections)
        
        # Convert processed image to base64
        img_pil = Image.fromarray(img_with_boxes)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': f'Detected {len(detections)} license plate(s)',
            'detections_count': len(detections),
            'results': detections,
            'original_size': list(original_size),
            'processed_size': [EXPECTED_SIZE, EXPECTED_SIZE],
            'image': f'data:image/png;base64,{img_base64}'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in predict: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    }), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting License Plate Detector")
    logger.info("=" * 60)
    
    if load_model():
        logger.info("✓ Ready to process images!")
    else:
        logger.warning("⚠ Model not loaded. Predictions will fail.")
    
    app.run(debug=False, host='0.0.0.0', port=10000, threaded=True)
