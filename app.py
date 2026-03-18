from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as rt
import cv2
import io
import base64
import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'best (1).onnx')
CONFIDENCE_THRESHOLD = 0.5
EXPECTED_SIZE = 640
MODEL_TIMEOUT = 30

# Global variables
session = None
input_name = None
output_names = None
model_loaded = False

def load_model():
    """Load ONNX model"""
    global session, input_name, output_names, model_loaded
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            model_loaded = False
            return False
        
        logger.info(f"Model file size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f}MB")
        
        # Load with CPU provider only
        session = rt.InferenceSession(
            MODEL_PATH,
            providers=['CPUExecutionProvider'],
            sess_options=rt.SessionOptions()
        )
        
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Input: {input_name}")
        logger.info(f"  Output shapes: {[o.shape for o in session.get_outputs()]}")
        
        model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {str(e)}")
        model_loaded = False
        return False

def preprocess_image(image):
    """Preprocess image for YOLO"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure 3 channels
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] != 3:
            img_array = img_array[:, :, :3]
        
        # Resize to 640x640
        img_resized = cv2.resize(
            img_array,
            (EXPECTED_SIZE, EXPECTED_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to 0-1
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # HWC to NCHW
        img_nchw = np.transpose(img_normalized, (2, 0, 1))
        img_nchw = np.expand_dims(img_nchw, axis=0)
        
        return img_nchw, img_resized
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

def postprocess_detections(outputs, conf_threshold=CONFIDENCE_THRESHOLD):
    """Postprocess YOLO outputs"""
    try:
        detections = []
        
        if not outputs or len(outputs) == 0:
            return detections
        
        output = outputs[0]
        
        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]
        
        if output.shape[0] == 0:
            return detections
        
        # Process each detection
        for detection in output:
            if len(detection) < 5:
                continue
            
            x, y, w, h = detection[:4]
            conf = float(detection[4])
            
            # Filter by confidence
            if conf < conf_threshold:
                continue
            
            # Convert center coords to corner coords
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            detections.append({
                'confidence': conf,
                'coordinates': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': int(w),
                    'height': int(h)
                }
            })
        
        return detections
        
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        return []

def draw_detections(image, detections):
    """Draw boxes on image"""
    try:
        img = image.copy()
        
        for det in detections:
            coords = det['coordinates']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                img,
                (coords['x1'], coords['y1']),
                (coords['x2'], coords['y2']),
                (0, 255, 0),
                2
            )
            
            # Draw text
            label = f"{conf:.2%}"
            cv2.putText(
                img,
                label,
                (coords['x1'], max(20, coords['y1'] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return img
        
    except Exception as e:
        logger.error(f"Drawing error: {str(e)}")
        return image

@app.route('/')
def index():
    """Serve HTML"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {str(e)}")
        return "Error loading page", 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict license plates"""
    try:
        # Validate model
        if not model_loaded or session is None:
            logger.warning("Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        # Validate input
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read image
        try:
            image = Image.open(io.BytesIO(file.read()))
            original_size = image.size
            logger.info(f"Image loaded: {original_size}")
        except Exception as e:
            logger.error(f"Image read error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Invalid image'
            }), 400
        
        # Preprocess
        try:
            img_array, img_resized = preprocess_image(image)
            logger.info(f"Image preprocessed: {img_array.shape}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Processing failed'
            }), 500
        
        # Inference
        try:
            input_feed = {input_name: img_array}
            outputs = session.run(output_names, input_feed)
            logger.info(f"Inference complete: {len(outputs)} outputs")
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Inference failed'
            }), 500
        
        # Postprocess
        detections = postprocess_detections(outputs)
        logger.info(f"Detections: {len(detections)}")
        
        # Draw results
        img_with_boxes = draw_detections(img_resized, detections)
        
        # Encode image
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {str(e)}")
            img_b64 = None
        
        # Response
        return jsonify({
            'success': True,
            'message': f'Found {len(detections)} license plate(s)',
            'detections_count': len(detections),
            'results': detections,
            'original_size': list(original_size),
            'processed_size': [EXPECTED_SIZE, EXPECTED_SIZE],
            'image': f'data:image/png;base64,{img_b64}' if img_b64 else None
        }), 200
        
    except Exception as e:
        logger.error(f"Predict error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Server error'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'success': False, 'error': 'Server error'}), 500

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("License Plate Detector Starting")
    logger.info("="*60)
    
    load_model()
    
    port = int(os.getenv('PORT', 10000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
