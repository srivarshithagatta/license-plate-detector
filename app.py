from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as rt
import io
import base64
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ONNX model
try:
    MODEL_PATH = "model.onnx"  # Update this path to your model
    session = rt.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logger.info(f"Model loaded successfully")
    logger.info(f"Input shape: {input_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    session = None

EXPECTED_SIZE = 640  # Model expects 640x640

def preprocess_image(image):
    """
    Preprocess image to match model input requirements
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to expected dimensions (640x640)
        image_resized = image.resize((EXPECTED_SIZE, EXPECTED_SIZE), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image_resized)
        
        # Normalize if needed (0-1 range)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension if needed: (H, W, C) -> (1, C, H, W) or (1, H, W, C)
        # Check your model's expected format
        if len(img_array.shape) == 3:
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def run_inference(img_array):
    """
    Run ONNX inference
    """
    try:
        if session is None:
            return None
        
        # Prepare input feed
        input_feed = {input_name: img_array}
        
        # Run inference
        output_names = [output.name for output in session.get_outputs()]
        results = session.run(output_names, input_feed)
        
        return results
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if session is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Check if file is present
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
        
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        original_size = image.size
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Run inference
        results = run_inference(img_array)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Inference failed'
            }), 500
        
        # Convert results to lists for JSON serialization
        processed_results = []
        for result in results:
            if isinstance(result, np.ndarray):
                processed_results.append(result.tolist())
            else:
                processed_results.append(result)
        
        # Encode image for display
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'results': processed_results,
            'original_size': original_size,
            'processed_size': (EXPECTED_SIZE, EXPECTED_SIZE),
            'image': f'data:image/png;base64,{img_base64}',
            'message': f'Successfully processed image. Resized from {original_size} to {EXPECTED_SIZE}x{EXPECTED_SIZE}'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': session is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
