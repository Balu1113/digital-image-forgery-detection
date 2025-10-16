from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from model_utils import preprocess_image, predict_forgery
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPEG or PNG image.'}), 400
    
    filepath = None
    ela_image_path = None
    
    try:
        if not file.filename:
            return jsonify({'error': 'Invalid filename'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        ela_image_path, prediction, confidence = predict_forgery(filepath)
        
        with open(filepath, 'rb') as f:
            original_image_data = base64.b64encode(f.read()).decode('utf-8')
        
        with open(ela_image_path, 'rb') as f:
            ela_image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'original_image': original_image_data,
            'ela_image': ela_image_data
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        if ela_image_path and os.path.exists(ela_image_path):
            os.remove(ela_image_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
