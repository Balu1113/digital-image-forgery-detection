import numpy as np
from PIL import Image, ImageEnhance, ImageChops
import tensorflow as tf
from tensorflow.keras import models
import os
import tempfile

MODEL_PATH =r"C:\Users\talat\OneDrive\Documents\frontend-ifd\attached_assets\image_real_or_fake.keras"
model = None

def load_model():
    global model
    if model is None:
        model = models.load_model(MODEL_PATH)
    return model

def image_to_ela(image_path, quality=90):
    ext_lower = image_path.lower()
    if not (ext_lower.endswith('.jpg') or ext_lower.endswith('.jpeg') or ext_lower.endswith('.png')):
        raise ValueError(f'Unsupported file extension for ELA: {image_path}')
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            resaved_path = temp_file.name
        
        try:
            image.save(resaved_path, 'JPEG', quality=quality)
            resaved = Image.open(resaved_path)
            
            ela_image = ImageChops.difference(image, resaved)
            
            band_values = ela_image.getextrema()
            max_value = max([val[1] for val in band_values])
            
            if max_value == 0:
                max_value = 1
            
            scale = 255.0 / max_value
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            return ela_image
        finally:
            if os.path.exists(resaved_path):
                os.remove(resaved_path)
    except Exception as e:
        raise Exception(f'Could not convert image to ELA: {str(e)}')

def preprocess_image(image_path, target_size=(224, 224)):
    ela_image = image_to_ela(image_path)
    
    ela_image = ela_image.resize(target_size)
    
    ela_array = np.array(ela_image)
    
    ela_array = ela_array.astype('float32') / 255.0
    
    ela_array = np.expand_dims(ela_array, axis=0)
    
    ela_save_path = image_path.replace('.', '_ela.')
    ela_image.save(ela_save_path, 'JPEG')
    
    return ela_array, ela_save_path

def predict_forgery(image_path):
    model = load_model()
    
    preprocessed_image, ela_image_path = preprocess_image(image_path)
    
    prediction = model.predict(preprocessed_image, verbose=0)
    
    confidence = prediction[0][0]
    
    if confidence >= 0.65:
        label = "Doctored"
        confidence_percent = confidence * 100
    else:
        label = "Real"
        confidence_percent = (1 - confidence) * 100
    
    return ela_image_path, label, confidence_percent
