from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from flask_cors import CORS, cross_origin
from flask import Flask, redirect, url_for, request, render_template, jsonify

from werkzeug.utils import secure_filename
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = './uploads'
MODEL_PATH = './Melanoma.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_array = img_array/255.0

        prediction = model.predict(img_array)
        print(prediction[0][0] < 0.00000000000005)
        print(prediction[0][0] < .05)
        
        if prediction[0][0] < .05:
            prediction_result = "The Person is not Infected with Melanoma"
        else:
            prediction_result = "The Person is Infected with Melanoma"
        return jsonify({'prediction': prediction_result, 'accuracy': str(prediction[0][0])}), 200

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.run(port=5001, debug=True)