from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model TensorFlow
MODEL_PATH = './model/model_fruits_classification.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Model file not found or failed to load: {MODEL_PATH}. Error: {str(e)}")

# Fungsi untuk memuat dan memproses gambar
def process_image(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(200, 200))  # Sesuaikan ukuran dengan model Anda
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisasi jika model Anda memerlukan input yang dinormalisasi
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image at path: {image_path}. Error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'foto' not in request.files:
            return "No file uploaded", 400
        
        imagefile = request.files['foto']
        if imagefile.filename == '':
            return "No selected file", 400

        # Simpan file dengan nama yang aman
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            imagefile.save(image_path)
        except Exception as e:
            return f"Failed to save file. Error: {str(e)}", 500

        # Proses gambar dan prediksi
        try:
            img_array = process_image(image_path)
            prediction = model.predict(img_array)
        except Exception as e:
            return f"Failed to process image for prediction. Error: {str(e)}", 500

        # Logika klasifikasi berdasarkan output model
        class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('predict.html', prediction=f"Prediction: {predicted_class}", image_path=image_path)

    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
