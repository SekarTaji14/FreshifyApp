from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter  # Import tflite-runtime

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model TFLite
MODEL_PATH = './model/model_fruits_classification.tflite'

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Ambil indeks input dan output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise FileNotFoundError(f"TFLite model file not found or failed to load: {MODEL_PATH}. Error: {str(e)}")

# Fungsi untuk memuat dan memproses gambar
def process_image(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path).resize((200, 200))  # Sesuaikan ukuran dengan model Anda
        img_array = np.asarray(img).astype('float32') / 255.0  # Normalisasi jika diperlukan
        img_array = np.expand_dims(img_array, axis=0)
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
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
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
