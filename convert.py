import tensorflow as tf

model = tf.keras.models.load_model('./model/model_fruits_classification.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('./model/model_fruits_classification.tflite', 'wb') as f:
    f.write(tflite_model)
