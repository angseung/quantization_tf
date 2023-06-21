import tensorflow as tf
from tensorflow import keras
from utils import inference


input_tensor = tf.random.uniform((1, 224, 224, 3))
model = tf.keras.applications.ResNet50()

fp_output = model(input_tensor).numpy()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization flag
tflite_model_quant = converter.convert()

with open("tflite/test_model.tflite", "wb") as f:
    f.write(tflite_model_quant)

pred_q = inference("tflite/test_model.tflite", input_tensor)
