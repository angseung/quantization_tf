import time
import tensorflow as tf
from tensorflow import keras
from utils import inference, representative_dataset


input_tensor = tf.random.uniform((1, 224, 224, 3))
model = keras.applications.ResNet50()

start_fp = time.time()
fp_output = model(input_tensor).numpy()
latency_fp = time.time() - start_fp
print(f"fp32 model: {latency_fp: .4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization flag
converter.representative_dataset = representative_dataset
tflite_model_quant = converter.convert()

with open("tflite/test_model.tflite", "wb") as f:
    f.write(tflite_model_quant)

pred_q = inference("tflite/test_model.tflite", input_tensor)
