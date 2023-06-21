import os
import time
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from utils import inference, RepresentativeDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory
target_dir = "tflite"

input_shape = (1, 224, 224, 3)
input_tensor = tf.random.uniform(input_shape)
model = keras.applications.ResNet50()

start_fp = time.time()
fp_output = model(input_tensor).numpy()
latency_fp = time.time() - start_fp
print(f"fp32 model: {latency_fp: .4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization flag
converter.representative_dataset = RepresentativeDataset(
    shape=input_shape[2]
).representative_dataset
tflite_model_quant = converter.convert()

if not os.path.isdir(os.path.join(ROOT, target_dir)):
    os.makedirs(os.path.join(ROOT, target_dir))

with open(os.path.join(ROOT, target_dir, "test_model.tflite"), "wb") as f:
    f.write(tflite_model_quant)

pred_q = inference(
    os.path.join(ROOT, target_dir, "test_model.tflite"), input_tensor, True
)
