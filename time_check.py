import tensorflow_model_optimization as tfmot
import tempfile
import cv2
from time import time

import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.applications import ResNet50
from keras.models import load_model


def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

def infer_time_h5(file, img):
    model = tf.keras.models.load_model(file)
    # model.summary()
    start = time()
    res = model.predict(img)
    end = time()
    infer_t = end - start
    
    return infer_t

def infer_time_tflite(file, img):
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], img)
    
    start = time()
    interpreter.invoke()
    end = time()
    infer_t = end - start
    
    return infer_t



keras_file = './models/tmpvyqm5ppw.h5'
pruned_keras_file = './models/tmpcwxqwynj.h5'
pruned_tflite_file = './models/tmp0qkqvwcf.tflite'
pruned_quant_tflite_file = './models/tmpoa9jdqwp.tflite'

img = cv2.imread('test.jpg')
img = np.float32(img)
img = cv2.resize(img, (32, 32))
img = img[np.newaxis, ...]


keras_file_time = 0
for i in range(10):
    keras_file_time += infer_time_h5(keras_file, img)

pruned_keras_file_time = 0
for i in range(10):
    pruned_keras_file_time += infer_time_h5(pruned_keras_file, img)
    
pruned_tflite_file_time = 0
for i in range(10):
    pruned_tflite_file_time += infer_time_tflite(pruned_tflite_file, img)
    
pruned_quant_tflite_file_time = 0
for i in range(10):
    pruned_quant_tflite_file_time += infer_time_tflite(pruned_quant_tflite_file, img)
    
    
print(keras_file_time/10, pruned_keras_file_time/10, pruned_tflite_file_time/10, pruned_quant_tflite_file_time/10)
print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_quant_tflite_file)))