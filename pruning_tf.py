import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.applications import ResNet50

# %load_ext tensorboard
# Load MNIST dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# expand new axis, channel axis 
x_train = np.expand_dims(x_train, axis=-1)

# [optional]: we may need 3 channel (instead of 1)
x_train = np.repeat(x_train, 3, axis=-1)

# it's always better to normalize 
x_train = x_train.astype('float32') / 255

# resize the input shape , i.e. old shape: 28, new shape: 32
x_train = tf.image.resize(x_train, [32,32]) # if we want to resize 

# one hot 
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)

print(x_train.shape, y_train.shape)

# expand new axis, channel axis 
x_test = np.expand_dims(x_test, axis=-1)

# [optional]: we may need 3 channel (instead of 1)
x_test = np.repeat(x_test, 3, axis=-1)

# it's always better to normalize 
x_test = x_test.astype('float32') / 255

# resize the input shape , i.e. old shape: 28, new shape: 32
x_test = tf.image.resize(x_test, [32,32]) # if we want to resize 

# one hot 
y_test = tf.keras.utils.to_categorical(y_test , num_classes=10)

print(x_test.shape, y_test.shape)

# Define the model architecture.
input = tf.keras.Input(shape=(32,32,3))
model = tf.keras.applications.ResNet50(weights='imagenet', include_top = False, input_tensor = input)

# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(model.output)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)(gap)

# bind all
resnet = tf.keras.Model(model.input, output)

# Train the digit classification model
resnet.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

resnet.fit(
  x_train,
  y_train,
  epochs=4,
  validation_split=0.1,
)

_, baseline_model_accuracy = resnet.evaluate(
    x_test, y_test, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(resnet, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


num_images = x_train.shape[0] * (1 - 0.1)
end_step = np.ceil(num_images / 128).astype(np.int32) * 2

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(resnet, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

print(model_for_pruning.summary())

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(
    x_train,
    y_train,
    epochs=2,
    validation_split=0.1,
    batch_size = 128,
    callbacks=callbacks
)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)
