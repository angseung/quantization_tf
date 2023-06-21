from typing import *
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter


def inference(tflite_model: interpreter, input_tensor: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    input_index = tflite_model.get_input_details()[0]["index"]
    output_index = tflite_model.get_output_details()[0]["index"]

    tflite_model.set_tensor(input_index, input_tensor)

    # Run inference.
    tflite_model.invoke()

    output = tflite_model.tensor(output_index)

    return output
