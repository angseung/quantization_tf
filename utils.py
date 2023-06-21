from typing import Union
import numpy as np
import tensorflow as tf


def inference(
    model_path: str, input_tensor: Union[tf.Tensor, np.ndarray]
) -> tf.Tensor:
    tflite_model = tf.lite.Interpreter(model_path)
    tflite_model.allocate_tensors()
    input_index = tflite_model.get_input_details()[0]["index"]
    output_index = tflite_model.get_output_details()[0]["index"]

    tflite_model.set_tensor(input_index, input_tensor)

    # Run inference.
    tflite_model.invoke()

    return tflite_model.get_tensor(output_index)
