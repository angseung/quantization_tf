import time
from typing import Union
from pathlib import Path
import tensorflow as tf
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory


def inference(
    tf_lite_model: Union[str, bytes],
    input_tensor: tf.Tensor,
    show_elapsed_time: bool = False,
) -> np.ndarray:
    if isinstance(tf_lite_model, str):
        tf_lite_model = tf.lite.Interpreter(model_path=tf_lite_model)
    elif isinstance(tf_lite_model, bytes):
        tf_lite_model = tf.lite.Interpreter(model_content=tf_lite_model)

    tf_lite_model.allocate_tensors()
    input_index = tf_lite_model.get_input_details()[0]["index"]
    output_index = tf_lite_model.get_output_details()[0]["index"]

    tf_lite_model.set_tensor(input_index, input_tensor)

    # Run inference.
    start_qint8 = time.time()
    tf_lite_model.invoke()
    latency_qint8 = time.time() - start_qint8

    if show_elapsed_time:
        print(f"qint8 model: {latency_qint8: .4f}")

    return tf_lite_model.get_tensor(output_index)
