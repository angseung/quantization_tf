import time
from typing import Union, Tuple
from pathlib import Path
import numpy as np
import tensorflow as tf

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory


def inference(
    tf_lite_model: Union[str, bytes],
    input_tensor: tf.Tensor,
    show_elapsed_time: bool = False,
) -> tf.Tensor:
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


class RepresentativeDataset:
    """
    Calibration data loader class for test only.
    """

    def __init__(self, shape: Union[Tuple[int, int], int]):
        if isinstance(shape, tuple):
            self.shape = (1, *shape, 3)
        elif isinstance(shape, int):
            self.shape = (1, shape, shape, 3)

    def representative_dataset(self):
        for _ in range(10):
            data = np.random.rand(*self.shape)  # Calibration Data
            yield [data.astype(np.float32)]  # Return data with generator
