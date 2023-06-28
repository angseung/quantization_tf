from typing import Union, Tuple, Optional
import tensorflow as tf
from keras.engine.functional import Functional
import numpy as np
from utils.tflite_utils import inference as inference_with_tflite


def cal_mse(
    pred: np.ndarray, target: Union[np.ndarray, tf.Tensor], norm: bool = False
) -> np.ndarray:
    pred = tf.convert_to_tensor(pred)
    if isinstance(target, np.ndarray):
        target = tf.convert_to_tensor(target)
    mse_loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE
    )

    if norm:
        mse = mse_loss(target, pred) / mse_loss(target, tf.zeros_like(target))

    else:
        mse = mse_loss(target, pred)

    return mse.numpy()


class RepresentativeDataset:
    """
    Calibration data loader class for test only.
    """

    def __init__(
        self, shape: Union[Tuple[int, int, int, int], Tuple[int, int, int], int]
    ):
        if isinstance(shape, tuple):
            self.shape = shape
        elif isinstance(shape, int):
            self.shape = (1, shape, shape, 3)

    def representative_dataset(self):
        for _ in range(10):
            data = np.random.rand(*self.shape)  # Calibration Data
            yield [data.astype(np.float32)]  # Return data with generator


class TFModelQuantizer:
    def __init__(
        self,
        model: Functional,
        dynamic: bool,
        input_shape: Optional[Tuple[int, int, int]],
        fully_quant: Optional[bool] = False,
    ):
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]

        input_shape_from_model = model.input_shape

        # model has no top
        if None in input_shape_from_model[1:]:
            self.converter.input_shape = (None, *input_shape[1:])

        # model has top
        else:
            self.converter.input_shape = input_shape_from_model

        # Integer only quantization where IO is also int
        if fully_quant:
            self.converter.inference_input_type = tf.int8
            self.converter.inference_output_type = tf.int8

        if not dynamic:
            self.converter.representative_dataset = RepresentativeDataset(
                shape=self.converter.input_shape[2]
            ).representative_dataset

        self.quantized_model = self.converter.convert()

    def inference(self, input_tensor: tf.Tensor, show_latency: bool) -> np.ndarray:
        return inference_with_tflite(
            self.quantized_model,
            input_tensor=input_tensor,
            show_elapsed_time=show_latency,
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.quantized_model)
