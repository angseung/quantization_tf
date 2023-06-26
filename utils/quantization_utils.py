from typing import Union, Tuple
import tensorflow as tf
import numpy as np


def cal_mse(
    pred: np.ndarray, target: Union[np.ndarray, tf.Tensor], norm: bool = False
) -> np.ndarray:
    pred = tf.convert_to_tensor(pred)
    if isinstance(target, np.ndarray):
        target = tf.convert_to_tensor(target)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

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
