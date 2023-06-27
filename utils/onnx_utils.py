import os
from typing import Union
import numpy as np
import tensorflow as tf
import onnxruntime


def inference_onnx(
    onnx_path: str, input_data: Union[np.ndarray, tf.Tensor]
) -> np.ndarray:
    """
    :param onnx_path: onnx file name with absolute path
    :param input_data: input data to be inferenced with onnx model
    :return: inference result of onnx model
    """
    ort_model = onnxruntime.InferenceSession(onnx_path)
    result = ort_model.run(None, {"input_1": input_data})[0]

    return result


def convert_tf2onnx(model, fname: str, opset: int = 13):
    """
    :param model: tf or keras model
    :param fname: onnx file name with absolute path
    :param opset: onnx opset, default version is 13
    """
    command = f"python -m tf2onnx.convert --saved-model {model} --output {fname} --opset {str(opset)}"
    os.system(command)
