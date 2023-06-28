import os
import time
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import tflite2onnx
from utils.quantization_utils import cal_mse, TFModelQuantizer
from utils.onnx_utils import convert_tf2onnx, inference_onnx

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory
target_dir = "tflite"
models = [
    keras.applications.DenseNet121(),
    keras.applications.DenseNet169(),
    keras.applications.DenseNet201(),
    keras.applications.ResNet50(),
    keras.applications.ResNet50V2(),
    keras.applications.ResNet101(),
    keras.applications.ResNet152(),
    keras.applications.MobileNet(),
    keras.applications.MobileNetV2(),
    keras.applications.MobileNetV3Small(),
    keras.applications.MobileNetV3Large(),
    keras.applications.EfficientNetB0(),
    keras.applications.EfficientNetB1(),
    keras.applications.EfficientNetB2(),
    keras.applications.EfficientNetB3(),
    keras.applications.EfficientNetB4(),
    keras.applications.EfficientNetB5(),
    keras.applications.EfficientNetB6(),
    keras.applications.EfficientNetB7(),
    keras.applications.EfficientNetV2B0(),
    keras.applications.EfficientNetV2B1(),
    keras.applications.EfficientNetV2B2(),
    keras.applications.EfficientNetV2S(),
    keras.applications.EfficientNetV2M(),
    keras.applications.EfficientNetV2L(),
]

for model in models:
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    input_shape = (1, *model.input_shape[1:])
    input_tensor = tf.random.uniform(input_shape)
    model_path = os.path.join(ROOT, "weights", f"{model.name}")
    model.save(model_path)

    start_fp = time.time()
    fp_output = model(input_tensor).numpy()
    latency_fp = time.time() - start_fp
    print(f"fp32 model: {latency_fp: .4f}")

    quantized_model = TFModelQuantizer(
        model, dynamic=False, input_shape=input_shape[1:], fully_quant=False
    )
    pred_q = quantized_model.inference(input_tensor, True)

    if not os.path.isdir(os.path.join(ROOT, target_dir)):
        os.makedirs(os.path.join(ROOT, target_dir))

    quantized_model.save(os.path.join(ROOT, target_dir, f"quant_{model.name}.tflite"))

    onnx_file = os.path.join(ROOT, "onnx", f"{model.name}.onnx")
    onnx_quant_file = os.path.join(ROOT, "onnx", f"{model.name}_quant.onnx")

    # TODO: Modify quantization config from per-channel to per-tensor.
    #  TFLite2onnx does not support per-channel method yet.
    # tflite2onnx.convert(
    #     os.path.join(ROOT, target_dir, "test_model.tflite"), onnx_quant_file
    # )

    mse = cal_mse(pred_q, fp_output)
    convert_tf2onnx(model_path, onnx_file, 13)
    input_name = [n.name for n in model.inputs][0]
    onnx_pred = inference_onnx(onnx_file, input_name, input_tensor.numpy())
    mse_onnx = cal_mse(onnx_pred, fp_output)

    with open("./log.txt", "a") as f:
        f.write(
            f"[{model.name}] mse fp-quant: {mse[0]: .6f}, mse fp-onnx: {mse_onnx[0]: .6f}\n"
        )
