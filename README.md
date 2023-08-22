# Optimizing Models with Tensorflow

# Dynamic Quantization

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()  # it quantizes ONLY weights
```

- 변환 단계에서는 Weights만 8비트 정수형으로 양자화한다.
- 실행 단계에서는 Weights와 Activation의 범위를 기준으로 동적으로 8비트 정수형으로 양자화한다.
    - 동적으로 양자화 범위를 설정하므로, 입출력 데이터에 해당하는 Activation 양자화는 실행 단계에서 수행할 수 밖에 없다.

# Full Integer Quantization

```python
import tensorflow as tf

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)  # Calibration Data
      yield [data.astype(np.float32)]  # Return data with generator

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset  # Dataset for Calibration
tflite_quant_model = converter.convert()
```

- *representative_dataset* Attribute를 지정해주면 Fully Quantized Model로 변환 가능하다.
- PyTorch의 PTQ와 마찬가지로, 양자화 한 모델의 입출력 자료형은 float32이므로 양자화 하지 않은 모델과 동등하게 사용할 수 있다.
    - 따라서 Floating Point Unit이 없는 하드웨어에서는 사용할 수 없다.

## Integer ONLY Quantization

- 이 변환 방식은 INT Processing Unit만 존재하는 하드웨어를 위한 것이다.
    - Requires Tensorflow ≥ 2.3.0

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
```

# Quantization Aware Training

```python
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Quantization aware training
q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

# Quantized QAT-applied model.
# It does not require representitive_dataset attribute, unlike PTQ
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
```

# Quantized TFLite Model

- Static input shape

```python
tf_lite_model.get_input_details()[0]
>>>

{'name': 'serving_default_input_1:0',
 'index': 0,
 'shape': array([1, 224, 224, 3]),  # This can not be modified
 'shape_signature': array([-1, -1, -1,  3]),
 'dtype': numpy.float32,
 'quantization': (0.0, 0),
 'quantization_parameters': {'scales': array([], dtype=float32),
 'zero_points': array([], dtype=int32),`
 'quantized_dimension': 0},
 'sparsity_parameters': {}
}
```

# Model Quantization Results

| Model | Quantization Support | FP32 model ONNX Export Support | Quantized model ONNX Export Support | ONNX opset |
| --- | --- | --- | --- | --- |
| ResNet50 | PTQ, QAT | Y | N | 13 |
| ResNet50V2 | PTQ, QAT | Y | N | 13 |
| ResNet101 | PTQ, QAT | Y | N | 13 |
| ResNet152 | PTQ, QAT | Y | N | 13 |
| DenseNet121 | PTQ, QAT | Y | N | 13 |
| DenseNet169 | PTQ, QAT | Y | N | 13 |
| DenseNet201 | PTQ, QAT | Y | N | 13 |
| EfficientNetB0 | PTQ, QAT | Y | N | 13 |
| EfficientNetB1 | PTQ, QAT | Y | N | 13 |
| EfficientNetB2 | PTQ, QAT | Y | N | 13 |
| EfficientNetB3 | PTQ, QAT | Y | N | 13 |
| EfficientNetB4 | PTQ, QAT | Y | N | 13 |
| EfficientNetB5 | PTQ, QAT | Y | N | 13 |
| EfficientNetB6 | PTQ, QAT | Y | N | 13 |
| EfficientNetB7 | PTQ, QAT | Y | N | 13 |
| EfficientNetV2B0 | PTQ, QAT | Y | N | 13 |
| EfficientNetV2B1 | PTQ, QAT | Y | N | 13 |
| EfficientNetV2B2 | PTQ, QAT | Y | N | 13 |
| EfficientNetV2S | PTQ, QAT | Y | N | 13 |
| EfficientNetV2M | PTQ, QAT | Y | N | 13 |
| EfficientNetV2L | PTQ, QAT | Y | N | 13 |
| MobileNet | PTQ, QAT | Y | N | 13 |
| MobileNetV2 | PTQ, QAT | Y | N | 13 |
| MobileNetV3S | N/A | Y | N/A | N/A |
| MobileNetV3L | N/A | Y | N/A | N/A |
- MobileNetV3: input_size error in tf2onnx (expected (1, 224, 224, 3), but got (1, 1, 1, 3))
- EfficientNet계열: requires tensorflow ≤ 2.9.3에서만 ONNX Export 가능

# Pruning

- with Keras Method

```python
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define model for pruning. This uses a method of Keras API.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model again for pruning the model
# This model can be quantized with TFLite convertor
model_for_pruning.fit(train_images, train_labels,
              batch_size=batch_size, epochs=epochs)
```

- with XNNPACK

```python
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define parameters for pruning. This uses a method of XNNPACK.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                               final_sparsity=0.75,
                                                               begin_step=0,
                                                               end_step=end_step),
      'pruning_policy': tfmot.sparsity.keras.PruneForLatencyOnXNNPack()
}

# next step is same with above.
```

- XNNPACK에서 추가 실험 진행 필요 (대략 30%정도의 추론시간 개선 효과가 있다고 함)
- 다음 표는 Keras의 Pruning 기능으로 최적화 한 ResNet50 모델의 벤치마크

| ResNet50 | Latency (s) | Latency Ratio | Model Size (MB) | Size Ratio |
| --- | --- | --- | --- | --- |
| original | 0.4734 | 1 | 87.79 | 1 |
| pruning | 0.4540 | 0.959 | 28.05 | 0.319 |
| pruning and quant | 0.0017 | 0.003 | 27.76 | 0.316 |

# Inference with TFLite Model

- 추론 Batch Size ≠ 1인 데이터 처리 불가

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter("tf_lite_model.tflite")
interpreter.allocate_tensors()  # load tflite model on memory

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, input_tensor)  # set input
interpreter.invoke()  # forward

pred = interpreter.get_tensor(output_index)  # get inference result
```

# Wrapper Class for Model Quantization

```python
class TFModelQuantizer:
    def __init__(
        self,
        model: Functional,
        dynamic: bool,
        input_shape: Optional[Tuple[int, int, int]],
        fully_quant: Optional[bool] = False,
    ):
        self._init_converter(model, dynamic, input_shape, fully_quant)

    def _init_converter(
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

# Usage
model = keras.applications.ResNet50()
input_shape = (1, *model.input_shape[1:])
quantized_model = TFModelQuantizer(
        model, dynamic=False, input_shape=input_shape[1:], fully_quant=False
    )
pred_q = quantized_model.inference(input_tensor, True)
quantized_model.save(os.path.join(ROOT, target_dir, f"quant_{model.name}.tflite"))
```

# Latency Benchmarks

| Model | FP32 Latency (s) | Quant Latency (s) | Ratio |
| --- | --- | --- | --- |
| ResNet50 | 0.314497 | 0.064987 | 0.206638 |
| ResNet50V2 | 0.26157 | 0.069004 | 0.263807 |
| ResNet101 | 0.445634 | 0.124176 | 0.27865 |
| ResNet152 | 1.045938 | 0.180206 | 0.172291 |
| DenseNet121 | 0.222457 | 0.052469 | 0.235861 |
| DenseNet169 | 0.399349 | 0.063428 | 0.158828 |
| DenseNet201 | 0.735438 | 0.076999 | 0.104698 |
| EfficientNetB0 | 0.201541 | 1.444292 | 7.166244 |
| EfficientNetB1 | 0.233323 | 2.562888 | 10.98429 |
| EfficientNetB2 | 0.323979 | 3.626248 | 11.19285 |
| EfficientNetB3 | 0.480443 | 6.741603 | 14.03206 |
| EfficientNetB4 | 0.802941 | 16.58146 | 20.65091 |
| EfficientNetB5 | 1.313606 | 38.06942 | 28.98085 |
| EfficientNetB6 | 1.644098 | 70.11892 | 42.64887 |
| EfficientNetB7 | 2.609175 | 137.9669 | 52.87758 |
| EfficientNetV2B0 | 0.301169 | 2.857428 | 9.487789 |
| EfficientNetV2B1 | 0.410881 | 4.505161 | 10.96464 |
| EfficientNetV2B2 | 0.472283 | 6.518381 | 13.80185 |
| EfficientNetV2S | 1.109034 | 31.32271 | 28.24323 |
| EfficientNetV2M | 2.161369 | 93.17476 | 43.10914 |
| EfficientNetV2L | 4.672688 | 210.4929 | 45.0475 |
| MobileNet | 0.145611 | 2.070502 | 14.21941 |
| MobileNetV2 | N/A | N/A | N/A |
| MobileNetV3S | N/A | N/A | N/A |
| MobileNetV3L | N/A | N/A | N/A |
- PTQ 이후 추론 시간이 증가하는 모델은 모두 Separable Convolution 구조를 사용하는 모델임
    - Separable Conv를 사용하는 모델은 “Framework에서 양자화” 대상에서 제외하는 것이 좋을 것으로 판단됨

# References

- PTQ

[Post-training quantization  |  TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quantization#optimization_methods)

- QAT

[Quantization aware training in Keras example  |  TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

- Layer Fusing

[TensorFlow operation fusion  |  TensorFlow Lite](https://www.tensorflow.org/lite/models/convert/operation_fusion)

- Pruning with Keras

[Pruning in Keras example  |  TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)

- Pruning with XNNPACK

[Pruning for on-device inference w/ XNNPACK  |  TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_for_on_device_inference)

- TFLite inference

[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference)