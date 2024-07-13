from setuptools import setup, find_packages

setup(
    name="quantization_tf",
    version="0.0.1",
    description="model quantization module for Tensorflow",
    author="swim",
    author_email="angseung@vueron.com",
    url="https://https://github.com/angseung/quantization_tf",
    install_requires=[
        "tensorflow",
        "protobuf",
        "tensorflow-model-optimization",
        "matplotlib",
        "tf2onnx",
        "tflite2onnx",
        "onnxruntime",
        "onnx",
    ],
    packages=find_packages(exclude=[]),
    package_data={
        "quantization_tf": ["main/*.py"],
    },
    include_package_data=True,
    keywords=["quantization", "pypi"],
    python_requires=">=3.9",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
