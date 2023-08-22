from setuptools import setup, find_packages

setup(
    name="quantization_tf",
    version="0.0.1",
    description="model quantization module for Tensorflow",
    author="swim",
    author_email="swim@surromind.ai",
    url="https://gitlab.surromind.ai/smartedgeai/quantization_tf",
    install_requires=[
        "tensorflow==2.10.1",
        "protobuf==3.19.6",
        "tensorflow-model-optimization==0.7.5",
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
