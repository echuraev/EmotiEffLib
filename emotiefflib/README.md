# EmotiEffLib Python implementation

EmotiEffLib is a Python library designed for efficient emotion and engagement
recognition in images and videos. It supports multiple inference engines,
including Torch and ONNX Runtime.

## Installing

```
    python setup.py install
```

It is also possible to install it via pip:
```
    pip install emotiefflib
```

## Usage
Examples of usage EmotiEffLib you can find in the [python examples section](../docs/tutorials/python/README.md).

## Running tests
Pytest framework is used for testing EmotiEffLib. To run the tests, you need to
follow the following steps:
1. Install dependencies:
  ```sh
  pip install -r tests/requirements.txt
  ```
2. Download testing data:
  ```sh
  cd <emotiefflib_root>/tests
  ./download_test_data.sh
  tar -xzf data.tar.gz
  cd -
  ```
2. Run tests:
  ```sh
  pytest <emotiefflib_root>/tests
  ```

## License

The code of EmotiEffCppLib Library is released under the Apache-2.0 License. There is no limitation for both academic and commercial usage.
