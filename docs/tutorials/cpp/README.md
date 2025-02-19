# EmotiEffLib C++ examples

## Building examples
To run the examples locally you need to do the following:
1. Build [EmotiEffCppLib](../../../emotieffcpplib) with Libtorch and ONNXRuntime. It is important to
   build EmotiEffCppLib with flags: `-DBUILD_TESTS=ON` and ` -DBUILD_SHARED_LIBS=ON` because we need
   to reuse one library which builds for tests and xeus-cling works with shared libraries.
2. Install [xeus-cling](https://github.com/jupyter-xeus/xeus-cling). Instruction how to build xeus-cling can be found [here](https://xeus-cling.readthedocs.io/en/latest/installation.html).
3. Prepare models for cpp runtime:
  ```
  python3 <EmotiEffLib_root>/models/prepare_models_for_emotieffcpplib.py
  ```
  After installing xeus-cling, you should be able to check available kernels and see `xcpp17` kernel:
  ```
  $ jupyter kernelspec list
  Available kernels:
    python3    /opt/anaconda3/envs/emotiefflib/share/jupyter/kernels/python3
    xcpp11     /opt/anaconda3/envs/emotiefflib/share/jupyter/kernels/xcpp11
    xcpp14     /opt/anaconda3/envs/emotiefflib/share/jupyter/kernels/xcpp14
    xcpp17     /opt/anaconda3/envs/emotiefflib/share/jupyter/kernels/xcpp17
  ```
4. Download and unpack test data:
  ```
  cd <EmotiEffLib_root>/tests
  ./download_test_data.sh
  tar -xzf data.tar.gz
  ```
5. Specify the following environment variables:
  ```
  export EMOTIEFFLIB_BUILD_DIR="<EmotiEffLib_root>/emotieffcpplib/build"
  export EMOTIEFFLIB_ROOT="<EmotiEffLib_root>"
  ```
