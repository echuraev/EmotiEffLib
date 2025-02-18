# EmotiEffLib C++ examples

## Building examples
To run the examples locally you need to do the following:
1. Build [EmotiEffCppLib](../../../emotieffcpplib) with Libtorch and ONNXRuntime.
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
