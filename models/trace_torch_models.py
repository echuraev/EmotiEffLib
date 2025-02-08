"""
Trace torch models for usage in C++ Libtorch
"""

import os
import pathlib

import torch

from emotiefflib.facial_analysis import get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def trace_model(torch_model: str, output: str) -> None:
    """
    Convert a PyTorch model into a TorchScript traced model and save it.

    Args:
        torch_model (str): Path to the input PyTorch model file.
        output (str): Path to save the traced TorchScript model.

    Returns:
        None
    """
    print(f"Processing {torch_model}...")
    img_size = 224 if "_b0_" in torch_model else 260
    model = torch.load(torch_model, map_location=torch.device("cpu"))
    example = torch.rand(1, 3, img_size, img_size)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(output)


def trace_torch_models() -> None:
    """
    Trace torch models from EmotiEffLib.
    """
    models_dir = os.path.join(FILE_DIR, "affectnet_emotions")
    # model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    model_files = [f + ".pt" for f in get_model_list()]
    output_dir = os.path.join(FILE_DIR, "traced_affectnet_emotions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mf in model_files:
        inp = os.path.join(models_dir, mf)
        out = os.path.join(output_dir, mf)
        trace_model(inp, out)


if __name__ == "__main__":
    trace_torch_models()
