"""
Trace torch models for usage in C++ Libtorch
"""

import os
import pathlib

import torch

from emotiefflib.facial_analysis import get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def split_model(model):
    """
    Remove the classifier layer from a model and return the feature extractor and the classifier
    separately.

    Args:
        model: A PyTorch model with a `classifier` attribute.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]:
            - The modified model with the classifier replaced by an identity layer.
            - The original classifier layer.
    """
    last_layer = model.classifier
    model.classifier = torch.nn.Identity()
    return model, last_layer


def trace_model(torch_model: str, model_out: str, classifier_out: str) -> None:
    """
    Convert a PyTorch model into a TorchScript traced model and save it.

    Args:
        torch_model (str): Path to the input PyTorch model file.
        model_out (str): Path to save the traced TorchScript feature extraction model.
        classifier_out (str): Path to save the traced TorchScript classification model.

    Returns:
        None
    """
    print(f"Processing {torch_model}...")
    img_size = 224 if "_b0_" in torch_model else 260
    input_shape = (1, 3, img_size, img_size)
    model = torch.load(torch_model, map_location=torch.device("cpu"))
    if isinstance(model.classifier, torch.nn.Sequential):
        classifier_shape = (1, model.classifier[0].in_features)
    else:
        classifier_shape = (1, model.classifier.in_features)
    model, classifier = split_model(model)
    model_example = torch.rand(*input_shape)
    classifier_example = torch.rand(*classifier_shape)
    traced_script_module = torch.jit.trace(model, model_example)
    traced_script_classifier = torch.jit.trace(classifier, classifier_example)
    traced_script_module.save(model_out)
    traced_script_classifier.save(classifier_out)


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
        model_out = os.path.join(output_dir, mf)
        classifier_out = os.path.join(output_dir, "classifier_" + mf)
        trace_model(inp, model_out, classifier_out)


if __name__ == "__main__":
    trace_torch_models()
