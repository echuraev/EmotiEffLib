"""
Trace torch models for usage in C++ Libtorch
"""

import os
import pathlib
from typing import Tuple

try:
    import torch
except ImportError:
    pass
try:
    import onnx
    from onnx import helper
except ImportError:
    pass

from emotiefflib.facial_analysis import get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def split_torch_model(model) -> Tuple[torch.nn.Module, torch.nn.Module]:
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
    model, classifier = split_torch_model(model)
    model_example = torch.rand(*input_shape)
    classifier_example = torch.rand(*classifier_shape)
    traced_script_module = torch.jit.trace(model, model_example)
    traced_script_classifier = torch.jit.trace(classifier, classifier_example)
    traced_script_module.save(model_out)
    traced_script_classifier.save(classifier_out)


def prepare_torch_models() -> None:
    """
    Trace torch models from EmotiEffLib.
    """
    models_dir = os.path.join(FILE_DIR, "affectnet_emotions")
    # model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    model_files = [f + ".pt" for f in get_model_list()]
    output_dir = os.path.join(FILE_DIR, "emotieffcpplib_prepared_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mf in model_files:
        inp = os.path.join(models_dir, mf)
        model_out = os.path.join(output_dir, mf)
        classifier_out = os.path.join(output_dir, "classifier_" + mf)
        if os.path.exists(model_out) and os.path.exists(classifier_out):
            print(f"SKIP {mf}")
        trace_model(inp, model_out, classifier_out)


def split_onnx_model(model_path: str, model_out: str, classifier_out: str) -> None:
    """
    Split an ONNX model into a feature extractor and a classifier.

    Args:
        model_path (str): Path to the input ONNX model.
        model_out (str): Path to save the modified model without the classifier.
        classifier_out (str): Path to save the extracted classifier as a separate ONNX model.

    Returns:
        None

    Raises:
        RuntimeError: If the model has no nodes or if the last classification node is unexpected.
    """
    print(f"Processing {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    # Ensure the model has at least one node
    if not graph.node:
        raise RuntimeError("Model has no nodes")

    # Extract last node (assumed to be the final layer)
    gemm_node = graph.node[-1]
    new_output_name = gemm_node.input[0]
    if gemm_node is None or len(gemm_node.input) < 3:
        raise RuntimeError("Unexpected gemm node!")
    weight_name = gemm_node.input[1]
    weight_tensor = next((t for t in graph.initializer if t.name == weight_name), None)
    classifier_weights = onnx.numpy_helper.to_array(weight_tensor) if weight_tensor else None

    # Remove the last node
    graph.node.remove(gemm_node)
    graph.output.remove(graph.output[0])
    new_output_shape = [None, classifier_weights.shape[1]]
    new_output = helper.make_tensor_value_info(
        new_output_name, onnx.TensorProto.FLOAT, new_output_shape
    )
    graph.output.append(new_output)

    # Model with the last node
    # Create a new model for the last layer
    new_graph = helper.make_graph(
        nodes=[gemm_node],
        name="classifier",
        inputs=[
            helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, None)
            for inp in gemm_node.input
        ],
        outputs=[helper.make_tensor_value_info(gemm_node.output[0], onnx.TensorProto.FLOAT, None)],
        initializer=[t for t in graph.initializer if t.name in gemm_node.input],
    )

    classifier = helper.make_model(new_graph, producer_name="onnx-classifier")

    onnx.save(model, model_out)
    onnx.save(classifier, classifier_out)


def prepare_onnx_models() -> None:
    """
    Prepare ONNX models from EmotiEffLib.
    """
    models_dir = os.path.join(FILE_DIR, "affectnet_emotions", "onnx")
    model_files = [f + ".onnx" for f in get_model_list()]
    output_dir = os.path.join(FILE_DIR, "emotieffcpplib_prepared_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mf in model_files:
        inp = os.path.join(models_dir, mf)
        model_out = os.path.join(output_dir, mf)
        classifier_out = os.path.join(output_dir, "classifier_" + mf)
        if os.path.exists(model_out) and os.path.exists(classifier_out):
            print(f"SKIP {mf}")
        split_onnx_model(inp, model_out, classifier_out)


if __name__ == "__main__":
    # pylint: disable=unused-import, import-outside-toplevel, redefined-outer-name, ungrouped-imports
    try:
        import torch

        prepare_torch_models()
    except ImportError:
        pass
    try:
        import onnx

        prepare_onnx_models()
    except ImportError:
        pass
