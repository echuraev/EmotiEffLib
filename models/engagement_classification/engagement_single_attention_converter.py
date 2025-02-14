"""
Engagement classification model converter to ONNX and torch
"""

# pylint: disable=no-name-in-module,import-error
import os
import pathlib

import numpy as np
import onnx
import torch
from onnx2pytorch import ConvertModel
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Input,
    Lambda,
    Multiply,
    Permute,
    RepeatVector,
    Reshape,
)
from tensorflow.keras.models import Model

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def _single_attention_model(n_classes, weights, feature_vector_dim, samples=None):
    """
    Build a single attention classification model.

    Args:
        n_classes (int): The number of output classes.
        weights (str): Path to the pre-trained model weights.
        feature_vector_dim (int): The dimensionality of the input feature vectors.
                                  Must be 2560; otherwise, a ValueError is raised.
        samples (int, optional): The number of samples in the input sequence. Defaults to None.

    Returns:
        Model: A Keras model implementing attention-based classification.

    Raises:
        ValueError: If the feature vector dimension is not 2560.
    """
    if feature_vector_dim != 2560:
        raise ValueError("Unsupported feature vector dim. Maybe you use unsupported model.")
    inputs = Input(
        shape=(samples, feature_vector_dim), name="image_set"
    )  # (batch, samples, features)
    e = Dense(1, activation="linear", name="e")(inputs)
    e = Reshape([samples], name="alignment")(e)
    alpha = Activation("softmax", name="alpha")(e)

    alpha_repeated = Permute([2, 1], name="alpha_repeated")(
        RepeatVector(feature_vector_dim, name="repeat")(alpha)
    )

    c = Multiply(name="c")([inputs, alpha_repeated])
    x = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(feature_vector_dim,), name="context")(
        c
    )

    x = Dense(units=512, activation="relu", name="hidden_FC")(x)  # (batch, units) #128 64

    pred = Dense(n_classes, activation="softmax")(x)  # (batch, classes)
    model = Model(inputs=inputs, outputs=pred)

    model.load_weights(weights)

    return model


def get_engagement_model(feature_vector_dim: int, number_of_frames: int):
    """
    Load and return the engagement classification model.

    Args:
        feature_vector_dim (int): The dimensionality of the input feature vectors.
        number_of_frames (int): The number of frames in the input sequence.

    Returns:
        Model: A Keras model for engagement classification.
    """
    weights_path = os.path.join(FILE_DIR, "engagement_single_attention.h5")
    model = _single_attention_model(2, weights_path, feature_vector_dim, number_of_frames)
    model.summary()
    input_data = np.random.random((1, 128, 2560))
    _ = model(input_data)
    return model


def engagement_to_onnx_converter(onnx_filename: str) -> onnx.ModelProto:
    """
    Converts and exports an engagement recognition model to ONNX format if it does not already
    exist.

    Args:
        onnx_filename (str): The file path where the ONNX model will be saved.

    Returns:
        onnx.ModelProto: The loaded ONNX model after validation.
    """
    if not os.path.exists(onnx_filename):
        model = get_engagement_model(2560, 128)

        model.export(onnx_filename, format="onnx")
    else:
        print("SKIP Engagement onnx model creation")

    loaded_model = onnx.load(onnx_filename)
    onnx.checker.check_model(loaded_model)
    print("The ONNX engagement model is valid.")
    return loaded_model


def engagement_to_torch_converter(onnx_model: onnx.ModelProto, torch_filename: str) -> None:
    """
    Converts an ONNX engagement recognition model to a TorchScript model and saves it.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to be converted.
        torch_filename (str): The file path where the TorchScript model will be saved.

    Returns:
        None
    """
    if os.path.exists(torch_filename):
        print("SKIP Engagement torch model creation")
    pytorch_model = ConvertModel(onnx_model)
    pytorch_model.eval()
    input_shape = (1, 128, 2560)
    model_example = torch.rand(*input_shape)
    traced_script_module = torch.jit.trace(pytorch_model, model_example)
    traced_script_module.save(torch_filename)
