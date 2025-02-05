"""
Pytests to check facial expression recognition functionality
"""

import os
import pathlib

import cv2
import numpy as np
import pytest
import torch
from facenet_pytorch import MTCNN

from emotiefflib.facial_emotions import EmotiEffLibRecognizer, get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def recognize_faces(image_path, device):
    """
    Detects faces in the given image and returns the facial images cropped from the original.

    This function reads an image from the specified path, detects faces using the MTCNN
    face detection model, and returns a list of cropped face images.

    Args:
        image_path (str): The path to the image file in which faces need to be detected.
        device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

    Returns:
        list: A list of numpy arrays, representing a cropped face image from the original image.

    Example:
        faces = recognize_faces('image.jpg', 'cuda')
        # faces contains the cropped face images detected in 'image.jpg'.
    """
    frame_bgr = cv2.imread(image_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def detect_face(frame):
        # pylint: disable=unbalanced-tuple-unpacking
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images


@pytest.mark.parametrize("model_name", get_model_list())
@pytest.mark.parametrize("engine", ["torch", "onnx"])
def test_one_image_prediction(model_name, engine):
    """
    Simple test with one image
    """
    if model_name == "enet_b0_8_va_mtl" or (
        engine == "onnx" and model_name == "enet_b0_8_best_afew"
    ):
        pytest.xfail("This model gives incorrect prediction")
    input_file = os.path.join(FILE_DIR, "..", "test_images", "20180720_174416.jpg")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    facial_images = recognize_faces(input_file, device)

    fer = EmotiEffLibRecognizer(engine=engine, model_name=model_name, device=device)

    emotions = []
    for face_img in facial_images:
        emotion, _ = fer.predict_emotions(face_img, logits=True)
        emotions.append(emotion)
    exp_emotions = ["Happiness", "Anger", "Fear"]

    assert emotions == exp_emotions


@pytest.mark.parametrize("model_name", get_model_list())
@pytest.mark.parametrize("engine", ["torch", "onnx"])
def test_one_image_multi_prediction(model_name, engine):
    """
    Simple test with one image and predict_multi_emotions API
    """
    if model_name == "enet_b0_8_va_mtl" or (
        engine == "onnx" and model_name == "enet_b0_8_best_afew"
    ):
        pytest.xfail("This model gives incorrect prediction")
    input_file = os.path.join(FILE_DIR, "..", "test_images", "20180720_174416.jpg")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    facial_images = recognize_faces(input_file, device)

    fer = EmotiEffLibRecognizer(engine=engine, model_name=model_name, device=device)

    emotions, _ = fer.predict_multi_emotions(facial_images, logits=True)
    exp_emotions = ["Happiness", "Anger", "Fear"]

    assert emotions == exp_emotions


@pytest.mark.parametrize("model_name", get_model_list())
def test_one_image_features(model_name):
    """
    Compare feature vectors for ONNX and Torch implementations
    """
    input_file = os.path.join(FILE_DIR, "..", "test_images", "20180720_174416.jpg")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    facial_images = recognize_faces(input_file, device)

    fer_onnx = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
    fer_torch = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)

    for face_img in facial_images:
        features_onnx = fer_onnx.extract_features(face_img)
        features_torch = fer_torch.extract_features(face_img)
        assert len(features_onnx) == len(features_torch)


@pytest.mark.parametrize("model_name", get_model_list())
def test_one_image_multi_features(model_name):
    """
    Compare feature vectors for ONNX and Torch implementations with extract_multi_features API
    """
    input_file = os.path.join(FILE_DIR, "..", "test_images", "20180720_174416.jpg")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    facial_images = recognize_faces(input_file, device)

    fer_onnx = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
    fer_torch = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)

    features_onnx = np.array(fer_onnx.extract_multi_features(facial_images))
    features_torch = np.array(fer_torch.extract_multi_features(facial_images))
    assert features_onnx.shape[0] == 3
    assert features_onnx.shape == features_torch.shape
