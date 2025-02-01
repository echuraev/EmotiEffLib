"""
Pytests to check facial expression recognition functionality
"""

import os
import pathlib

import cv2
import pytest
import torch
from facenet_pytorch import MTCNN

from hsei.facial_emotions import HSEmotionRecognizer, get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize("model_name", get_model_list())
def test_one_image(model_name):
    """
    Simple test with one image
    """
    if model_name == "enet_b0_8_va_mtl":
        pytest.xfail("This model gives incorrect prediction")
    input_file = os.path.join(FILE_DIR, "..", "test_images", "20180720_174416.jpg")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

    def detect_face(frame):
        # pylint: disable=unbalanced-tuple-unpacking
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    frame_bgr = cv2.imread(input_file)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    bounding_boxes = detect_face(frame)

    emotions = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        face_img = frame[y1:y2, x1:x2, :]
        emotion, _ = fer.predict_emotions(face_img, logits=True)
        emotions.append(emotion)
    exp_emotions = ["Happiness", "Anger", "Fear"]

    assert emotions == exp_emotions
