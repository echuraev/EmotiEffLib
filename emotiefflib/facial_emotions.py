"""
Facial emotions recognition implementation
"""

from __future__ import absolute_import, division, print_function

import os
import urllib.request
from abc import ABC, abstractmethod

import cv2
import numpy as np

try:
    import torch
    from torchvision import transforms
except ImportError:
    pass
try:
    import onnxruntime as ort
except ImportError:
    pass
from PIL import Image

# def get_path(model_name):
#    return '../../models/affectnet_emotions/'+model_name+'.pt'


def download_model(model_file, path_in_repo):
    """
    Returns local path to a model
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".emotiefflib")
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = (
            "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/"
            + path_in_repo
            + model_file
            + "?raw=true"
        )
        print("Downloading", model_file, "from", url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


def get_model_path_torch(model_name):
    """
    Returns local path to a torch model
    """
    model_file = model_name + ".pt"
    path_in_repo = "models/affectnet_emotions/"
    return download_model(model_file, path_in_repo)


def get_model_path_onnx(model_name):
    """
    Returns local path to an ONNX model
    """
    model_file = model_name + ".onnx"
    path_in_repo = "models/affectnet_emotions/onnx/"
    return download_model(model_file, path_in_repo)


def get_model_list():
    """
    Returns a list of available model names.

    These models are supported by HSEmoitonRecognizer.

    Returns:
        list of str: A list of model names.
    """
    return [
        "enet_b0_8_best_vgaf",
        "enet_b0_8_best_afew",
        "enet_b2_8",
        "enet_b0_8_va_mtl",
        "enet_b2_7",
    ]


def get_supported_engines():
    """
    Returns a list of supported inference engines.

    Returns:
        list of str: A list of inference engines.
    """
    return ["torch", "onnx"]


class EmotiEffLibRecognizerBase(ABC):
    """
    Abstract class for emotion recognizer classes
    """

    def __init__(self, model_name):
        self.is_mtl = "_mtl" in model_name
        if "_7" in model_name:
            self.idx_to_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

    @abstractmethod
    def _preprocess(self, img):
        """
        Preprocess input image
        """
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def extract_features(self, face_img):
        """
        Extract features from facial image
        """
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def predict_emotions(self, face_img, logits=True):
        """
        Predict emotions for facial image
        """
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def extract_multi_features(self, face_img_list):
        """
        Extract multi features from a sequence of facial images
        """
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def predict_multi_emotions(self, face_img_list, logits=True):
        """
        Predict emotions on a sequence of facial images
        """
        raise NotImplementedError("It should be implemented")


class EmotiEffLibRecognizerTorch(EmotiEffLibRecognizerBase):
    """
    EmotiEffLibRecognizer class
    """

    def __init__(self, model_name="enet_b0_8_best_vgaf", device="cpu"):
        super().__init__(model_name)
        self.device = device

        self.img_size = 224 if "_b0_" in model_name else 260

        path = get_model_path_torch(model_name)
        if device == "cpu":
            model = torch.load(path, map_location=torch.device("cpu"))
        else:
            model = torch.load(path)
        if isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias = model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights = model.classifier.weight.cpu().data.numpy()
            self.classifier_bias = model.classifier.bias.cpu().data.numpy()

        model.classifier = torch.nn.Identity()
        model = model.to(device)
        self.model = model.eval()

    def _preprocess(self, img):
        """
        Preprocess input image
        """
        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return test_transforms(Image.fromarray(img))

    def get_probab(self, features):
        """
        Returns probab
        """
        x = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias
        return x

    def extract_features(self, face_img):
        """
        Extract features from facial image
        """
        img_tensor = self._preprocess(face_img)
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_emotions(self, face_img, logits=True):
        """
        Predict emotions for facial image
        """
        features = self.extract_features(face_img)
        scores = self.get_probab(features)[0]
        if self.is_mtl:
            x = scores[:-2]
        else:
            x = scores
        pred = np.argmax(x)

        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2] = e_x
            else:
                scores = e_x
        return self.idx_to_class[pred], scores

    def extract_multi_features(self, face_img_list):
        """
        Extract multi features from a sequence of facial images
        """
        imgs = [self._preprocess(face_img) for face_img in face_img_list]
        features = self.model(torch.stack(imgs, dim=0).to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_multi_emotions(self, face_img_list, logits=True):
        """
        Predict emotions on a sequence of facial images
        """
        features = self.extract_multi_features(face_img_list)
        scores = self.get_probab(features)
        if self.is_mtl:
            preds = np.argmax(scores[:, :-2], axis=1)
        else:
            preds = np.argmax(scores, axis=1)
        if self.is_mtl:
            x = scores[:, :-2]
        else:
            x = scores

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_class[pred] for pred in preds], scores


class EmotiEffLibRecognizerOnnx(EmotiEffLibRecognizerBase):
    """
    EmotiEffLibRecognizer class
    """

    def __init__(self, model_name="enet_b0_8_best_vgaf"):
        super().__init__(model_name)

        if "mbf_" in model_name:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.img_size = 112
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            if "_b2_" in model_name:
                self.img_size = 260
            elif "ddamfnet" in model_name:
                self.img_size = 112
            else:
                self.img_size = 224

        path = get_model_path_onnx(model_name)
        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def _preprocess(self, img):
        """
        Preprocess input image
        """
        x = cv2.resize(img, (self.img_size, self.img_size)) / 255
        for i in range(3):
            x[..., i] = (x[..., i] - self.mean[i]) / self.std[i]
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]

    def extract_features(self, face_img):
        """
        Extract features from facial image
        """
        raise NotImplementedError("It should be implemented")

    def predict_emotions(self, face_img, logits=True):
        """
        Predict emotions for facial image
        """
        scores = self.ort_session.run(None, {"input": self._preprocess(face_img)})[0][0]
        if self.is_mtl:
            x = scores[:-2]
        else:
            x = scores
        pred = np.argmax(x)
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2] = e_x
            else:
                scores = e_x
        return self.idx_to_class[pred], scores

    def extract_multi_features(self, face_img_list):
        """
        Extract multi features from a sequence of facial images
        """
        raise NotImplementedError("It should be implemented")

    def predict_multi_emotions(self, face_img_list, logits=True):
        """
        Predict emotions on a sequence of facial images
        """
        imgs = np.concatenate([self._preprocess(face_img) for face_img in face_img_list], axis=0)
        scores = self.ort_session.run(None, {"input": imgs})[0]
        if self.is_mtl:
            preds = np.argmax(scores[:, :-2], axis=1)
        else:
            preds = np.argmax(scores, axis=1)
        if self.is_mtl:
            x = scores[:, :-2]
        else:
            x = scores

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_class[pred] for pred in preds], scores


# pylint: disable=invalid-name
def EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_vgaf", device="cpu"):
    """
    Creates EmotiEffLibRecognizer instance.
    """
    # pylint: disable=unused-import, import-outside-toplevel, redefined-outer-name
    if engine not in get_supported_engines():
        raise ValueError("Unsupported engine specified")
    if engine == "torch":
        try:
            import torch
            from torchvision import transforms
        except ImportError as e:
            raise ImportError("Looks like torch module is not installed: ", e) from e
        return EmotiEffLibRecognizerTorch(model_name, device)
    # ONNX
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Looks like torch module is not installed: ", e) from e
    return EmotiEffLibRecognizerOnnx(model_name)
