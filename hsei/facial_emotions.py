"""
Facial emotions recognition implementation
"""

from __future__ import absolute_import, division, print_function

import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# def get_path(model_name):
#    return '../../models/affectnet_emotions/'+model_name+'.pt'


def get_model_path(model_name):
    """
    Returns local path to a model
    """
    model_file = model_name + ".pt"
    cache_dir = os.path.join(os.path.expanduser("~"), ".hsemotion")
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = (
            "https://github.com/HSE-asavchenko/face-emotion-recognition/"
            "blob/main/models/affectnet_emotions/" + model_file + "?raw=true"
        )
        print("Downloading", model_name, "from", url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


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


class HSEmotionRecognizer:
    """
    HSEmotionRecognizer class
    """

    # supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8,
    # enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name="enet_b0_8_best_vgaf", device="cpu"):
        self.device = device
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

        self.img_size = 224 if "_b0_" in model_name else 260
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        path = get_model_path(model_name)
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
        print(path, self.test_transforms)

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
        img_tensor = self.test_transforms(Image.fromarray(face_img))
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
        imgs = [self.test_transforms(Image.fromarray(face_img)) for face_img in face_img_list]
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
