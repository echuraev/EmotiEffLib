#include "emotiefflib/backends/torch/facial_analysis.h"

#include <filesystem>

#include <xtensor/xadapt.hpp>

namespace fs = std::filesystem;

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& modelPath)
    : EmotiEffLibRecognizer(modelPath) {
    torch::jit::script::Module model = torch::jit::load(modelPath);
    models.push_back(model);
    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
}

EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& dirWithModels,
                                                       const std::string& modelName)
    : EmotiEffLibRecognizer(modelName) {
    fs::path featureExtractorPath(dirWithModels);
    featureExtractorPath /= modelName + ".pt";
    fs::path classifierPath(dirWithModels);
    classifierPath /= "classifier_" + modelName + ".pt";
    torch::jit::script::Module featureExtractor = torch::jit::load(featureExtractorPath);
    torch::jit::script::Module classifier = torch::jit::load(classifierPath);
    models.push_back(featureExtractor);
    models.push_back(classifier);
    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
}

xt::xarray<float> EmotiEffLibRecognizerTorch::preprocess(const cv::Mat& img) {
    cv::Mat resized, float_img, normalized;

    // Resize the image to (img_size, img_size)
    cv::resize(img, resized, cv::Size(imgSize_, imgSize_));

    // Convert to float and scale to [0, 1]
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Normalize using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::merge(channels, normalized);

    // Convert HWC OpenCV Mat to xtensor
    std::vector<float> hwcData;
    hwcData.assign((float*)normalized.datastart, (float*)normalized.dataend);

    return xt::adapt(hwcData, {imgSize_, imgSize_, 3});
}
} // namespace EmotiEffLib
