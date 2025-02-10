#include "emotiefflib/backends/torch/facial_analysis.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& modelPath)
    : EmotiEffLibRecognizer(modelPath) {
    torch::jit::script::Module model = torch::jit::load(modelPath);
    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
    auto x = torch::randn({1, 3, 224, 224});
    auto output = model.forward({x}).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl;
}

EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& dirWithModels,
                                                       const std::string& modelName)
    : EmotiEffLibRecognizer(modelName) {
    // torch::jit::script::Module model = torch::jit::load(modelPath);
    // bool isB0 = modelName_.find("_b0_") != std::string::npos;
    // imgSize_ = isB0 ? 224 : 260;
    // auto x = torch::randn({1, 3, 224, 224});
    // auto output = model.forward({x}).toTensor();
    // std::cout << "Output shape: " << output.sizes() << std::endl;
}

cv::Mat EmotiEffLibRecognizerTorch::preprocess(const cv::Mat& img) {
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

    return normalized;
}
} // namespace EmotiEffLib
