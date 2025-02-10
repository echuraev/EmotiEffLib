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

cv::Mat EmotiEffLibRecognizerTorch::preprocess(const cv::Mat& img) { return cv::Mat(); }
} // namespace EmotiEffLib
