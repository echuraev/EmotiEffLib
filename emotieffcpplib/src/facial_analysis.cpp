#include "emotiefflib/facial_analysis.h"
#include "emotiefflib/backends/onnx/facial_analysis.h"
#include "emotiefflib/backends/torch/facial_analysis.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace EmotiEffLib {
std::vector<std::string> getAvailableBackends() {
    return {
#ifdef WITH_TORCH
        "torch",
#endif
#ifdef WITH_ONNX
        "onnx",
#endif
    };
}

std::unique_ptr<EmotiEffLibRecognizer>
EmotiEffLibRecognizer::createInstance(const std::string& engine, const std::string& modelPath) {
    auto backends = getAvailableBackends();
    auto it = std::find(backends.begin(), backends.end(), engine);
    if (it == backends.end()) {
        throw std::runtime_error("This backend (" + engine +
                                 ") is not supported. Please check your EmotiEffLib build.");
    }
    if (engine == "torch")
        return std::make_unique<EmotiEffLibRecognizerTorch>(modelPath);
    return std::make_unique<EmotiEffLibRecognizerOnnx>(modelPath);
}

std::unique_ptr<EmotiEffLibRecognizer>
EmotiEffLibRecognizer::createInstance(const std::string& engine, const std::string& dirWithModels,
                                      const std::string& modelName) {
    auto backends = getAvailableBackends();
    auto it = std::find(backends.begin(), backends.end(), engine);
    if (it == backends.end()) {
        throw std::runtime_error("This backend (" + engine +
                                 ") is not supported. Please check your EmotiEffLib build.");
    }
    if (engine == "torch")
        return std::make_unique<EmotiEffLibRecognizerTorch>(dirWithModels, modelName);
    return std::make_unique<EmotiEffLibRecognizerOnnx>(dirWithModels, modelName);
}

EmotiEffLibRecognizer::EmotiEffLibRecognizer(const std::string& modelPath) {
    modelName_ = fs::path(modelPath).filename().string();
    isMtl_ = modelName_.find("_mtl") != std::string::npos;
    bool is7 = modelName_.find("_7") != std::string::npos;
    if (is7) {
        idxToEmotionClass_.resize(7);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Disgust";
        idxToEmotionClass_[2] = "Fear";
        idxToEmotionClass_[3] = "Happiness";
        idxToEmotionClass_[4] = "Neutral";
        idxToEmotionClass_[5] = "Sadness";
        idxToEmotionClass_[6] = "Surprise";
    } else {
        idxToEmotionClass_.resize(8);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Contempt";
        idxToEmotionClass_[2] = "Disgust";
        idxToEmotionClass_[3] = "Fear";
        idxToEmotionClass_[4] = "Happiness";
        idxToEmotionClass_[5] = "Neutral";
        idxToEmotionClass_[6] = "Sadness";
        idxToEmotionClass_[7] = "Surprise";
    }
}

} // namespace EmotiEffLib
