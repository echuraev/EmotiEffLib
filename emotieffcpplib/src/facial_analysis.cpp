#include "emotiefflib/facial_analysis.h"
#include "emotiefflib/backends/onnx/facial_analysis.h"
#include "emotiefflib/backends/torch/facial_analysis.h"

#include <filesystem>

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

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

std::vector<std::string> getSupportedModels() {
    return {
        "enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8", "enet_b0_8_va_mtl", "enet_b2_7",
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

EmotiEffLibRes EmotiEffLibRecognizer::processScores(const xt::xarray<float>& score, bool logits) {
    xt::xarray<float> x;
    xt::xarray<float> scores = score;
    // Select relevant part of the scores based on is_mtl
    if (isMtl_) {
        x = xt::view(scores, xt::all(),
                     xt::range(xt::placeholders::_, -2)); // Equivalent to scores[:, :-2]
    } else {
        x = scores;
    }

    // Compute predictions
    auto preds = xt::argmax(x, 1);

    // Apply softmax if logits is false
    if (!logits) {
        xt::xarray<float> max_x = xt::amax(x, {1}, xt::evaluation_strategy::immediate);
        xt::xarray<float> e_x = xt::exp(x - max_x);
        e_x /= xt::sum(e_x, {1}, xt::evaluation_strategy::immediate);

        if (isMtl_) {
            xt::view(scores, xt::all(), xt::range(xt::placeholders::_, -2)) =
                e_x; // Modify in-place
        } else {
            scores = e_x; // Replace scores with softmaxed values
        }
    }

    // Convert predictions to emotion class names
    EmotiEffLibRes res;
    for (auto pred : preds) {
        res.labels.push_back(idxToEmotionClass_[pred]);
    }
    res.scores = scores;
    return res;
}

} // namespace EmotiEffLib
