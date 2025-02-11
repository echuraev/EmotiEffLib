#ifndef BACKENDS_TORCH_FACIAL_ANALYSIS_H
#define BACKENDS_TORCH_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

#include <torch/script.h>

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerTorch(const std::string& modelPath);
    EmotiEffLibRecognizerTorch(const std::string& dirWithModels, const std::string& modelName);
    EmotiEffLibRes precictEmotions(const cv::Mat& faceImg, bool logits = true) override;

private:
    xt::xarray<float> preprocess(const cv::Mat& img) override;

private:
    std::vector<torch::jit::script::Module> models;
};
} // namespace EmotiEffLib

#endif
