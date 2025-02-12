#ifndef BACKENDS_TORCH_FACIAL_ANALYSIS_H
#define BACKENDS_TORCH_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

#include <torch/script.h>

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch final : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerTorch(const std::string& fullPipelineModelPath);
    EmotiEffLibRecognizerTorch(const EmotiEffLibConfig& config);
    xt::xarray<float> extractFeatures(const cv::Mat& faceImg) override;
    EmotiEffLibRes classifyEmotions(const xt::xarray<float>& features, bool logits = true) override;
    EmotiEffLibRes predictEmotions(const cv::Mat& faceImg, bool logits = true) override;

private:
    void initRecognizer(const std::string& modelPath) override;
    xt::xarray<float> preprocess(const cv::Mat& img) override;
    void configParser(const EmotiEffLibConfig& config) override;

private:
    std::vector<torch::jit::script::Module> models_;
};
} // namespace EmotiEffLib

#endif
