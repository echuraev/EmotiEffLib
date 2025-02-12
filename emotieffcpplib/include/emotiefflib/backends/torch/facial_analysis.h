#ifndef BACKENDS_TORCH_FACIAL_ANALYSIS_H
#define BACKENDS_TORCH_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

#include <torch/script.h>

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch final : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerTorch(const std::string& fullPipelineModelPath);
    EmotiEffLibRecognizerTorch(const EmotiEffLibConfig& config);
    EmotiEffLibRes precictEmotions(const cv::Mat& faceImg, bool logits = true) override;

private:
    void initRecognizer(const std::string& modelPath) override;
    xt::xarray<float> preprocess(const cv::Mat& img) override;
    void configParser(const EmotiEffLibConfig& config) override;

private:
    std::vector<torch::jit::script::Module> models_;
};
} // namespace EmotiEffLib

#endif
