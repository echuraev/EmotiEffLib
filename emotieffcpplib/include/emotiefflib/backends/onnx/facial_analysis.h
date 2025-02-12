#ifndef BACKENDS_ONNX_FACIAL_ANALYSIS_H
#define BACKENDS_ONNX_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

#include <onnxruntime_cxx_api.h>

namespace EmotiEffLib {
class EmotiEffLibRecognizerOnnx final : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerOnnx(const std::string& fullPipelineModelPath);
    EmotiEffLibRecognizerOnnx(const EmotiEffLibConfig& config);
    EmotiEffLibRes precictEmotions(const cv::Mat& faceImg, bool logits = true) override;

private:
    void initRecognizer(const std::string& modelPath) override;
    xt::xarray<float> preprocess(const cv::Mat& img) override;
    void configParser(const EmotiEffLibConfig& config) override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
    Ort::Env env_ = {ORT_LOGGING_LEVEL_WARNING, "EmotiEffLib"};
    std::vector<Ort::Session> models_;
};
} // namespace EmotiEffLib

#endif
