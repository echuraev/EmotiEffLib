#ifndef BACKENDS_ONNX_FACIAL_ANALYSIS_H
#define BACKENDS_ONNX_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

#include <onnxruntime_cxx_api.h>

namespace EmotiEffLib {
class EmotiEffLibRecognizerOnnx final : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerOnnx(const std::string& fullPipelineModelPath);
    EmotiEffLibRecognizerOnnx(const EmotiEffLibConfig& config);
    xt::xarray<float> extractFeatures(const cv::Mat& faceImg) override;
    EmotiEffLibRes classifyEmotions(const xt::xarray<float>& features, bool logits = true) override;
    EmotiEffLibRes precictEmotions(const cv::Mat& faceImg, bool logits = true) override;

private:
    void initRecognizer(const std::string& modelPath) override;
    xt::xarray<float> preprocess(const cv::Mat& img) override;
    void configParser(const EmotiEffLibConfig& config) override;
    std::vector<Ort::Value> modelRunWrapper(int modelIdx, const std::vector<Ort::Value>& inputs);

private:
    std::vector<float> mean_;
    std::vector<float> std_;
    Ort::Env env_ = {ORT_LOGGING_LEVEL_WARNING, "EmotiEffLib"};
    std::vector<Ort::Session> models_;
};
} // namespace EmotiEffLib

#endif
