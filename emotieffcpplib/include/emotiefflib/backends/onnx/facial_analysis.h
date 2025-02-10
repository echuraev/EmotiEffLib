#ifndef BACKENDS_ONNX_FACIAL_ANALYSIS_H
#define BACKENDS_ONNX_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

namespace EmotiEffLib {
class EmotiEffLibRecognizerOnnx : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerOnnx(const std::string& modelPath);
    EmotiEffLibRecognizerOnnx(const std::string& dirWithModels, const std::string& modelName);

private:
    xt::xarray<float> preprocess(const cv::Mat& img) override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};
} // namespace EmotiEffLib

#endif
